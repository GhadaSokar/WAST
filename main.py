import os
import sys
from xmlrpc.client import boolean
sys.path.append(os.getcwd())
import time
import logging
import copy
import shutil

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import numpy as np

from test import svm_test
from utils import get_data, get_dataset_loader
from dst import dst_FS
import models
from argparser import get_parser

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def setup_logger(args):
    global logger
    if logger == None:
        logger = logging.getLogger()
    else:  # wish there was a logger.close()
        for handler in logger.handlers[:]:  # make a copy of the list
            logger.removeHandler(handler)
    args_copy = copy.deepcopy(args)
    # copy to get a clean hash
    # use the same log file hash if iterations or verbose are different
    # these flags do not change the results
    args_copy.iters = 1
    args_copy.verbose = False
    args_copy.log_interval = 1
    log_path = './logs/{0}.log'.format('ae')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def print_and_log(msg):
    global logger
    print(msg)
    logger.info(msg)

# calculate the reconstruction loss
def evaluate_mse(model, device, test_loader):
    model.eval()
    test_loss = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.t = target
            output = model(data)
            test_loss += F.mse_loss(output, torch.reshape(data, (data.shape[0], data.shape[2]*data.shape[3]))).item()
            n += target.shape[0]
    test_loss /= float(n)
    return test_loss

# calculate the classification accuracy on the selected features
def evaluate_fs(args, K, model, device, test_loader, masks, Input_IMP, train_X, train_y, test_X, test_y):
    model.eval()
    test_loss = 0
    n = 0
    print('K',K)
    with torch.no_grad():
        # calculate the reconstruction loss on test data
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model.t = target
            output = model(data)
            test_loss += F.mse_loss(output, torch.reshape(data, (data.shape[0], data.shape[2]*data.shape[3]))).item()

        # calculate the classification accuracy
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == 'fc1.weight':
                    # identify the best imformative features
                    if args.strength == 'weight': # the criteria used by QS baseline 
                        strength = (masks[name]*param).abs().sum(dim=0)
                    elif args.strength == 'IMP': # the criteria used of WAST
                        strength = Input_IMP 
                    values, indices = strength.topk(K)
                    selected_features = indices.cpu().detach().numpy()
                    train_X_new = train_X[:, selected_features]
                    # learn svm classifier using the K selected features
                    SVCacc = svm_test(train_X_new, train_y, test_X, test_y, selected_features)
                    print_and_log('SVCacc = {:.4f}'.format(SVCacc))
            n += target.shape[0]
    test_loss /= float(n)
    return test_loss, SVCacc
    
def train(args, model, device, train_loader, optimizer, epoch, FS_core):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        noisy_data = data + 0.2 * (torch.randn(data.shape).to(device)) # proposed by QS
        output = model(noisy_data)
        input = torch.reshape(data, (data.shape[0], data.shape[2]*data.shape[3]))
        loss = F.mse_loss(output, input)
        loss.backward()
        optimizer.step()
        FS_core.apply_mask_on_weights() 
        # Neuron importance (1) senstivity of neuron to loss
        diff = abs(input - output) 
        mean_diff = torch.mean(diff, dim=0) 
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Neuron importance (2) sum of the magnitite of the connected weights in the input and output layers
                if name == 'fc1.weight':
                    strength_input = (FS_core.mask[name]*param.detach()).abs().sum(dim=0)
                if name == 'fc2.weight':
                    strength_output = (FS_core.mask[name]*param.detach()).abs().sum(dim=1)

        factor = args.lamda                 
        FS_core.layers_importance['0'] += (factor*(mean_diff.detach()) + (1-factor)*strength_input) #input layer
        FS_core.layers_importance['2'] += (factor*(mean_diff.detach()) + (1-factor)*strength_output) #output layer

        if batch_idx % args.log_interval == 0:
            print_and_log('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))
            sys.stdout.flush()

        # update mask every batch update
        if args.update_batch:
            FS_core.update_mask(args.rmv_strategy, args.add_strategy)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main():
    parser = get_parser()
    args = parser.parse_args()  
    print("*******************************************************")
    setup_logger(args)
    print_and_log(args)
    print("*******************************************************")
    print_and_log(torch.cuda.is_available())
    print_and_log(torch.cuda.device_count())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device", device) 
    print("*******************************************************")

    # use 5 random seeds
    seeds = [0, 1, 2, 3, 4]
    print(seeds)

    # log accuracy at epochs 1, 5, 10
    svm_acc_1 = {} 
    svm_acc_5 = {} 
    svm_acc_10 = {} 
    # log accuracy through training for k = 50
    svm_acc_k50 = np.zeros((len(seeds), args.epochs))

    for seed_idx in range(len(seeds)):
        seed_everything(seeds[seed_idx])
        # loading data 
        train_loader, valid_loader, test_loader, input_size = get_dataset_loader(args)
        # model
        model = models.AE(args.data, input_size, args.nhidden).to(device)
        #WAST logic
        FS_core = dst_FS(model, device, args.alpha, args.density, args.hidden_IMP)

        # optimizer
        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        ks = [25, 50, 75, 100, 150, 200]
        if args.data == 'madelon':
            ks = [20]
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            # perform one training epoch
            train(args, model, device, train_loader, optimizer, epoch, FS_core)
            if epoch == 1 or epoch==10 or epoch==args.eval_epoch: # if True to log at each epoch 
                print("Epoch = {} , Evaluating feature selection".format(str(epoch)))
                train_X, train_y, test_X, test_y, _ = get_data(args)
                for ki in range(len(ks)):
                    ## evaluate classification accuracy
                    _, SVCacc = evaluate_fs(args, ks[ki], model, device, test_loader, FS_core.mask, FS_core.layers_importance['0'], train_X, train_y, test_X, test_y)
                    ## logging accuracy
                    if ks[ki] == 50 or ks[ki] == 20:   
                      svm_acc_k50[seed_idx][epoch-1] =  SVCacc  

                    if epoch==1:
                        if seed_idx == 0:
                            svm_acc_1[str(ks[ki])]= np.zeros(len(seeds))
                        svm_acc_1[str(ks[ki])][seed_idx] = SVCacc 
                    elif epoch==5:
                        if seed_idx == 0:
                            svm_acc_5[str(ks[ki])]= np.zeros(len(seeds))
                        svm_acc_5[str(ks[ki])][seed_idx] = SVCacc 
                    elif epoch == 10:
                        if seed_idx == 0:
                            svm_acc_10[str(ks[ki])]= np.zeros(len(seeds))
                        svm_acc_10[str(ks[ki])][seed_idx] = SVCacc 

            if args.valid_split > 0.0:
                print("Loss (valid) = ", evaluate_mse(model, device, valid_loader))
            else:
                print("Loss (test) = ", evaluate_mse(model, device, test_loader))

            # update mask every epoch (for QS baseline)
            if args.update_batch == False and epoch < args.epochs:
                FS_core.update_mask(args.rmv_strategy, args.add_strategy)

            print_and_log('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))

    # steps for k = 50
    print(np.mean(svm_acc_k50,axis=0))
    print(np.std(svm_acc_k50,axis=0))
    # accuracy for each K at epochs 1, 5, 10
    if args.data == 'madelon':
        print_and_log('Epoch1 SVCacc 20 = {:.4f}'.format(np.mean(svm_acc_1['20'])))
        print_and_log('Epoch5 SVCacc 20 = {:.4f}'.format(np.mean(svm_acc_5['20'])))
        print_and_log('Epoch10 SVCacc 20 = {:.4f}'.format(np.mean(svm_acc_10['20'])))
        print_and_log('Epoch1 SVCacc 20 = {:.4f}'.format(np.std(svm_acc_1['20'])))
        print_and_log('Epoch5 SVCacc 20 = {:.4f}'.format(np.std(svm_acc_5['20'])))
        print_and_log('Epoch10 SVCacc 20 = {:.4f}'.format(np.std(svm_acc_10['20'])))
    else:
        print_and_log('Epoch1 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.mean(svm_acc_1['25']), np.mean(svm_acc_1['50']), np.mean(svm_acc_1['75']), np.mean(svm_acc_1['100']), np.mean(svm_acc_1['150']), np.mean(svm_acc_1['200'])))
        print_and_log('Epoch5 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.mean(svm_acc_5['25']), np.mean(svm_acc_5['50']), np.mean(svm_acc_5['75']), np.mean(svm_acc_5['100']), np.mean(svm_acc_5['150']), np.mean(svm_acc_5['200'])))
        print_and_log('Epoch10 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.mean(svm_acc_10['25']), np.mean(svm_acc_10['50']), np.mean(svm_acc_10['75']), np.mean(svm_acc_10['100']), np.mean(svm_acc_10['150']), np.mean(svm_acc_10['200'])))
        print_and_log('Epoch1 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.std(svm_acc_1['25']), np.std(svm_acc_1['50']), np.std(svm_acc_1['75']), np.std(svm_acc_1['100']), np.std(svm_acc_1['150']), np.std(svm_acc_1['200'])))
        print_and_log('Epoch5 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.std(svm_acc_5['25']), np.std(svm_acc_5['50']), np.std(svm_acc_5['75']), np.std(svm_acc_5['100']), np.std(svm_acc_5['150']), np.std(svm_acc_5['200'])))
        print_and_log('Epoch10 SVCacc 25 = {:.4f}, 50 {:.4f}, 75 {:.4f}, 100 = {:.4f}, 150 = {:.4f}, 200 = {:.4f}'.format(np.std(svm_acc_10['25']), np.std(svm_acc_10['50']), np.std(svm_acc_10['75']), np.std(svm_acc_10['100']), np.std(svm_acc_10['150']), np.std(svm_acc_10['200'])))
    
if __name__ == '__main__':
    main()
