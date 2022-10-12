from __future__ import print_function

import torch
import numpy as np
import copy

from torch import detach
class dst_FS():
    def __init__(self, model, device, alpha, density, hidden_IMP):
        self.model = model
        self.device = device
        self.prune_rate = alpha
        self.density = density
        self.inf = 99999
        self.mask = {}   
        self.layers_importance =  {}
        self.hidden_IMP = hidden_IMP
        self.init_layer_importance()
        self.init_masks()
        self.apply_mask_on_weights()

    def init_layer_importance(self):
        #### index of layer Importance is hard-coded for single hidden layer of WAST
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:
                    if idx==1 and self.hidden_IMP==False: ### neurons of hidden layer is equally important
                        self.layers_importance[str(idx)] = torch.ones(param.shape[1]).to(self.device)
                    else:
                        self.layers_importance[str(idx)] = torch.zeros(param.shape[1]).to(self.device)
                    last_dim = param.shape[0]
                    idx+=1
        
        self.layers_importance[str(idx)] = torch.zeros(last_dim).to(self.device)

    def init_masks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:                    
                    noRows = param.data.shape[0]
                    noCols = param.data.shape[1]
                    self.mask[name] =  (torch.rand(param.data.shape) < self.density).float().data.to(self.device)                    
                    noParameters = torch.sum(self.mask[name])
                    sparsity = 1-noParameters/(noRows * noCols)
                    print("Sparse Initialization ",": Density ",self.density,"; Sparsity ",sparsity,"; NoParameters ",noParameters,"; NoRows ",noRows,"; NoCols ",noCols,"; NoDenseParam ",noRows*noCols)
                    print (" OutDegreeBottomNeurons %.2f ± %.2f; InDegreeTopNeurons %.2f ± %.2f" % (torch.mean(self.mask[name].sum(axis=1)),torch.std(self.mask[name].sum(axis=1)),torch.mean(self.mask[name].sum(axis=0)),torch.std(self.mask[name].sum(axis=0))))

    # ensure same sparsity after weight update
    def apply_mask_on_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1: 
                    param.data = param.data*self.mask[name].to(self.device)

    # adapt the sparse topology
    def update_mask(self, rmv_strategy, add_strategy):
        # drop phase
        if rmv_strategy == 'magnitute':
            self.rmv_magnitute()
        elif rmv_strategy == 'rmv_IMP':
            self.rmv_IMP()

        # grow phase
        if add_strategy == 'random':
            self.add_random()
        elif add_strategy == 'add_IMP':
            self.add_IMP()

    # Drop. connection importance is based on its magnitute (QS baseline)
    def rmv_magnitute(self):
        self.num_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:
                    # remove zeta largest negative and smallest positive weights
                    self.num_weights[name] = torch.sum(self.mask[name])
                    weights = (param * self.mask[name]).detach()
                    values = np.sort(weights.cpu().numpy().ravel())
                    firstZeroPos = self.find_first_pos(values, 0)
                    lastZeroPos = self.find_last_pos(values, 0)
                    largestNegative = values[int((1 - self.prune_rate) * firstZeroPos)]
                    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.prune_rate * (values.shape[0] - lastZeroPos)))]
                    self.mask[name] = copy.deepcopy(weights.data)
                    self.mask[name][self.mask[name] > smallestPositive] = 1
                    self.mask[name][self.mask[name] < largestNegative] = 1
                    self.mask[name][self.mask[name] != 1] = 0; 
    
    # Drop. connection importance is based on its magnitute and importance of connected neuron (WAST)
    def rmv_IMP(self):
        idx = 0
        self.num_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:
                    # remove zeta largest negative and smallest positive weights
                    self.num_weights[name] = torch.sum(self.mask[name])
                    if idx ==0:
                        weights = (param * self.mask[name]).detach() * self.layers_importance[str(idx)] 
                    else:
                        weights_org = (param * self.mask[name]).detach()
                        weights = weights_org.T * self.layers_importance[str(idx)]  
                    values = np.sort(weights.cpu().detach().numpy().ravel())
                    firstZeroPos = self.find_first_pos(values, 0)
                    lastZeroPos = self.find_last_pos(values, 0)
                    largestNegative = values[int((1 - self.prune_rate) * firstZeroPos)]
                    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos + self.prune_rate * (values.shape[0] - lastZeroPos)))]
                    if idx>0:
                        weights = weights.T
                    self.mask[name] = copy.deepcopy(weights.data)
                    self.mask[name][self.mask[name] > smallestPositive] = 1
                    self.mask[name][self.mask[name] < largestNegative] = 1
                    self.mask[name][self.mask[name] != 1] = 0
                    idx+=2 # hardcoded for single layer

    # Grow. Randomly grow new connections (QS baseline)
    def add_random(self):
        # add random weights    
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:
                    noRewires = self.num_weights[name] - torch.sum(self.mask[name])
                    idx_zeros_i, idx_zeros_j = np.where(self.mask[name].to("cpu") == 0) 
                    new_conn_idx = np.random.choice(range(idx_zeros_i.shape[0]), size=int(noRewires.data), replace=False)
                    self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]] = 1

    # Grow. Grow connections based on the importance of its two connected neurons (WAST)
    def add_IMP(self):
        ## index of layer Importance hard-coded for 1 hidden layer of WAST
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if len(param.shape)>1:
                    noRewires = self.num_weights[name] - torch.sum(self.mask[name])
                    noRewires = int(noRewires.cpu())
                    layer_name = str(idx)
                    nxt_layer_name= str(idx+1)
                    layer_importance = torch.mm(self.layers_importance[nxt_layer_name].reshape(self.layers_importance[nxt_layer_name].shape[0], 1),
                                        self.layers_importance[layer_name].reshape(self.layers_importance[layer_name].shape[0], 1).T)
                    layer_importance[self.mask[name]==1] = -self.inf
                    layer_importance = -layer_importance.flatten()
                    idx_add = np.argpartition(layer_importance.detach().cpu(), noRewires)
                    
                    added_mask = torch.zeros_like(layer_importance).to(self.device)
                    added_mask[idx_add[:noRewires]] = 1
                    added_mask = added_mask.reshape(self.mask[name].shape)
                    self.mask[name][added_mask==1]=1

                    idx+=1

    def find_first_pos(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def find_last_pos(self, array, value):
        idx = (np.abs(array - value))[::-1].argmin()
        return array.shape[0] - idx