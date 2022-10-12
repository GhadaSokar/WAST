import os
from matplotlib import dates
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import torch
import torch.nn.functional as F
import urllib.request as urllib2 
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from torchvision import datasets, transforms
from PIL import Image

class DatasetSplitter(torch.utils.data.Dataset):
    """This splitter makes sure that we always use the same training/validation split"""
    def __init__(self,parent_dataset,split_start=-1,split_end= -1):
        split_start = split_start if split_start != -1 else 0
        split_end = split_end if split_end != -1 else len(parent_dataset)
        assert split_start <= len(parent_dataset) - 1 and split_end <= len(parent_dataset) and     split_start < split_end , "invalid dataset split"

        self.parent_dataset = parent_dataset
        self.split_start = split_start
        self.split_end = split_end

    def __len__(self):
        return self.split_end - self.split_start


    def __getitem__(self,index):
        assert index < len(self),"index out of bounds in split_datset"
        return self.parent_dataset[index + self.split_start]

def get_loaders(train, test, batch_size, test_batch_size):
    train_loader = torch.utils.data.DataLoader(
    train,
    batch_size,
    num_workers=8,
    pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))
    valid_loader = None
    test_loader = torch.utils.data.DataLoader(
        test,
        test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_dataset_loader(args):
    if args.data == 'mnist':
        train_loader, valid_loader, test_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
        input_size = 28*28
    elif args.data == 'FashionMnist':
        train_loader, valid_loader, test_loader = get_FashionMnist_dataloaders(args, validation_split=args.valid_split)
        input_size = 28*28
    elif args.data == 'madelon':
        train_loader, valid_loader, test_loader = get_madelon_dataloaders(args, validation_split=args.valid_split)
        input_size = 500    
    elif args.data == 'coil':
        train_loader, valid_loader, test_loader = get_coil_dataloaders(args, validation_split=args.valid_split)
        input_size = 1024    
    elif args.data == 'USPS':
        train_loader, valid_loader, test_loader = get_USPS_dataloaders(args, validation_split=args.valid_split)
        input_size = 256
    elif args.data == 'HAR':
        train_loader, valid_loader, test_loader = get_HAR_dataloaders(args, validation_split=args.valid_split)
        input_size = 561 
    elif args.data == 'Isolet':
        train_loader, valid_loader, test_loader = get_Isolet_dataloaders(args, validation_split=args.valid_split)
        input_size = 617
    elif args.data == 'PCMAC':
        train_loader, valid_loader, test_loader = get_PCMAC_dataloaders(args, validation_split=args.valid_split)
        input_size = 3289
    elif args.data == 'SMK':
        train_loader, valid_loader, test_loader = get_SMK_dataloaders(args, validation_split=args.valid_split)
        input_size = 19993
    elif args.data == 'GLA':
        train_loader, valid_loader, test_loader = get_GLA_dataloaders(args, validation_split=args.valid_split)
        input_size = 49151

    return train_loader, valid_loader, test_loader, input_size
    
def get_mnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_FashionMnist_dataloaders(args, validation_split=0.0):
    """Creates augmented train, validation, and test data loaders."""
    normalize = transforms.Normalize((0.2859,), (0.3530,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])

    full_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)

    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    valid_loader = None
    if validation_split > 0.0:
        split = int(np.floor((1.0-validation_split) * len(full_dataset)))
        train_dataset = DatasetSplitter(full_dataset,split_end=split)
        val_dataset = DatasetSplitter(full_dataset,split_start=split)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.test_batch_size,
            num_workers=2,
            pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            full_dataset,
            args.batch_size,
            num_workers=8,
            pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_madelon_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_madelon()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = Madelon('./data', train=True, download=True, transform=transform)
    test = Madelon('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train,
        args.batch_size,
        num_workers=8,
        pin_memory=True, shuffle=True)

    print('Train loader length', len(train_loader))
    valid_loader = None
    test_loader = torch.utils.data.DataLoader(
        test,
        args.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return train_loader, valid_loader, test_loader

def get_coil_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_coil()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('coil', './data', train=True, download=True, transform=transform)
    test = custom_data('coil', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_USPS_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_USPS()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('USPS', './data', train=True, download=True, transform=transform)
    test = custom_data('USPS', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader


def get_SMK_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_SMK()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('SMK', './data', train=True, download=True, transform=transform)
    test = custom_data('SMK', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_PCMAC_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_PCMAC()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('PCMAC', './data', train=True, download=True, transform=transform)
    test = custom_data('PCMAC', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader


def get_HAR_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_HAR()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('HAR', './data', train=True, download=True, transform=transform)
    test = custom_data('HAR', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_Isolet_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_Isolet()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('Isolet', './data', train=True, download=True, transform=transform)
    test = custom_data('Isolet', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

def get_GLA_dataloaders(args, validation_split=0.0):
    m, std = get_m_std_GLA()
    normalize = transforms.Normalize((m,), (std,))
    transform = transforms.Compose([transforms.ToTensor(),normalize])
    train = custom_data('GLA', './data', train=True, download=True, transform=transform)
    test = custom_data('GLA', './data', train=False, download=True, transform=transform)
    train_loader, valid_loader, test_loader = get_loaders(train, test, args.batch_size, args.test_batch_size)
    return train_loader, valid_loader, test_loader

class custom_data(torch.utils.data.Dataset):
    def __init__(self, dataset, root, train=True, download=False, transform=None):
        if dataset == 'Isolet':
            X_train, y_train, X_test, y_test = read_Isolet()
        elif dataset == 'coil':
            X_train, y_train, X_test, y_test = read_coil()
        elif dataset == 'HAR':
            X_train, y_train, X_test, y_test = read_HAR()
        elif dataset == 'GLA':
            X_train, y_train, X_test, y_test = read_GLA()
        elif dataset == 'USPS':
            X_train, y_train, X_test, y_test = read_USPS()
        elif dataset == 'SMK':
            X_train, y_train, X_test, y_test = read_SMK()
        elif dataset == 'PCMAC':
            X_train, y_train, X_test, y_test = read_PCMAC()
        if train:
            self.data = X_train
            self.targets = y_train
        else:
            self.data = X_test
            self.targets = y_test
        self.transform = transform   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(np.array(img))

        return img, target 
       
class Madelon(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=False, transform=None):
        train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
        val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
        train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
        val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
        test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
        if train:
            self.data = np.loadtxt(urllib2.urlopen(train_data_url))
            self.targets = np.loadtxt(urllib2.urlopen(train_resp_url))
        else:
            self.data =  np.loadtxt(urllib2.urlopen(val_data_url))
            self.targets =  np.loadtxt(urllib2.urlopen(val_resp_url))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args: index (int): Index
        Returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(np.array(img))

        return img, target

def load_mnist():

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    test_X  = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    train_X = train_X.astype('float32')
    test_X  = test_X.astype('float32')    
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)    
    print("X_test shape = "+ str( test_X.shape))
    return train_X, train_y, test_X, test_y

def load_FashionMnist():

    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]))
    test_X  = test_X.reshape((test_X.shape[0],test_X.shape[1]*test_X.shape[2]))
    train_X = train_X.astype('float32')
    test_X  = test_X.astype('float32')    
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)    
    print("X_test shape = "+ str( test_X.shape))
    return train_X, train_y, test_X, test_y

def read_madelon():
    train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
    val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
    train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
    val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
    test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_test.data'
    train_X = np.loadtxt(urllib2.urlopen(train_data_url))
    train_y = np.loadtxt(urllib2.urlopen(train_resp_url))
    test_X =  np.loadtxt(urllib2.urlopen(val_data_url))
    test_y =  np.loadtxt(urllib2.urlopen(val_resp_url))
    train_X = train_X.astype('float32')
    test_X  = test_X.astype('float32')   
    return train_X, train_y, test_X, test_y
def get_m_std_madelon():
    train_X, train_y, test_X, test_y = read_madelon()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_madelon():
    train_X, train_y, test_X, test_y = read_madelon() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_coil():
    mat = loadmat('./data/COIL20.mat')
    X = mat['fea']
    y = mat['gnd'] 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_coil():
    train_X, train_y, test_X, test_y = read_coil()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_coil():
    train_X, train_y, test_X, test_y = read_coil() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_USPS():
    mat = loadmat('./data/USPS.mat')
    X = mat['X']
    y = mat['Y'] 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_USPS():
    train_X, train_y, test_X, test_y = read_USPS()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_USPS():
    train_X, train_y, test_X, test_y = read_USPS() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_PCMAC():
    mat = loadmat('./data/PCMAC.mat')
    X = mat['X']
    y = mat['Y'] 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_PCMAC():
    train_X, train_y, test_X, test_y = read_PCMAC()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_PCMAC():
    train_X, train_y, test_X, test_y = read_PCMAC() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y


def read_SMK():
    mat = loadmat('./data/SMK_CAN_187.mat')
    X = mat['X']
    y = mat['Y'] 
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_SMK():
    train_X, train_y, test_X, test_y = read_SMK()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_SMK():
    train_X, train_y, test_X, test_y = read_SMK() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_GLA():
    mat = loadmat('./data/GLA-BRA-180.mat', squeeze_me=True)
    X = mat["X"]
    y = mat["Y"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_GLA():
    train_X, train_y, test_X, test_y = read_GLA()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_GLA():
    train_X, train_y, test_X, test_y = read_GLA() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_Isolet():
    import pandas as pd
    df=pd.read_csv('./data/isolet.csv', sep=',',header=None)
    data = df.values
    X = data[1:,:-1].astype('float')
    y = [int(x.replace('\'','')) for x in data[1:,-1]]
    #y = list(map(int, data[1:,-1]))
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42) 
    return train_X, train_y, test_X, test_y
def get_m_std_Isolet():
    train_X, train_y, test_X, test_y = read_Isolet()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_Isolet():
    train_X, train_y, test_X, test_y = read_Isolet() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def read_HAR():
    X_train = np.loadtxt('./data/UCI_HAR_Dataset/train/X_train.txt')
    y_train = np.loadtxt('./data/UCI_HAR_Dataset/train/y_train.txt')
    X_test =  np.loadtxt('./data/UCI_HAR_Dataset/test/X_test.txt')
    y_test =  np.loadtxt('./data/UCI_HAR_Dataset/test/y_test.txt')
    return X_train, y_train, X_test, y_test
def get_m_std_HAR():
    train_X, train_y, test_X, test_y = read_HAR()
    m = np.mean(train_X)
    std = np.std(train_X)
    return m, std
def load_HAR():
    train_X, train_y, test_X, test_y = read_HAR() 
    train_X, test_X = std_transform(train_X, test_X)
    return train_X, train_y, test_X, test_y

def std_transform(train_X, test_X):
    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)  
    print("X_test shape = "+ str( test_X.shape))
    return train_X, test_X


def get_data(args):
    if args.data == 'mnist':
        train_X, train_y, test_X, test_y = load_mnist() 

    elif args.data == 'FashionMnist':
        train_X, train_y, test_X, test_y = load_FashionMnist() 

    elif args.data == 'madelon':
        train_X, train_y, test_X, test_y = load_madelon() 

    elif args.data == 'coil':
        train_X, train_y, test_X, test_y = load_coil() 

    elif args.data == 'HAR':
        train_X, train_y, test_X, test_y = load_HAR() 

    elif args.data == 'Isolet':
        train_X, train_y, test_X, test_y = load_Isolet()

    elif args.data == 'GLA':
        train_X, train_y, test_X, test_y = load_GLA()

    elif args.data == 'USPS':
        train_X, train_y, test_X, test_y = load_USPS()

    elif args.data == 'SMK':
        train_X, train_y, test_X, test_y = load_SMK()

    elif args.data == 'PCMAC':
        train_X, train_y, test_X, test_y = load_PCMAC()
        
    return train_X, train_y, test_X, test_y