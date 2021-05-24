import os
import os.path as path
import numpy as np
import random
from sys import getsizeof



class Dataset:
    """
        Dataset is the superclass which contains high level variables:
        x_{train,val,test}, y_{train,val,test}
        dir_{train,val,test}
        loader : function to read the dataset
        preprocess : function to preprocess the datasets
        normalize : normalize train, val and test datasets
        Note: while normalizing:
            1. stats obtained from train can be used to normalize val and test
            2. val and test can be normalized independently
            This needs to be fixed.... #LOOK

        Example code is provided for NUS-wide
    """
    def __init__(self, directories,
                         loader, 
                         preprocess=None,
                         preprocess_params=None,
                         normalize=None,
                         normalization_params=None,
                         read_directories=(True,True,True),
                         summarize=None
                         ):
        self.dir_train, self.dir_val, self.dir_test = directories
        self.read_train, self.read_val, self.read_test = read_directories
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.loader = loader
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.normalize = normalize
        self.normalization_params = normalization_params
        self.summarize = summarize
        self.stats = None

    def get_train_labels(self):
        return self.y_train

    def get_test_labels(self):
        return self.y_test

    def get_val_labels(self):
        return self.y_val

    def check_directory(self, dir, msg='{} Directory does not exist!!'):
        assert path.exists(dir), msg.format(dir)

    def load_data(self):
        if self.read_train:
            self.check_directory(self.dir_train)
            self.x_train, self.y_train = self.loader(self.dir_train, "train")
        if self.read_val:
            self.check_directory(self.dir_val)
            self.x_val, self.y_val = self.loader(self.dir_val, "val")
        if self.read_test:
            self.check_directory(self.dir_test)
            self.x_test, self.y_test = self.loader(self.dir_test, "test")

    def preprocess_data(self):
        if not self.preprocess:
            def func(x,y, params=None):
                return x,y
            self.preprocess = func

        if not self.normalize:
            def func(x,y, params=None):
                return x,y
            self.normalize = func

        if self.read_train:
            self.x_train, self.y_train = self.preprocess(self.x_train, self.y_train, self.preprocess_params)
            self.x_train, self.y_train = self.normalize(self.x_train, self.y_train, self.normalization_params)
        if self.read_val:
            self.x_val, self.y_val = self.preprocess(self.x_val, self.y_val, self.preprocess_params)
            self.x_val, self.y_val = self.preprocess(self.x_val, self.y_val, self.normalization_params)
        if self.read_test:
            self.x_test, self.y_test = self.preprocess(self.x_test, self.y_test, self.preprocess_params)
            self.x_test, self.y_test = self.preprocess(self.x_test, self.y_test, self.normalization_params)

    def get_stats(self):
        if (self.summarize == None):
            return None

        if self.stats:
            return self.stats

        stats = {}
        if self.read_train:
            stats['train'] = self.summarize(self.x_train, self.y_train, 'train')
        if self.read_test:
            stats['test'] = self.summarize(self.x_test, self.y_test, 'test')
        if self.read_val:
            stats['val'] = self.summarize(self.x_val, self.y_val, 'val')

        self.stats = stats
        return self.stats