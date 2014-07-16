"""
Class for reading path solutions into a dense design matrix.
"""
__authors__ = ["Dustin Webb"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "Dustin Webb"
__email__ = "webbd@iro"

import cPickle

import pylearn2.utils.serial as serial
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class PathSolutionsDataset(DenseDesignMatrix):
    def __init__(self, pkl_path, which_set):
        fh = open(pkl_path, 'rb')
        X = cPickle.load(fh)
        y = cPickle.load(fh)

        set_types = ['train', 'validate', 'test']
        assert(which_set in set_types)
        percent_train_samples = 0.8
        percent_other_samples = (1 - percent_train_samples)/2
        training_samples = int(percent_train_samples*X.shape[0])
        other_samples = int(percent_other_samples*X.shape[0])
        if which_set == 'train':
            X = X[:training_samples]
            y = y[:training_samples]
        elif which_set == 'validate':
            X = X[training_samples:(training_samples+other_samples)]
            y = y[training_samples:(training_samples+other_samples)]
        else:
            X = X[(training_samples+other_samples):]
            y = y[(training_samples+other_samples):]

        super(PathSolutionsDataset, self).__init__(X=X, y=y)
