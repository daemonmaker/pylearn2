"""Generic "model" class."""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import numpy as np

from pylearn2.space import VectorSpace
from pylearn2.models.model import Model
from pylearn2.models.sparse_block.layers import HiddenRandomBlockLayer
from pylearn2.utils import sharedX
from pylearn2.space import CompositeSpace

import theano.tensor as T
from theano import config
from theano.compat.python2x import OrderedDict


class RandomProjectionGater(Model):
    def __init__(self, n_vis, n_classes, batch_size, layers):
        super(RandomProjectionGater, self).__init__()

        assert(n_vis > 0)
        self.n_vis = n_vis

        assert(n_classes > 0)
        self.n_classes = n_classes

        assert(batch_size > 0)
        self.batch_size = batch_size

        self.input_space = VectorSpace(dim=self.n_vis)
        self.output_space = VectorSpace(dim=self.n_classes)

        self.input_space.make_batch_theano

        # Shared variable used for always activating one block in a layer as in
        # the input and output layers
        self.one_block_idxs = sharedX(
            np.zeros((self.batch_size, 1)),
            'one_block_idxs',
            dtype='int64'
        )

        self._params = []
        for idx in range(len(layers)):
            if layers[idx].name is None:
                layers[idx].name = 'Layer_%d_%d' % (len(layers), idx)
            self._params += layers[idx].params
            print 'Layer specs: ', layers[idx]

        self.layers = layers

    def fprop(self, x):
        activation = x.dimshuffle(0, 'x', 1)
        if self.layers[0].in_idxs is None:
            self.layers[0].in_idxs = self.one_block_idxs
        if self.layers[0].out_idxs is None:
            self.layers[0].set_out_idxs(activation)
        activation = self.layers[0].output(activation)
        for idx in range(1, len(self.layers)-1):
            if self.layers[idx].out_idxs is None:
                self.layers[idx].set_out_idxs(activation)

            if self.layers[idx].in_idxs is None:
                self.layers[idx].in_idxs = self.layers[idx-1].out_idxs

            activation = self.layers[idx].output(activation)
        self.layers[-1].in_idxs = self.layers[-2].out_idxs
        self.layers[-1].out_idxs = self.one_block_idxs
        activation = self.layers[-1].output(activation)

        shp = activation.shape
        prediction = T.nnet.softmax(activation.reshape((shp[0], shp[2])))
        return prediction

    def _init_parameters(self):
        bound = np.sqrt(6. / (self.n_in + self.n_out))
        W_val = np.asarray(self.rng.uniform(
            low=-bound,
            high=bound,
            size=(
                self.n_in,
                self.n_out
            )
        ), dtype=config.floatX)
        #W_val = np.ones((self.n_in, self.n_out), dtype=config.floatX)

        b_val = np.zeros((self.n_out,)).astype(config.floatX)

        return W_val, b_val

    def get_monitoring_data_specs(self):
        space = CompositeSpace([self.get_input_space(),
                            self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.fprop(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()

        return OrderedDict([('error', error)])
