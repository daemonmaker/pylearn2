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
from theano import printing


class RandomProjectionGater(Model):
    def __init__(self, n_vis, n_classes, batch_size, layers, debug=False):
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

        self.debug = debug

    def fprop(self, x):
        idxs = [self.one_block_idxs]

        # Reshape input for block sparse operator
        activation = x.dimshuffle(0, 'x', 1)

        # Calculate activation of the first layer
        # This should activate all blocks in the second layer
        out_idxs = self.layers[0].calc_out_idxs(activation, True)
        if self.debug:
            out_idxs = printing.Print('idxs in:')(out_idxs)
        idxs.append(out_idxs)
        activation = self.layers[0].output(activation, idxs[-2], idxs[-1])

        # Calculate the activation of the intermediary layers
        for idx in range(1, len(self.layers)-2):
            out_idxs = self.layers[idx].calc_out_idxs(activation)
            if self.debug:
                out_idxs = printing.Print('idxs %d:' % idx)(out_idxs)
            idxs.append(out_idxs)
            activation = self.layers[idx].output(
                activation,
                idxs[-2],
                idxs[-1]
            )

        # Calculate activation for second to last layer
        # This should activate all blocks in the second to last layer
        if len(self.layers) > 2:
            out_idxs = self.layers[-2].calc_out_idxs(activation, True)
            if self.debug:
                out_idxs = printing.Print('idxs out:')(out_idxs)
            idxs.append(out_idxs)
            activation = self.layers[-2].output(activation, idxs[-2], idxs[-1])

        # Calculate the activation of the final layer
        # This is always activate the one and only block in the last layer
        activation = self.layers[-1].output(activation, idxs[-1], idxs[0])

        shp = activation.shape
        #prediction = T.nnet.softmax(activation.reshape((shp[0], shp[2])))
        #prediction = T.nnet.softmax(activation.reshape(x.shape[0], shp[2]))
        prediction = T.nnet.softmax(activation.reshape((x.shape[0], self.layers[-1].n_units_per_out)))
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
        error = T.cast(
            T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean(),
            config.floatX
        )

        monitors = OrderedDict()
        monitors['misclass'] = error
        return monitors