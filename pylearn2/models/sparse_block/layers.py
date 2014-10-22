import numpy as np

import theano
from theano import function
from theano import tensor as T
from theano import config
from theano.tensor.shared_randomstreams import RandomStreams

from theano.sandbox.cuda.blocksparse import sparse_block_dot_SS

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


from pylearn2.utils import sharedX


class HiddenLayer(object):
    def __init__(
        self,
        n_in,
        n_out,
        batch_size,
        k=0.05,
        activation=T.tanh,
        name='HiddenLayer',
        rng=None
    ):
        assert(
            n_in > 0
            and n_out > 0
        )

        self.n_in = n_in
        self.n_out = n_out

        assert(batch_size > 0)
        self.batch_size = batch_size

        self.activation = activation
        self.name = name

        if rng is None:
            self.rng = np.random.RandomState()
            self.rng.seed(0)
        else:
            self.rng = rng

        W_val, b_val = self._init_parameters()
        self._setup_shared_parameters(W_val, b_val)

        assert(k >= 0. and k <= 1.)
        if k > 0.:
            self.k = int(k*n_out)
            assert(self.k > 0)

            name = 'top_active'
            if name is not None:
                name = self.name + '_' + name

            self.top_active = sharedX(
                np.repeat(
                    np.arange(self.k).reshape(1, self.k),
                    self.batch_size,
                    axis=0
                ),
                name,
                dtype='int64'
            )

    def __str__(self):
        return "n_in: %d, n_out: %d" % (self.n_in, self.n_out)

    def _setup_shared_parameters(self, W_val, b_val):
        self.W = sharedX(W_val, str(self.name)+'_W')
        self.b = sharedX(b_val, str(self.name)+'_b')

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

    def get_params(self):
        return [self.W, self.b]

    params = property(get_params)

    def set_parameters(self, W, b):
        self.W.set_value(W)
        self.b.set_value(b)

    def most_active(self, x):
        return function([], updates=(x, T.argsort(x)))

    def output(self, x):
        lin = T.dot(x, self.W) + self.b
        return (lin if self.activation is None
                else self.activation(lin))

    def prediction(self, x):
        return T.argmax(x, axis=1)

    def cost(self, x, y):
        return -T.mean(T.log(x))  # [T.arange(y.shape[0]), y])

    def error(self, x, y):
        return T.mean(T.neq(self.prediction(x), y))


class HiddenBlockLayer(HiddenLayer):
    def __init__(
        self,
        n_in,
        n_out,
        in_idxs,
        out_idxs,
        batch_size,
        k=0,
        activation=T.tanh,
        name='HiddenBlockLayer',
        rng=None,
        l_params=None,
        l_param_map=None
    ):
        assert(
            type(n_in) == tuple
            and type(n_out) == tuple
        )

        self.n_units_per_in = n_in[1]
        self.n_units_per_out = n_out[1]

        assert(
            n_in[1] > 0
            and n_out[1] > 0
        )

        super(
            HiddenBlockLayer,
            self
        ).__init__(
            n_in[0],
            n_out[0],
            batch_size,
            k=k,
            activation=activation,
            name=name,
            rng=rng
        )

        self.in_idxs = in_idxs
        self.out_idxs = out_idxs

        if l_params is not None:
            assert(l_param_map is not None)
        self.l_params = l_params
        self.l_param_map = l_param_map

    def __str__(self):
        return ("n_in: %d (%d units) , n_out: %d (%d units)"
               % (self.n_in, self.n_units_per_in, self.n_out, self.n_units_per_out))

    def _init_parameters(self):
        inputSize = self.n_in*self.n_units_per_in
        outputSize = self.n_out*self.n_units_per_out

        bound = np.sqrt(6. / (inputSize + outputSize))
        W_val = np.asarray(self.rng.uniform(
            low=-bound,
            high=bound,
            size=(
                self.n_in,
                self.n_out,
                self.n_units_per_in,
                self.n_units_per_out
            )
        ), dtype=config.floatX)
        #W_val = np.ones(
        #    (
        #        self.n_in,
        #        self.n_out,
        #        self.n_units_per_in,
        #        self.n_units_per_out
        #    ),
        #    dtype=config.floatX
        #)

        b_val = np.zeros(
            outputSize
        ).reshape(
            self.n_out,
            self.n_units_per_out
        ).astype(config.floatX)

        return W_val, b_val

    def output(self, x):
        if self.l_params is None:
            sparse = sparse_block_dot_SS(
                self.W,
                x,
                self.in_idxs,
                self.b,
                self.out_idxs
            )
        else:
            sparse = sparse_block_dot_SS(
                self.l_params[0].dimshuffle(
                    *self.l_param_map[0]
                )*self.W,
                #self.W,
                x,
                self.in_idxs,
                self.l_params[1].dimshuffle(
                    *self.l_param_map[1]
                )*self.b,
                self.out_idxs
            )

        return (sparse if self.activation is None
                else self.activation(sparse))


class HiddenRandomBlockLayer(HiddenBlockLayer):
    def __init__(
        self,
        batch_size,
        n_in,
        n_out,
        k=0.05,
        activation=T.tanh,
        name='HiddenRandomBlockLayer',
        rng=None,
    ):
        assert(k > 0.)

        super(
            HiddenRandomBlockLayer,
            self
        ).__init__(
            n_in=n_in,
            n_out=n_out,
            in_idxs=None,
            out_idxs=None,
            k=k,
            batch_size=batch_size,
            activation=activation,
            name=name,
            rng=rng,
        )

    def calc_out_idxs(self, input):
        iWin = self.k

        if self.n_in == 1:
            iWin = 1

        inner_dim = iWin*self.n_units_per_in

        trng = RandomStreams(seed=0)
        rand_proj_mat = trng.normal(
            #(iWin*self.n_units_per_in, self.n_out)
            (inner_dim, self.n_out)
        )
        #temp = input.shape[1]*input.shape[2]
        #temp = self.n_units_per_in*self.k
        #if self.n_in == 1:
        #    temp = self.n_units_per_in
        rnd_proj = T.dot(
            input.reshape((input.shape[0], input.shape[1]*input.shape[2])),
            #x.reshape((self.batch_size, temp)),
            rand_proj_mat
        )
        self.out_idxs = T.cast(T.argsort(rnd_proj)[:, -self.k:], 'int64')

    def __str__(self):
        return ("n_in: %d (%d units) , n_out: %d (%d units)"
               % (self.n_in, self.n_units_per_in, self.n_out, self.n_units_per_out))

    def _init_parameters(self):
        inputSize = self.n_in*self.n_units_per_in
        outputSize = self.n_out*self.n_units_per_out

        bound = np.sqrt(6. / (inputSize + outputSize))
        W_val = np.asarray(self.rng.uniform(
            low=-bound,
            high=bound,
            size=(
                self.n_in,
                self.n_out,
                self.n_units_per_in,
                self.n_units_per_out
            )
        ), dtype=config.floatX)
        #W_val = np.ones(
        #    (
        #        self.n_in,
        #        self.n_out,
        #        self.n_units_per_in,
        #        self.n_units_per_out
        #    ),
        #    dtype=config.floatX
        #)

        b_val = np.zeros(
            outputSize
        ).reshape(
            self.n_out,
            self.n_units_per_out
        ).astype(config.floatX)

        return W_val, b_val

    def output(self, x):
        sparse = sparse_block_dot_SS(
            self.W,
            x,
            self.in_idxs,
            self.b,
            self.out_idxs
        )

        return (sparse if self.activation is None
                else self.activation(sparse))
