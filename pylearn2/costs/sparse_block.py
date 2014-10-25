"""Generic "model" class."""
__authors__ = "Dustin Webb"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Dustin Webb"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.compat.six.moves import zip as izip

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class MSE(DefaultDataSpecsMixin, Cost):
    supervised = True

    def __init__(self, L1_reg=0.0, L2_reg=0.0):
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        loss = model.layers[-1].cost(outputs, targets)
        for layer in model.layers:
            loss += (self.L1_reg*abs(layer.W).sum()
                     + self.L2_reg*(layer.W**2).sum())
        return loss
