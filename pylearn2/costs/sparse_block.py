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

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        loss = model.layers[-1].cost(outputs, targets)
        return loss

    def get_gradients(self, model, data, ** kwargs):
        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            message = "Error while calling " + str(type(self)) + ".expr"
            reraise_as(TypeError(message))

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        consider_constants = []
        for i in range(len(model.layers)):
            consider_constants += [model.layers[i].in_idxs,
                                   model.layers[i].out_idxs]

        params = list(model.get_params())

        grads = T.grad(
            cost,
            params,
            disconnected_inputs='ignore',
            consider_constant=consider_constants
        )

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates
