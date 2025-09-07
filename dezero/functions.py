import numpy as np
from dezero.core import Function, Variable


class Sin(Function):
    def forward(self, *xs):
        return np.sin(xs[0])

    def backward(self, *gys):
        return gys[0] * cos(self.inputs[0])


def sin(x) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, *xs):
        return np.cos(xs[0])

    def backward(self, *gys):
        return -gys[0] * sin(self.inputs[0])


def cos(x) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, *xs):
        return np.tanh(xs[0])

    def backward(self, *gys):
        y = self.outputs[0]()
        assert y is not None
        return gys[0] * (1 - y**2)


def tanh(x) -> Variable:
    return Tanh()(x)
