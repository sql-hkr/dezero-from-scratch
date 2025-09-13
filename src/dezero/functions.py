import numpy as np
from dezero.core import Function, Variable, as_variable


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


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return xs[0].reshape(self.shape)

    def backward(self, *gys):
        return reshape(gys[0], self.x_shape)


def reshape(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None) -> None:
        self.axes = axes

    def forward(self, *xs):
        return xs[0].transpose(self.axes)

    def backward(self, *gys):
        if self.axes is None:
            return transpose(gys[0])
        return transpose(gys[0], self.axes)


def transpose(x, axes=None) -> Variable:
    return Transpose(axes)(x)
