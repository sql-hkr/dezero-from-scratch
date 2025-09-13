import numpy as np
from dezero import utils
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


class Exp(Function):
    def forward(self, *xs):
        return np.exp(xs[0])

    def backward(self, *gys):
        y = self.outputs[0]()
        return gys[0] * y


def exp(x) -> Variable:
    return Exp()(x)


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


class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return xs[0].sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, *gys):
        gy = utils.reshape_sum_backward(gys[0], self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis=None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return utils.sum_to(xs[0], self.shape)

    def backward(self, *gys):
        return broadcast_to(gys[0], self.x_shape)


def sum_to(x, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return np.broadcast_to(xs[0], self.shape)

    def backward(self, *gys):
        return sum_to(gys[0], self.x_shape)


def broadcast_to(x, shape: tuple):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, *xs):
        return xs[0].dot(xs[1])

    def backward(self, *gys):
        x, W = self.inputs
        gx = matmul(gys[0], W.T)
        gW = matmul(x.T, gys[0])
        return [gx, gW]


def matmul(x, W) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, *xs):
        y = xs[0].dot(xs[1])
        if xs[2] is not None:
            y += xs[2]
        return y

    def backward(self, *gys):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gys[0], b.shape)
        gx = matmul(gys[0], W.T)
        gW = matmul(x.T, gys[0])
        return [gx, gW, gb]


def linear(x, W, b=None) -> Variable:
    return Linear()(x, W, b)


def linear_simple(x, W, b=None) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def sigmoid_simple(x) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))


class Sigmoid(Function):
    def forward(self, *xs):
        return np.tanh(xs[0] * 0.5) * 0.5 + 0.5

    def backward(self, *gys):
        y = self.outputs[0]()
        return gys[0] * y * (1 - y)


def sigmoid(x) -> Variable:
    return Sigmoid()(x)


def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)


class MeanSquaredError(Function):
    def forward(self, *xs):
        diff = xs[0] - xs[1]
        return (diff**2).sum() / len(diff)

    def backward(self, *gys):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gys[0] * diff * (2.0 / len(diff))
        gx1 = -gx0
        return [gx0, gx1]


def mean_squared_error(x0, x1) -> Variable:
    return MeanSquaredError()(x0, x1)
