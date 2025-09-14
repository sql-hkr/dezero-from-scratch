from __future__ import annotations
from typing import Optional, Any
import contextlib
import weakref
import numpy as np
import dezero


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def test_mode():
    return using_config("train", False)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        self.data = data
        self.name = name
        self.grad: Optional[Variable] = None
        self.creator: Optional[Function] = None
        self.generation: int = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __getitem__(self, other):
        return dezero.functions.get_item(self, other)

    def set_creator(self, func: Function) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self) -> None:
        self.creator = None

    def cleargrad(self) -> None:
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False) -> None:
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
        funcs = []
        seen_set = set()

        def add_func(f: Function) -> None:
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        assert self.creator is not None
        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs if output().grad is not None]
            gxs = f.backward(*gys)
            with using_config("enable_backprop", create_graph):
                if not isinstance(gxs, list):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    if x.creator is not None:
                        add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def reshape(self, *shape: int):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes: int):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    @property
    def T(self):
        return dezero.functions.transpose(self)


class Parameter(Variable):
    pass


def as_variable(obj) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *input: Variable | np.ndarray | int | float) -> Any:
        inputs = [as_variable(x) for x in input]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *xs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, *gys: Variable) -> Variable | list[Variable]:
        raise NotImplementedError()


class Add(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        self.x0_shape, self.x1_shape = xs[0].shape, xs[1].shape
        return xs[0] + xs[1]

    def backward(self, *gys: Variable) -> list[Variable]:
        gx0, gx1 = gys[0], gys[0]
        if self.x0_shape != self.x1_shape:  # for broadcast
            return [
                dezero.functions.sum_to(gx0, self.x0_shape),
                dezero.functions.sum_to(gx1, self.x1_shape),
            ]
        return [gx0, gx1]


def add(x0, x1) -> Variable:
    if not isinstance(x0, (np.ndarray, Variable)):
        x0 = as_array(x0)
    if not isinstance(x1, (np.ndarray, Variable)):
        x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, *xs):
        return xs[0] * xs[1]

    def backward(self, *gys):
        x0, x1 = self.inputs
        gx0 = gys[0] * x1
        gx1 = gys[0] * x0
        if x0.shape != x1.shape:  # for broadcast
            return [
                dezero.functions.sum_to(gx0, x0.shape),
                dezero.functions.sum_to(gx1, x1.shape),
            ]
        return [gx0, gx1]


def mul(x0, x1) -> Variable:
    if not isinstance(x0, (np.ndarray, Variable)):
        x0 = as_array(x0)
    if not isinstance(x1, (np.ndarray, Variable)):
        x1 = as_array(x1)
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, *xs):
        return -xs[0]

    def backward(self, *gys):
        return -gys[0]


def neg(x) -> Variable:
    return Neg()(x)


class Sub(Function):
    def forward(self, *xs):
        self.x0_shape, self.x1_shape = xs[0].shape, xs[1].shape
        return xs[0] - xs[1]

    def backward(self, *gys):
        gx0 = gys[0]
        gx1 = -gys[0]
        if self.x0_shape != self.x1_shape:  # for broadcast
            return [
                dezero.functions.sum_to(gx0, self.x0_shape),
                dezero.functions.sum_to(gx1, self.x1_shape),
            ]
        return [gx0, gx1]


def sub(x0, x1) -> Variable:
    if not isinstance(x0, (np.ndarray, Variable)):
        x0 = as_array(x0)
    if not isinstance(x1, (np.ndarray, Variable)):
        x1 = as_array(x1)
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, *xs):
        return xs[0] / xs[1]

    def backward(self, *gys):
        x0, x1 = self.inputs
        gx0 = gys[0] / x1
        gx1 = gys[0] * (-x0 / x1**2)
        if x0.shape != x1.shape:  # for broadcast
            return [
                dezero.functions.sum_to(gx0, x0.shape),
                dezero.functions.sum_to(gx1, x1.shape),
            ]
        return [gx0, gx1]


def div(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


class Pow(Function):
    def __init__(self, c) -> None:
        self.c = c

    def forward(self, *xs):
        return xs[0] ** self.c

    def backward(self, *gys):
        x = self.inputs[0]
        return self.c * x ** (self.c - 1) * gys[0]


def pow(x, c) -> Variable:
    return Pow(c)(x)
