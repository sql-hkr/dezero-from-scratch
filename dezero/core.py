from typing import Optional, Any
import contextlib
import weakref
import numpy as np


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name: str, value: bool):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


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

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
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
        return xs[0] + xs[1]

    def backward(self, *gys: Variable) -> list[Variable]:
        return [gys[0]] * 2


def add(x0, x1) -> Variable:
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, *xs):
        return xs[0] * xs[1]

    def backward(self, *gys):
        x0, x1 = self.inputs
        return [gys[0] * x1, gys[0] * x0]


def mul(x0, x1) -> Variable:
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
        return xs[0] - xs[1]

    def backward(self, *gys):
        return [gys[0], -gys[0]]


def sub(x0, x1) -> Variable:
    return Sub()(x0, x1)


class Div(Function):
    def forward(self, *xs):
        return xs[0] / xs[1]

    def backward(self, *gys):
        x0, x1 = self.inputs
        gx0 = gys[0] / x1
        gx1 = gys[0] * (-x0 / x1**2)
        return [gx0, gx1]


def div(x0, x1) -> Variable:
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
