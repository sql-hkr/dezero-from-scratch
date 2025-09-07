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
    def __init__(self, data: np.ndarray, name: Optional[str] = None) -> None:
        self.data = data
        self.name = name
        self.grad: Optional[np.ndarray] = None
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

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

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
            if not isinstance(gxs, list):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs: Variable) -> Any:
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

    def backward(self, *gys: np.ndarray) -> np.ndarray | list[np.ndarray]:
        raise NotImplementedError()


class Square(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return xs[0] ** 2

    def backward(self, *gys: np.ndarray) -> np.ndarray:
        return 2 * self.inputs[0].data * gys[0]


class Add(Function):
    def forward(self, *xs: np.ndarray) -> np.ndarray:
        return xs[0] + xs[1]

    def backward(self, *gys: np.ndarray) -> list[np.ndarray]:
        return [gys[0]] * 2


def square(x: Variable) -> Variable:
    return Square()(x)


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
x.name = "x"

print(x.name)
print(x.shape)
print(x)
