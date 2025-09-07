from typing import Optional, Any
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        assert self.creator is not None
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs if output.grad is not None]
            gxs = f.backward(*gys)
            if not isinstance(gxs, list):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)


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
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
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


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))
z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)
