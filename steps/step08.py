from typing import Optional
import numpy as np


class Varialbe:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator: Optional[Function] = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        assert self.creator is not None
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            assert y.grad is not None
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


class Function:
    def __call__(self, input: Varialbe) -> Varialbe:
        self.input = input
        self.output = Varialbe(self.forward(input.data))
        self.output.set_creator(self)
        return self.output

    def forward(self, x: np.ndarray):
        raise NotImplementedError()

    def backward(self, gy: np.ndarray):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x: np.ndarray):
        return x**2

    def backward(self, gy: np.ndarray):
        return 2 * self.input.data * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        return np.exp(self.input.data) * gy


A = Square()
B = Exp()
C = Square()

x = Varialbe(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
