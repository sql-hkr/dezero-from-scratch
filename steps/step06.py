from typing import Optional
import numpy as np


class Variable:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None


class Function:
    def __call__(self, input: Variable) -> Variable:
        self.input = input
        return Variable(self.forward(input.data))

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

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)
print(x.grad)
