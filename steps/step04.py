from typing import Callable
import numpy as np


class Varialbe:
    def __init__(self, data) -> None:
        self.data = data


class Function:
    def __call__(self, input: Varialbe) -> Varialbe:
        return Varialbe(self.forward(input.data))

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x**2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


def numerical_diff(f: Callable[[Varialbe], Varialbe], x: Varialbe, eps=1e-4):
    return (f(Varialbe(x.data + eps)).data - f(Varialbe(x.data - eps)).data) / (2 * eps)


f = Square()
x = Varialbe(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)


def g(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Varialbe(np.array(0.5))
dy = numerical_diff(g, x)
print(dy)
