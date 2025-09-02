# python -m unittest step10.py

from typing import Optional, Callable
import unittest
import numpy as np


class Varialbe:
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
            x, y = f.input, f.output
            assert y.grad is not None
            x.grad = f.backward(y.grad)
            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input: Varialbe) -> Varialbe:
        self.input = input
        self.output = Varialbe(as_array(self.forward(input.data)))
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


def square(x: Varialbe) -> Varialbe:
    return Square()(x)


def exp(x: Varialbe) -> Varialbe:
    return Exp()(x)


def numerical_diff(f: Callable[[Varialbe], Varialbe], x: Varialbe, eps=1e-4):
    return (f(Varialbe(x.data + eps)).data - f(Varialbe(x.data - eps)).data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Varialbe(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Varialbe(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Varialbe(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        assert x.grad is not None
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
