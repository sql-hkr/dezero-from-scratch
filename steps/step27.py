if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, *xs):
        return np.sin(xs[0])

    def backward(self, *gys):
        x = self.inputs[0].data
        return gys[0] * np.cos(x)


def sin(x) -> Variable:
    return Sin()(x)


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print("--- original ---")
print(y.data)
print(x.grad)


def my_sin(x, threshold=0.0001) -> Variable:
    y = Variable(np.array(0.0))
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x = Variable(np.array(np.pi / 4))
y = my_sin(x)
y.backward()
print(y.data)
print(x.grad)

x.name = "x"
y.name = "y"
plot_dot_graph(y, verbose=False, to_file="my_sin.png")
