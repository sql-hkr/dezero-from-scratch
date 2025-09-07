import numpy as np


class Variable:
    def __init__(self, data) -> None:
        self.data = data


class Function:
    def __call__(self, input: Variable) -> Variable:
        return Variable(self.forward(input.data))

    def forward(self, x):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x**2


x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
