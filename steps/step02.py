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


x = Varialbe(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)
