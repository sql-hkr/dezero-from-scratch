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


A = Square()
B = Exp()
C = Square()

x = Varialbe(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)
