if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 1

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    assert gx is not None
    gx.backward(create_graph=True)

gx = x.grad
assert gx is not None
gx.name = "gx" + str(iters + 1)
plot_dot_graph(gx, verbose=False, to_file="tanh.png")
