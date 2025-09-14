import numpy as np
from dezero import test_mode
import dezero.functions as F

np.random.seed(0)

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
