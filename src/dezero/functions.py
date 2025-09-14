from typing import Optional
import numpy as np
import dezero
from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize


class Sin(Function):
    def forward(self, *xs):
        return np.sin(xs[0])

    def backward(self, *gys):
        return gys[0] * cos(self.inputs[0])


def sin(x) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, *xs):
        return np.cos(xs[0])

    def backward(self, *gys):
        return -gys[0] * sin(self.inputs[0])


def cos(x) -> Variable:
    return Cos()(x)


class Tanh(Function):
    def forward(self, *xs):
        return np.tanh(xs[0])

    def backward(self, *gys):
        y = self.outputs[0]()
        assert y is not None
        return gys[0] * (1 - y**2)


def tanh(x) -> Variable:
    return Tanh()(x)


class Exp(Function):
    def forward(self, *xs):
        return np.exp(xs[0])

    def backward(self, *gys):
        y = self.outputs[0]()
        return gys[0] * y


def exp(x) -> Variable:
    return Exp()(x)


class Log(Function):
    def forward(self, *xs):
        return np.log(xs[0])

    def backward(self, *gys):
        (x,) = self.inputs
        return gys[0] / x


def log(x) -> Variable:
    return Log()(x)


class Reshape(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return xs[0].reshape(self.shape)

    def backward(self, *gys):
        return reshape(gys[0], self.x_shape)


def reshape(x, shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None) -> None:
        self.axes = axes

    def forward(self, *xs):
        return xs[0].transpose(self.axes)

    def backward(self, *gys):
        if self.axes is None:
            return transpose(gys[0])
        return transpose(gys[0], self.axes)


def transpose(x, axes=None) -> Variable:
    return Transpose(axes)(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, *xs):
        return xs[0][self.slices]

    def backward(self, *gys):
        (x,) = self.inputs
        return GetItemGrad(self.slices, x.shape)(gys[0])


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, *xs):
        gx = np.zeros(self.in_shape, dtype=xs[0].dtype)
        np.add.at(gx, self.slices, xs[0])
        return gx


def get_item(x, slices) -> Variable:
    return GetItem(slices)(x)


class Sum(Function):
    def __init__(self, axis, keepdims) -> None:
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return xs[0].sum(axis=self.axis, keepdims=self.keepdims)

    def backward(self, *gys):
        gy = utils.reshape_sum_backward(gys[0], self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)


def sum(x, axis: Optional[tuple] = None, keepdims=False) -> Variable:
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return utils.sum_to(xs[0], self.shape)

    def backward(self, *gys):
        return broadcast_to(gys[0], self.x_shape)


def sum_to(x, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape) -> None:
        self.shape = shape

    def forward(self, *xs):
        self.x_shape = xs[0].shape
        return np.broadcast_to(xs[0], self.shape)

    def backward(self, *gys):
        return sum_to(gys[0], self.x_shape)


def broadcast_to(x, shape: tuple) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    def forward(self, *xs):
        return xs[0].dot(xs[1])

    def backward(self, *gys):
        x, W = self.inputs
        gx = matmul(gys[0], W.T)
        gW = matmul(x.T, gys[0])
        return [gx, gW]


def matmul(x, W) -> Variable:
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, *xs):
        y = xs[0].dot(xs[1])
        if xs[2] is not None:
            y += xs[2]
        return y

    def backward(self, *gys):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gys[0], b.shape)
        gx = matmul(gys[0], W.T)
        gW = matmul(x.T, gys[0])
        return [gx, gW, gb]


def linear(x, W, b=None) -> Variable:
    return Linear()(x, W, b)


def linear_simple(x, W, b=None) -> Variable:
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def sigmoid_simple(x) -> Variable:
    x = as_variable(x)
    return 1 / (1 + exp(-x))


class Sigmoid(Function):
    def forward(self, *xs):
        return np.tanh(xs[0] * 0.5) * 0.5 + 0.5

    def backward(self, *gys):
        y = self.outputs[0]()
        return gys[0] * y * (1 - y)


def sigmoid(x) -> Variable:
    return Sigmoid()(x)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x) -> Variable:
    return ReLU()(x)


def softmax_simple(x, axis=1):
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, *xs):
        y = xs[0] - xs[0].max(axis=self.axis, keepdims=True)
        y = np.exp(y)
        return y / y.sum(axis=self.axis, keepdims=True)

    def backward(self, *gys):
        y = self.outputs[0]()
        gx = y * gys[0]
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        return gx - y * sumdx


def softmax(x, axis=1) -> Variable:
    return Softmax(axis)(x)


def mean_squared_error_simple(x0, x1):
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    return sum(diff**2) / len(diff)


class MeanSquaredError(Function):
    def forward(self, *xs):
        diff = xs[0] - xs[1]
        return (diff**2).sum() / len(diff)

    def backward(self, *gys):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gys[0] * diff * (2.0 / len(diff))
        gx1 = -gx0
        return [gx0, gx1]


def mean_squared_error(x0, x1) -> Variable:
    return MeanSquaredError()(x0, x1)


def softmax_cross_entropy_simple(x, t) -> Variable:
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    def forward(self, *xs):
        x, t = xs
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, *gys):
        (gy,) = gys
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t) -> Variable:
    return SoftmaxCrossEntropy()(x, t)


def accuracy(y, t) -> Variable:
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x, dropout_ratio=0.5) -> Variable:
    x = as_variable(x)
    if dezero.Config.train:
        mask = np.random.rand(*x.shape) > dropout_ratio
        scale = np.array(1.0 - dropout_ratio).astype(x.dtype)
        return x * mask / scale
    else:
        return x


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        (x,) = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max) -> Variable:
    return Clip(x_min, x_max)(x)


# =============================================================================
# [simple version] conv2d_simple / pooling_simple
# =============================================================================
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)

    Weight = W
    N, C, H, W = x.shape
    OC, C, KH, KW = Weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
    Weight = Weight.reshape(OC, -1).transpose()
    t = linear(col, Weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class Conv2d(Function):
    def __init__(self, stride=1, pad=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x, W, b):

        KH, KW = W.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = np.rollaxis(y, 3, 1)
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        # ==== gx ====
        gx = deconv2d(
            gy,
            W,
            b=None,
            stride=self.stride,
            pad=self.pad,
            outsize=(x.shape[2], x.shape[3]),
        )
        # ==== gW ====
        gW = Conv2DGradW(self)(x, gy)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, pad=0):
    return Conv2d(stride, pad)(x, W, b)


class Deconv2d(Function):
    def __init__(self, stride=1, pad=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x, W, b):
        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = np.tensordot(Weight, x, (0, 1))
        gcol = np.rollaxis(gcol, 3)
        y = col2im_array(
            gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False
        )
        # b, k, h, w
        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        # ==== gx ====
        gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
        # ==== gW ====
        f = Conv2DGradW(self)
        gW = f(gy, x)
        # ==== gb ====
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gW, gb


def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
    return Deconv2d(stride, pad, outsize)(x, W, b)


class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gW = np.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        (gW,) = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
#  pooling(max-pooling) / average_pooling
# =============================================================================
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = np.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = self.indexes.ravel() + np.arange(
            0, self.indexes.size * KH * KW, KH * KW
        )

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = np.swapaxes(gcol, 2, 4)
        gcol = np.swapaxes(gcol, 3, 5)

        gx = col2im_array(
            gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= KW * KH
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(
            gcol,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            to_matrix=False,
        )
        return gx


def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy):
        gx = col2im(
            gy,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            self.to_matrix,
        )
        return gx


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    """Extract patches from an image based on the filter.

    Args:
        x (`dezero.Variable` or `ndarray`): Input variable of shape
            `(N, C, H, W)`
        kernel_size (int or (int, int)): Size of kernel.
        stride (int or (int, int)): Stride of kernel.
        pad (int or (int, int)): Spatial padding width for input arrays.
        to_matrix (bool): If True the `col` will be reshaped to 2d array whose
            shape is `(N*OH*OW, C*KH*KW)`

    Returns:
        `dezero.Variable`: Output variable. If the `to_matrix` is False, the
            output shape is `(N, C, KH, KW, OH, OW)`, otherwise
            `(N*OH*OW, C*KH*KW)`.

    Notation:
    - `N` is the batch size.
    - `C` is the number of the input channels.
    - `H` and `W` are the height and width of the input image, respectively.
    - `KH` and `KW` are the height and width of the filters, respectively.
    - `SH` and `SW` are the strides of the filter.
    - `PH` and `PW` are the spatial padding sizes.
    - `OH` and `OW` are the the height and width of the output, respectively.
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        y = col2im_array(
            x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix
        )
        return y

    def backward(self, gy):
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    img = np.pad(
        img,
        ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
        mode="constant",
        constant_values=(0,),
    )
    col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
    for j in range(KH):
        j_lim = j + SH * OH
        for i in range(KW):
            i_lim = i + SW * OW
            img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
    return img[:, :, PH : H + PH, PW : W + PW]
