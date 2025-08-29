import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array, # --> (N.. x I) matrix - input matrix
    w: mx.array, # --> (O x I) matrix - output matrix
    bias: mx.array | None = None, # --> (O , ) matrix - bias vector 
) -> mx.array:
    if bias is not None:
        return x @ w.T + bias 
    return x @ w.T


def silu(x: mx.array) -> mx.array:
    pass
