from math import inf
import mlx.core as mx
from numpy import reshape
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    
    scale_factor = 1 / mx.sqrt(query.shape[-1]) if scale is None else scale
    L, S = query.shape[-2], key.shape[-2]
    attn_bias = mx.zeros([L, S], dtype=query.dtype)

    if mask is not None:
        if mask.dtype == mx.bool_:
            attn_bias.where(mask, float(-inf), 0.0)
        else:
            attn_bias = attn_bias + mask
    
    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = mx.softmax(attn_weight, axis=-1)
    return attn_weight @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size//num_heads
        self.scale = mx.rsqrt(self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        N, L, _ = query.shape
        projection_q = linear(query, self.wq).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1 ,3)
        print("projection shape: ", projection_q.shape)
        projection_k = linear(key, self.wk).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1 ,3)
        projection_v = linear(value, self.wv).reshape(N, L, self.num_heads, self.head_dim).transpose(0, 2, 1 ,3)

        attn_output = scaled_dot_product_attention_simple(projection_q, projection_k, projection_v, self.scale, mask)
        print("attn_output shape: ", attn_output.shape)
        
        attn_output_r = attn_output.transpose(0, 2, 1, 3).reshape(N, L, self.hidden_size)

        print("attn_output reshaped: ", attn_output_r.shape)

        return linear(attn_output_r, self.wo)



def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass

import mlx.core as mx

def reshape_for_gqa(x: mx.array, back: bool = False) -> mx.array:

    ndim = len(x.shape)

    if not back:  # forward: normalize to 5D
        if ndim == 3:
            H, L, D = x.shape
            x = x.reshape(1, 1, H, L, D)
        elif ndim == 4:
            N, H, L, D = x.shape
            x = x.reshape(1, N, H, L, D)
        elif ndim == 5:
            pass  # already in correct form
        else:
            raise ValueError(f"Unexpected shape {x.shape} for forward reshape")

    else:  # backward: restore original shape
        if ndim == 5:
            b, N, H, L, D = x.shape
            if b == 1 and N == 1:
                x = x.reshape(H, L, D)       # back to 3D
            elif b == 1:
                x = x.reshape(N, H, L, D)    # back to 4D
            else:
                pass
        elif ndim in (3, 4):
            pass  # already restored
        else:
            raise ValueError(f"Unexpected shape {x.shape} for backward reshape")

    return x


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    # reshape query, key and value first and then do attention calculations
    print("Key shape: ", key.shape)
    print("Value shape: ", value.shape)
    print("Query shape: ", query.shape)
    print()

    key = reshape_for_gqa(key)
    query = reshape_for_gqa(query)
    value = reshape_for_gqa(value)

    if mask is not None:
        if mask.dtype != mx.bool_:
            mask = reshape_for_gqa(mask)

    print("Key shape after: ", key.shape)
    print("Value shape after: ", value.shape)
    print("Query shape after: ", query.shape)
    print()

    _, N, Hq, L, D = query.shape # 0, 1, 2 ,3; -4, -3, -2, -1
    _, _, H, S, _  =   key.shape # 0, 1, 2, 3; -4, -3, -2, -1
    Hv = value.shape[-3]

    key = mx.repeat(key, Hq//H, -3)
    value = mx.repeat(value, Hq//H, -3)

    print("Key shape after grouping: ", key.shape)
    print("Value shape after grouping: ", value.shape)
    print()

    attn_result =  scaled_dot_product_attention_simple(query, key, value, scale, mask)

    return reshape_for_gqa(attn_result, back = True)


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
