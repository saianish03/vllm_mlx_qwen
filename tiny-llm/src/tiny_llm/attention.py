from math import inf
import mlx.core as mx
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


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
