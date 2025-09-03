import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
    
    def polar(self, a, b):
        return a * mx.exp(1j * b)

    def view_as_complex(self, a):
        real = a[..., 0]
        complex = a[..., 1]
        return mx.array([real + complex*1j], dtype=mx.complex64)

    def view_as_real(self, a):
        real = a.real
        complex = a.imag
        return mx.stack([real, complex], axis=-1)
    
    def rope_init(self):
        theta_numerator = mx.arange(0, self.dims, 2, dtype=mx.float32)
        theta = 1.0 / (self.base ** (theta_numerator / self.dims))
        self.theta = theta
        self.build_rope(self.seq_len)

    def build_rope(self, seq_len: int):
        seq_idx = mx.arange(seq_len, dtype=mx.float32)
        idx_freq = mx.outer(seq_idx, self.theta)
        self.freqs_complex = self.polar(mx.ones_like(idx_freq), idx_freq)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        self.rope_init()
        x_complex = self.view_as_complex(x.reshape(N, L, H, D//2, 2))
        freqs_complex = self.freqs_complex[:L, :] if offset is None else self.freqs_complex[offset, :]
        freqs_complex = freqs_complex.reshape(-1, L, 1, D//2)

        if not self.traditional:
            x1 = x[..., 0:D//2]
            x2 = x[..., D//2:D]

            cos = freqs_complex.real
            sin = freqs_complex.imag

            o1 = mx.multiply(x1, cos) - mx.multiply(x2, sin)
            o2 = mx.multiply(x2, cos) + mx.multiply(x1, sin)

            x_out = mx.concat([o1, o2], axis = -1)
            x_out = x_out.reshape(x.shape)
        else:
            x_rotated = x_complex * freqs_complex
            x_out = self.view_as_real(x_rotated)
            x_out = x_out.reshape(x.shape)
        return x_out.astype(mx.float32)