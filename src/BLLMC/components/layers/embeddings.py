"""
licence : mit
author : amzad hossain rafi
email : amzad.rafi@northsouth.edu

change log :
    23-5-2026 : start
    23-5-2026 : implement RoPE (Rotary Position Embedding)

to do :
    1. Integrate with MultiHeadAttention
    2. Implement test cases

"""

import torch


def compute_rope_params(
    head_dim,
    theta_base=10_000,
    context_length=4096,
    dtype=torch.float32,
    scaling_factor=1.0,
    initial_context_length=4096,
    ntk_alpha=1.0,
    ntk_beta=32.0,
):
    """
    Precompute the cosine and sine tables for Rotary Position Embeddings (RoPE).

    RoPE encodes absolute position information into the attention mechanism by rotating
    query and key vectors in 2D subspaces. This function precomputes the rotation
    matrices (as cos/sin pairs) for all positions up to `context_length`.

    Supports YaRN / NTK-by-parts scaling for context window extension.

    Args:
        head_dim (int): Dimension of each attention head. Must be even since RoPE
            operates on pairs of dimensions.
        theta_base (float): Base frequency for the inverse frequency computation.
            Higher values spread frequencies over a wider range, allowing the model
            to distinguish positions at larger distances. Default: 10_000.
        context_length (int): Maximum sequence length to precompute rotations for.
            Default: 4096.
        dtype (torch.dtype): Data type for computation. Default: torch.float32.
        scaling_factor (float): Scaling factor for context extension. Default: 1.0.
        initial_context_length (int): Original context length of base model. Default: 4096.
        ntk_alpha (float): YaRN/NTK alpha parameter. Default: 1.0.
        ntk_beta (float): YaRN/NTK beta parameter. Default: 32.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - cos: Cosine table of shape (context_length, head_dim)
            - sin: Sine table of shape (context_length, head_dim)

    Example:
        >>> cos, sin = compute_rope_params(head_dim=64, context_length=2048)
        >>> cos.shape  # (2048, 64)
        >>> sin.shape  # (2048, 64)
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    freq = theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
    )

    if scaling_factor > 1.0:
        import math
        concentration = 0.1 * math.log(scaling_factor) + 1.0

        d_half = head_dim / 2
        # NTK by parts
        low = (
            d_half
            * math.log(initial_context_length / (ntk_beta * 2 * math.pi))
            / math.log(theta_base)
        )
        high = (
            d_half
            * math.log(initial_context_length / (ntk_alpha * 2 * math.pi))
            / math.log(theta_base)
        )
        assert 0 < low < high < d_half - 1

        interpolation = 1.0 / (scaling_factor * freq)
        extrapolation = 1.0 / freq

        ramp = (
            torch.arange(d_half, dtype=dtype, device=freq.device) - low
        ) / (high - low)
        mask = 1 - ramp.clamp(0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
    else:
        concentration = 1.0
        inv_freq = 1.0 / freq

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles: outer product of positions and inverse frequencies
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(
        0
    )  # Shape: (context_length, head_dim // 2)

    # Duplicate angles to match head_dim (each pair of dims shares the same angle)
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles) * concentration
    sin = torch.sin(angles) * concentration

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    """
    Apply Rotary Position Embeddings to the input tensor.

    Rotates query or key vectors using precomputed cos/sin tables. Each pair of
    dimensions (x1, x2) is rotated by:
        x_rotated = x * cos + rotate(x) * sin
    where rotate(x) = [-x2, x1] (the 90-degree rotation).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_heads, seq_len, head_dim).
            Typically the Q or K tensor after head splitting and transposing.
        cos (torch.Tensor): Precomputed cosine table from `compute_rope_params`,
            shape (context_length, head_dim).
        sin (torch.Tensor): Precomputed sine table from `compute_rope_params`,
            shape (context_length, head_dim).
        offset (int): Position offset for KV-cache inference. When generating token
            at position `t`, set offset=t so the correct rotation is applied.
            Default: 0.

    Returns:
        torch.Tensor: Rotated tensor of same shape and dtype as input x.

    Example:
        >>> cos, sin = compute_rope_params(head_dim=64, context_length=2048)
        >>> q = torch.randn(2, 8, 128, 64)  # (batch, heads, seq_len, head_dim)
        >>> q_rotated = apply_rope(q, cos, sin)
        >>> q_rotated.shape  # (2, 8, 128, 64)
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    half_dim = head_dim // 2

    # Slice and broadcast cos/sin for the current sequence positions.
    # We only slice the first half (dim=-1) to avoid redundant computation.
    cos_half = cos[offset : offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(0)
    sin_half = sin[offset : offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(0)

    # Split x into first half and second half
    x1 = x[..., :half_dim]  # First half
    x2 = x[..., half_dim:]  # Second half

    # Apply the rotary transformation:
    #   o1 = x1 * cos_half - x2 * sin_half
    #   o2 = x2 * cos_half + x1 * sin_half
    o1 = (x1 * cos_half) - (x2 * sin_half)
    o2 = (x2 * cos_half) + (x1 * sin_half)

    x_rotated = torch.cat((o1, o2), dim=-1)

    # Cast back to input dtype (cos/sin may have been float32)
    return x_rotated.to(dtype=x.dtype)
