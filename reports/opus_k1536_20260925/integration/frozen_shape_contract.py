def a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k):
    """Scalar-only shape contract shared by native-E8M0 candidate filtering.

    The generated exact-kid launcher repeats these checks before launching.
    In particular, fixed-K kernels must never be offered for another K.
    """
    if instance.kernel_tag != 'a8w8_mxscale_gemm_bpreshuffle' or min(m, n, k) <= 0 or m % instance.m_align or n % instance.B_N or n % instance.GROUP_N or k % instance.B_K or (instance.fixed_k is not None and k != instance.fixed_k):
        return False
    return max(m * k, n * k, 2 * m * n) <= instance.max_tensor_bytes
