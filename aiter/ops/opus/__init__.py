# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public OPUS GEMM selection and launch APIs."""

from ._arch import _detect_arch

_SUPPORTED = {"gfx950", "gfx942", "gfx1250"}
_FEATURE = "aiter.ops.opus"
_HINT = (
    "opus_gemm supports gfx950 (MFMA 16x16x32 / ds_read_b64_tr / 160 KiB "
    "LDS), gfx942 (MFMA 16x16x16 / ds_read_b128 / 64 KiB LDS), and gfx1250. Set "
    "GPU_ARCHS to one of these (or run on a matching device) to use this "
    "module."
)

_arch_ok, _detected_arch = _detect_arch(_SUPPORTED)


def _make_unsupported_arch_stub(name: str):
    """Return a callable that reports the unsupported architecture."""

    def _stub(*_args, **_kwargs):
        raise RuntimeError(
            f"{name} requires GPU arch in {sorted(_SUPPORTED)}; "
            f"detected {_detected_arch!r}. {_HINT}"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = f"Stub: {_FEATURE} unavailable on {_detected_arch!r}."
    return _stub


if _arch_ok:
    from .gemm_op_a16w16 import (
        gemm_a16w16_opus,
        opus_gemm_a16w16_launch,
        opus_gemm_a16w16_tune,
        opus_gemm_workspace_init,
    )

    def opus_gemm_a8w8_blockscale_bpreshuffle_tune(*args, **kwargs):
        from .gemm_op_a8w8 import (
            opus_gemm_a8w8_blockscale_bpreshuffle_tune as _impl,
        )

        return _impl(*args, **kwargs)

    def opus_gemm_a8w8_launch(*args, **kwargs):
        from .gemm_op_a8w8 import opus_gemm_a8w8_launch as _impl

        return _impl(*args, **kwargs)

    def opus_gemm_a8w8_blockscale_launch(*args, **kwargs):
        from .gemm_op_a8w8 import opus_gemm_a8w8_blockscale_launch as _impl

        return _impl(*args, **kwargs)

    def opus_gemm_a8w8_blockscale_bpreshuffle_launch(*args, **kwargs):
        from .gemm_op_a8w8 import (
            opus_gemm_a8w8_blockscale_bpreshuffle_launch as _impl,
        )

        return _impl(*args, **kwargs)

else:
    # Keep the top-level aiter import available on unsupported architectures.
    gemm_a16w16_opus = _make_unsupported_arch_stub("gemm_a16w16_opus")
    opus_gemm_a16w16_launch = _make_unsupported_arch_stub(
        "opus_gemm_a16w16_launch"
    )
    opus_gemm_a16w16_tune = _make_unsupported_arch_stub("opus_gemm_a16w16_tune")
    opus_gemm_a8w8_blockscale_bpreshuffle_tune = _make_unsupported_arch_stub(
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune"
    )
    opus_gemm_a8w8_launch = _make_unsupported_arch_stub(
        "opus_gemm_a8w8_launch"
    )
    opus_gemm_a8w8_blockscale_launch = _make_unsupported_arch_stub(
        "opus_gemm_a8w8_blockscale_launch"
    )
    opus_gemm_a8w8_blockscale_bpreshuffle_launch = (
        _make_unsupported_arch_stub(
            "opus_gemm_a8w8_blockscale_bpreshuffle_launch"
        )
    )

    def opus_gemm_workspace_init() -> None:
        """Deprecated no-op on every architecture."""
        from .gemm_op_a16w16 import opus_gemm_workspace_init as _impl

        return _impl()


__all__ = [
    "gemm_a16w16_opus",
    "opus_gemm_a8w8_launch",
    "opus_gemm_a8w8_blockscale_launch",
    "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
    "opus_gemm_a16w16_launch",
    "opus_gemm_a16w16_tune",
    "opus_gemm_workspace_init",
]
