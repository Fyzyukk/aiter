# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CLI used by Ninja without importing aiter or initializing HIP."""

from opus_compiler import main


if __name__ == "__main__":
    raise SystemExit(main())
