#!/usr/bin/env python3
"""Prepare stage3 pipeline/parent repeat plan on CPU; never run benchmarks."""
import sys
from analyze_stage3_existing import main

if __name__ == "__main__":
    if "--write-repeat" not in sys.argv:
        sys.argv.append("--write-repeat")
    main()
