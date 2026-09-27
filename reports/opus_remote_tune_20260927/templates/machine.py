"""Machine configuration prepared locally; importing this module never queries a GPU."""

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "csrc/opus_gemm/opus_gemm_common.py").is_file())
CONFIG_PATH = HERE / "machine_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.is_file() else {}
DEVICES = {int(k): v for k, v in CONFIG.get("gpus", {}).items()}
GPU_MAP = {index: (row["uuid"], row["bus"]) for index, row in DEVICES.items()}
GPU_INDEX = int(os.environ.get("OPUS_TUNE_GPU_INDEX", str(min(DEVICES, default=0))))
DEVICE = DEVICES.get(GPU_INDEX, {})
GPU_UUID = DEVICE.get("uuid", "")
GPU_BUS = DEVICE.get("bus", -1)
GPU_DOMAIN = DEVICE.get("domain", -1)
GPU_DEVICE = DEVICE.get("device", -1)
