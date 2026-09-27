"""Verify physical PCI/KFD identity and three idle samples; no GPU allocations."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import time

from machine import DEVICES


def pci_parts(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    if not match:
        raise ValueError(f"Invalid PCI address: {value!r}")
    domain, bus, device, function = (int(v, 16) for v in match.groups())
    if device > 31 or function != 0:
        raise ValueError("This harness requires PCI function 0 and a valid PCI device")
    return domain, bus, device, function


def discover(gpus):
    nodes = {}
    for path in Path("/sys/class/kfd/kfd/topology/nodes").glob("*/properties"):
        fields = dict(line.split(maxsplit=1) for line in path.read_text().splitlines()
                      if len(line.split(maxsplit=1)) == 2)
        if int(fields.get("unique_id", 0)):
            key = int(fields.get("domain", 0)), int(fields["location_id"])
            nodes[key] = f"GPU-{int(fields['unique_id']):016x}"
    cards = json.loads(subprocess.run(["rocm-smi", "--showbus", "--json"],
                       text=True, capture_output=True, timeout=10, check=True).stdout)
    result = {}
    for gpu in gpus:
        pci = cards[f"card{gpu}"]["PCI Bus"].lower()
        domain, bus, device, function = pci_parts(pci)
        uuid = nodes.get((domain, (bus << 8) | (device << 3) | function))
        if uuid is None:
            raise ValueError(f"No KFD UUID for physical card {gpu}, PCI {pci}")
        result[gpu] = dict(uuid=uuid, pci_bus=pci, domain=domain, bus=bus,
                           device=device, function=function)
    return result


def number(value):
    text = str(value).strip().rstrip("%").strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", text):
        raise ValueError(f"Unavailable GPU telemetry: {value!r}")
    result = float(text)
    if not math.isfinite(result):
        raise ValueError("Nonfinite GPU telemetry")
    return result


def read_sample(gpus):
    command = ["rocm-smi", "-d", *map(str, gpus), "--showuse", "--showmemuse",
               "--showmeminfo", "vram", "--json"]
    records = json.loads(subprocess.run(command, text=True, capture_output=True,
                                       timeout=10, check=True).stdout)
    sample = {"time": datetime.now(timezone.utc).isoformat(), "cards": {}}
    for gpu in gpus:
        raw = records[f"card{gpu}"]
        used_key = next(k for k in raw if "vram" in k.lower() and "used" in k.lower()
                        and "(b)" in k.lower())
        sample["cards"][str(gpu)] = dict(raw=raw, use=number(raw["GPU use (%)"]),
            gfx=number(raw["GFX Activity"]), vram=number(raw[used_key]))
    return sample


def check(gpus, gpu_map):
    identities = discover(gpus)
    for gpu in gpus:
        if identities[gpu] != DEVICES[gpu] or gpu_map[gpu] != (DEVICES[gpu]["uuid"], DEVICES[gpu]["bus"]):
            raise RuntimeError(f"GPU identity changed since preparation: physical GPU {gpu}")
    samples = []
    for index in range(3):
        if index:
            time.sleep(5)
        samples.append(read_sample(gpus))
    for gpu in gpus:
        cards = [row["cards"][str(gpu)] for row in samples]
        if (any(c["use"] != 0 or c["vram"] >= 2 * 1024**3 for c in cards)
                or len({c["gfx"] for c in cards}) != 1):
            raise RuntimeError(f"Physical GPU {gpu} is not idle; choose a fresh batch after it is free")
    return dict(status="passed", identities=identities, idle_samples=samples,
                policy="Matching physical PCI/KFD UUID; GPU use=0, VRAM<2 GiB and unchanged GFX across three samples")
