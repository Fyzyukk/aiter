#!/usr/bin/env python3
"""Read per-device KFD queue/VRAM owners without creating GPU contexts."""
from pathlib import Path


KFD_ROOT = Path("/sys/class/kfd/kfd")
# Persistent read-only registrations observed in this environment use 2 MiB
# and no queues. Active tensor allocations or any queue indicate ownership.
REGISTRATION_BYTES = 8 * 1024 * 1024


def gpu_ids_by_bdf(root=KFD_ROOT):
    result = {}
    for node in (root / "topology/nodes").iterdir():
        gpu_id = int((node / "gpu_id").read_text())
        if gpu_id == 0:
            continue
        properties = dict(line.split() for line in (node / "properties").read_text().splitlines())
        location = int(properties["location_id"])
        domain = int(properties["domain"])
        bdf = f"{domain:04x}:{location >> 8:02x}:{(location >> 3) & 31:02x}.{location & 7}"
        assert bdf not in result, f"Duplicate KFD BDF: {bdf}"
        result[bdf] = gpu_id
    assert result, "KFD GPU topology is unavailable"
    return result


def snapshot(gpu_ids, root=KFD_ROOT):
    """Return allocation/queue owners, or raise if ownership is unreadable.

    A PID disappearing during the scan is no longer an owner. Other read
    failures remain unknown and callers must not treat them as an empty list.
    """
    owners = {gpu_id: [] for gpu_id in gpu_ids}
    for process in (root / "proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            vram = {gpu_id: int((process / f"vram_{gpu_id}").read_text()) for gpu_id in owners}
            queues = {gpu_id: [] for gpu_id in owners}
            for queue in (process / "queues").iterdir():
                try:
                    gpu_id = int((queue / "gpuid").read_text())
                    queue_type = int((queue / "type").read_text())
                except FileNotFoundError:
                    # A removed queue is no longer scheduled.
                    if not queue.exists():
                        continue
                    raise
                if gpu_id in queues:
                    queues[gpu_id].append({"id": queue.name, "type": queue_type})
        except FileNotFoundError:
            if not process.exists():
                continue
            raise
        for gpu_id in owners:
            if vram[gpu_id] > REGISTRATION_BYTES or queues[gpu_id]:
                owners[gpu_id].append({"pid": int(process.name), "vram_bytes": vram[gpu_id],
                                       "queues": queues[gpu_id]})
    return owners


def external(owners, host_pid):
    return [owner for owner in owners if owner["pid"] != host_pid]
