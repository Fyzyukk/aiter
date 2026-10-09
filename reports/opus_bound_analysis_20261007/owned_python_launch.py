#!/usr/bin/env python3
"""Run a Python target in the same mm after identifying its KFD host PID.

Opening KFD registers the process without allocating GPU memory or submitting
work. The parent must acknowledge an unambiguous sysfs identity before the
target may import HIP. Keeping this process and fd avoids exec/mm teardown.
"""
import json
import os
from pathlib import Path
import runpy
import select
import sys


def main():
    target = sys.argv[1:]
    if not target or not Path(target[0]).is_file():
        raise RuntimeError("owned launcher requires a Python script")
    ready_fd = int(os.environ.pop("OPUS_OWNER_READY_FD"))
    ack_fd = int(os.environ.pop("OPUS_OWNER_ACK_FD"))
    nonce = os.environ.pop("OPUS_OWNER_NONCE")
    kfd_fd = os.open("/dev/kfd", os.O_RDWR | os.O_CLOEXEC)
    try:
        ready = {"inner_pid": os.getpid(), "nonce": nonce, "kfd_fd": kfd_fd}
        os.write(ready_fd, (json.dumps(ready) + "\n").encode())
        os.close(ready_fd)
        if not select.select([ack_fd], [], [], 15)[0]:
            raise RuntimeError("KFD host identity acknowledgement timed out")
        ack = json.loads(os.read(ack_fd, 8192))
        os.close(ack_fd)
        if ack.get("nonce") != nonce or not isinstance(ack.get("host_pid"), int):
            raise RuntimeError("Invalid KFD host identity acknowledgement")
        sys.argv = target
        sys.path.insert(0, str(Path(target[0]).resolve().parent))
        runpy.run_path(target[0], run_name="__main__")
    finally:
        os.close(kfd_fd)


if __name__ == "__main__":
    main()
