# Small flow v4

One runtime U1 matrix-ring loop and one BF16 vector8 output serve both tile
instances. K is never a template argument and never selects a separate flow.

| ID | Tile | Matrix ring | Scale groups per panel | LDS bytes |
|---|---|---:|---:|---:|
| 21310 | 128x128x128 | 3 slots | 32 | 105504 |
| 21311 | 160x128x128 | 2 slots | 32 | 81184 |

The 128 tile trades its second possible resident CTA for lower single-CTA
pipeline latency. Pilot2's narrow-N regressions have 48..256 CTAs over 256
CUs, where that second resident CTA would be unused. The 160 tile retains
its two-slot allocation and can still fit two CTAs within 160 KiB LDS.
This is a shape-constant scheduling choice, with common source for prefetch,
consume, ring advancement, scale refill, and output.

The scale panel stays raw, as in v2/v3. Increasing its capacity from 16 to 32
cuts K7168 refills from three to one and K16384 refills from seven to three.
Loads remain conditional on the actual runtime group count. The 160 tile
needs two static scale-load passes; the second pass has no active loads for
short K. Packed scales and the production 9010 XOR A layout are not added
in this candidate, so the measurements can isolate ring/panel changes.

## Ring ownership and synchronization

An advance enters with group g in registers, chooses next=(stage+1)%S, and
issues group g+2 to future=(stage+2)%S when that group exists. Every producer
wave issues exactly 8 direct-to-LDS requests for M128 or 9 for M160.
After the first current M repeat, vmcnt(8/9) retires older group g+1 requests
while leaving current group g+2 requests in flight. If no future exists,
vmcnt(0) drains the queue. A publication barrier precedes every next operand
read in both instances.

For S2, future is the current slot. The following advance writes g+3 into
the just-read g+1 slot, so the end lgkmcnt(0) and consumer barrier remain.
The seed likewise retains its reader barrier before overwriting K0 with K2.

For S3, future is the third slot, distinct from current and next. The
following advance writes g+3 into the older g slot. All g operand reads
precede the current advance's publication barrier, so that barrier already
retires every wave's g readers before g+3 can overwrite the slot. The end
consumer barrier is unnecessary: outstanding g+1 reads target another
slot. The seed first writes K2 into its independent third slot, and the
first publication barrier protects the later K3 overwrite of K0.

At a panel refill, the current scales were read before the preceding
publication barrier; every wave has therefore finished its old-panel
reads. Current scales remain in registers. The existing scale refill
waits for VMEM/LDS completion and publishes its stores, also safely draining
outstanding matrix requests at that boundary.

The final advance has no future and drains VMEM. S2 retires its final
operand readers with its normal end barrier. S3 performs one lgkmcnt(0)
plus barrier before output reuses matrix LDS. For K128, the seed drains
the only matrix group and this same S3 pre-output barrier protects LDS
reuse. There is one output flow for all K and both tile heights.

## Static build evidence

Pinned-clang compilation with machine-instruction verification passed.
Resources are VGPR200/AGPR64/SGPR55 for M128 and
VGPR240/AGPR80/SGPR59 for M160; all spill counts and private bytes are zero.
The parent report contains `small_v4_device_audit.json`,
`small_v4_queue_audit.json`, and `small_v4_isa.txt`.
The ISA audit confirms actual 8/9-request future blocks, partial waits,
zero-count drains, and a publication barrier before the first next operand
read. S3 has no end-of-advance barrier and has a consumer barrier before its
first output LDS store; S2 retains its end consumer barrier. GPU validation
and timing are delegated to the shared batch.
