import sys, subprocess, re
img=sys.argv[1]
out=subprocess.run(['/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/llvm-readelf','--notes',img],capture_output=True,text=True).stdout
blocks=re.split(r'\n\s+- \.', out)
for b in blocks:
    m=re.search(r'\.name:\s+(\S+)', b)
    if not m or 'kernel' not in m.group(1): continue
    nm=m.group(1)
    t=re.search(r'seqIJLi(\d+)ELi(\d+)ELi256',nm); md=re.search(r'Li1ELi1ELi32EEEELi(\d)ELb(\d)',nm)
    g=lambda k: (re.search(r'\.%s:\s+(\d+)'%k, '.'+b) or [None,'?'])[1]
    print(f"{t.group(1)}x{t.group(2)} mode{md.group(1)} bnd{md.group(2)}: vgpr {g('vgpr_count')} agpr {g('agpr_count')} vgpr_spill {g('vgpr_spill_count')} sgpr_spill {g('sgpr_spill_count')} lds {g('group_segment_fixed_size')} scratch {g('private_segment_fixed_size')}")
