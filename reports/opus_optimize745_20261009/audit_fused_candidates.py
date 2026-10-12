#!/usr/bin/env python3
"""Scalar grids/fragments and static publication ISA for private fusion families."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    builds = []
    for directory, prefix in (("fused_register","fused_"),("fused_lds","fused_lds_"),
                              ("fused_wavek","fused_wavek_")):
        path = HERE / directory
        build = json.loads((path / "build.json").read_text())
        assert build["returncode"] == 0 and sha(path / "experiments.so") == build["library_sha256"]
        for name, expected in build["sources"].items():
            assert sha(path / name) == expected
        builds.append((path,prefix,build))
    grid_runs = resource_checks = 0
    layouts = []
    for path,prefix,build in builds:
        isa = (path / "device.s").read_text()
        labels = list(re.finditer(r"^([0-9a-f]+) <([^>]+)>:$",isa,re.M))
        functions = {match.group(2):isa[match.end():labels[i+1].start() if i+1<len(labels) else len(isa)]
                     for i,match in enumerate(labels)}
        for row,kernel in zip(build["configurations"],build["kernels"]):
            assert all(kernel["resources"][key] == 0 for key in
                       ("private_segment_fixed_size","vgpr_spill_count","sgpr_spill_count"))
            code=functions[kernel["name"]]
            atomic=re.search(r"global_atomic_add .*sc0 sc1",code)
            assert atomic and "buffer_wbl2 sc1" in code[:atomic.start()]
            assert re.search(r"s_waitcnt.*vmcnt\(0\)",code[:atomic.start()])
            assert re.search(r"buffer_load_dwordx4 .*sc0 sc1",code[atomic.end():])
            assert re.search(r"global_store_dword .*sc0 sc1",code[atomic.end():])
            c=row["compile_params"]
            tm,tn,wm,wn=(c[k] for k in ("tile_m","tile_n","wave_m","wave_n"))
            threads=wm*wn*64;em,en=tm//(wm*16),tn//(wn*16)
            positions=set()
            for fragment in range(em*en):
                for thread in range(threads):
                    for element in range(4):
                        offset=(fragment*threads+thread)*4+element
                        assert offset not in positions
                        positions.add(offset)
            assert positions == set(range(tm*tn))
            # MFMA16's four elements per lane cover the tile once; the same
            # fragments are used for producer stores and deterministic sums.
            output_positions=set()
            for fragment in range(em*en):
                for thread in range(threads):
                    lane,wave=thread%64,thread//64
                    for element in range(4):
                        rr=(fragment//en*wm+wave%wm)*16+lane%16
                        nn=(fragment%en*wn+wave//wm)*16+lane//16*4+element
                        assert (rr,nn) not in output_positions
                        output_positions.add((rr,nn))
            assert len(output_positions)==tm*tn
            for m in (1,15,16,17,31,32,47,48,63,64,65,112,129):
                for n in (128,256):
                    tiles=(m+tm-1)//tm*(n//tn)
                    for split in (1,2,3,4,8,16):
                        seen=Counter()
                        for bid in range((tiles+7)//8*8*split):
                            tile=bid//(8*split)*8+bid%8
                            if tile<tiles:seen[tile,bid//8%split]+=1
                        assert len(seen)==tiles*split and set(seen.values())=={1}
                        grid_runs+=1
            for case in plan["cases"]:
                if case["M"]>c["max_m"]:continue
                for split in range(1,min(16,case["K"]//128)+1):
                    total=case["K"]//128
                    parts=[(i*(total//split)+min(i,total%split),total//split+(i<total%split))
                           for i in range(split)]
                    assert parts[0][0]==0 and parts[-1][0]+parts[-1][1]==total
                    assert all(a[0]+a[1]==b[0] for a,b in zip(parts,parts[1:]))
                    for begin,count in parts:
                        wk=c["wave_k"]
                        local=[(begin+i*(count//wk)+min(i,count%wk),count//wk+(i<count%wk))
                               for i in range(wk)]
                        assert local[0][0]==begin and local[-1][0]+local[-1][1]==begin+count
                        assert all(a[0]+a[1]==b[0] for a,b in zip(local,local[1:]))
                    if row["pipeline"] == "lds":
                        loops=(total+split-1)//split
                        matrix=min(loops,c["stages"])*(tm+tn)//8*1056
                        scales=(tm+(tn+127)//128)*loops
                        assert matrix+scales+kernel["resources"]["group_segment_fixed_size"]<=160*1024
                    resource_checks+=1
            layouts.append({"tag":prefix+str(row["id"]),"threads":threads,"tile_elements":tm*tn,
                            "publication_isa":"passed","fragment_mapping":"complete_disjoint"})
    # Execute the real AST adapter for both private families with fake C ABI
    # calls, verifying pointers, dimensions, stream and library selection.
    tree=ast.parse((HERE/"benchmark.py").read_text())
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
    adapter=next(n for n in ast.walk(main_node) if isinstance(n,ast.FunctionDef) and n.name=="variant")
    calls=[]
    def fake_library(family):
        def launch(*args):
            calls.append((family,args))
            return 0
        return SimpleNamespace(launch=launch)
    class Pointer:
        def __init__(self,value):self.value=value
        def data_ptr(self):return self.value
    pool=[([Pointer(i*4096) for i in range(1,8)],)]
    env={"fused_configs":{},"fused_libraries":{},"workspace_size":lambda *args:64,
         "m":17,"n":128,"k":384,
         "torch":SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=8192)))}
    exec(compile(ast.Module(body=[adapter],type_ignores=[]),str(HERE/'benchmark.py'),'exec'),env)
    for path,prefix,build in builds:
        row=build['configurations'][0];tag=prefix+str(row['id'])
        env['fused_configs'][tag]=row;env['fused_libraries'][tag]=fake_library(row['pipeline'])
        task={'config':tag,'split_k':3}
        actual_tag,run=env['variant'](task,pool)
        assert actual_tag==tag+':3' and run(0) is pool[0][0][4]
        assert calls[-1]==(row['pipeline'],(row['id'],4096,8192,12288,16384,20480,24576,28672,17,128,384,3,8192))
    report={'status':'passed','configurations':len(layouts),'full_grid_enumerations':grid_runs,
            'legal_split_resource_checks':resource_checks,'layouts':layouts,'private_adapter_libraries':len(builds),
            'benchmark_sha256':sha(HERE/'benchmark.py'),'kernel_launches':0,
            'scope':'CPU grid, fragment, LDS extent and adapter checks; static ISA only; GPU validation pending',
            'build_sha256':{str(path.relative_to(HERE)):sha(path/'build.json') for path,_,_ in builds}}
    (HERE/'fused_candidates_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='layouts'}))


if __name__=='__main__':
    main()
