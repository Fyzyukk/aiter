# Latest recovery: v19, 2026-10-09 18:05 UTC

Goal active and incomplete. OPUS must beat FlyDSL BMM stably on all745 shapes,
pass signed/reference/guard checks, and integrate into production parameterized
dispatch. Current-source valid performance0/745. Historical BMM-only428/317
remains historical. Normal numerical smoke2/9, private smoke not started.

Live watcher PID2059925, exec session23410, log`watch_stdout_v19.log`.
Verified live process and fresh status after restart. v18PID2052695 was stopped
only while idle, exit130confirmed. Check currentPID before any restart.
Physical AMD-SMI0–3 only; monitor all8. Optional range question unanswered.
At restart front-four owners2931581/2932814/2932123/2929501, gfx100%,~284GiB/card.

v19 fixed actual private C++ host contracts discovered by a CPU rejection check:

- Single-wave register launcher previously acceptedM513 although config maxM512.
  It now rejectsM>512. Register and local-WaveK requirecounter4bytealignment.
- LDS launcher now requiresSFA16byte/SFB4bytealignment,matching current config.
- Build generators updated to preserve guards on future generation.
- All3priorprivate libraries/builds/ISA retained in`private_libraries_before_v19/`.
- All3libraries fresh CPU rebuilt with unchanged device source.22device kernel
  images are byte-identical to archived deviceSHA;resources identical,scratch/spill0.
  Newbuilds savepreviouslibrarySHA,hostcontractrevisionv19anddeviceunchanged.
- `audit_fused_host.py`extractsactualtraits/argumentheaders/completehostlaunchers,
  replacesdevice declarations withCPU dummies,and recordsHIPlauncharguments.
  Host-onlyHIPcompile thenplainC++link;readelf rejectsHIP/HSA dependencies.
  22configs/88validplans/279rejectedcalls passed;host/deviceargblocks120bytes.
  Checksgrid,block,LDS,pointers,dimensions,strides,split,stream,kernelidentity.
- `fused_host_audit_before_v19/`preservesM513rejectionfailure;
  `fused_host_audit_v19/report.json`preservespassingresults.
- Combinedscalar/ISAfusionaudit rerun:3432grids/106088legalchecks/3adapterspassed.

Benchmark,watcher,KFDhelper and planunchanged. Existing two owner-isolated
numerical smokes remain reusable:1×768×7168 and16×2048×7168;41variants,
39publicAPIvariants,451signedchecks,invalid0,guardspassed. Newprivate receipts
bindchangedlibrary/build hashes,so oldprivate data could not be silently reused.
`evidence_audit_v19.json`revalidates numerical receipts and correctly reports
incomplete evidence. No newGPUperformanceclaim orproductionCSV/sourcechange.

Continue normal9shape smoke,private10shape smoke,745baseline,measuredloss
parametersearch,allsplits,legalMprivatefusion,replay745. Allshape workerspersist
independently. Inspectnewlogs/results immediatelywhenallowedGPUbecomesfree.
Ifprivatewinnersselected,integrateparameterizedproductionthenfreshpublic745run.
Ifunresolvedremain,continuefrommeasuredgaps. CPUbuilds/watcher/partialsmoke
cannot completegoal. v18KFDqueue/VRAM/unknowngatesremainactive.
