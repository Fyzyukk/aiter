# Latest recovery: v16, 2026-10-09 17:37 UTC

The all745 performance goal remains active and incomplete. Historical BMM-only
comparison:428 wins/317 losses. No current-source performance shape measured yet.

Live watcher PID2046250, exec session28402, log`watch_stdout_v16.log`.
Check the livePID and`status.json` before any restart. PhysicalSMI0-3 only.
Current external owners:2931581/2932814/2932123/2929501;activity100%,~284GiB/card.
The optional question about extending to8cards is unanswered;do not infer approval.

Current-source validGPU evidence now exists:

- GPU0/PCI0000:05:00.0 brieflyidle around17:29:40–17:29:59UTC.
- `smoke_0000_8aec31a05369.1791566980047369445`:1x768x7168,
  18numericalvariants,17publicAPIvariants,198signedchecks,invalid0.
- `smoke_0001_779287cc56cb.1791566991159003697`:16x2048x7168,
  23numericalvariants,22publicAPIvariants,253signedchecks,invalid0.
- Bothhave validowner-isolatedreceipts,script/plan/outputSHAchecked.
  `validated_smoke_first2.json`binds41variants/39publicAPI/451signedchecks.
- Earlier contaminatedsmoke remains excluded. Totalworkers3,validreceipts2.
- Another externaljobheld~38GiB onfront4afterwindow,thennewprocesses occupied
  front4. Do not loosenidle thresholds becauseAMD-SMI processlist brieflyempty;
  KFD sysfs showed2918269VRAMownership duringthatperiod.

This continuation changed:

- v14checks everytouchedoutputagainst originalFP32bounds aftereachprofiler
  timinground,outsidemeasurement;BMMcommon tolerance retained. Resultsstore
  `post_timing_error_ratios`. Failurewrites`post_timing_numerical`andaborts.
- `fused_wavek/`6private register localWaveK2/4+global last-arrival variants;
  16x16/16x32/32x32,prefetch3,Bcache2. Allwavesremainliveforarrivalbarriers;
  K-wave0owns localFP32sum andglobalpartial. Fullsame-streamcounterreset.
  `build_fused_wavek.py`6/6compiled,scratch/spill0,VGPR72-138,
  staticLDS1028-12292bytes;GPUdynamicvalidationpending.
- `build_wavek_controls.py`6/6public exactcontrols CPUprebuilt,noGPUcontexts,
  libraryloads orlaunches. Controlsoutsideoriginal204plan areloadedfrommetadata.
- v15loads3privateCABI libraries,tags`fused_0..9`,`fused_lds_0..5`,
  `fused_wavek_0..5`. Watchreceiptdependenciesincludeallthree source/.so/buildsets.
- Fusion22configs,smoke10shapes/all7Klengths,2126OPUStasks,
  1058private/1068public;`fused_smoke_coverage_v15.json`currentcoverage.
- `audit_fused_candidates.py`22configs/3432fullgrids/106088legal split/resource
  checks,includingemptylocalwavepartitions;3actualASTCABIadapterchecks and
  publication/atomic/bypass/resetstaticISA passed. DynamicGPUpending.
- v16shortensidlepollintervalto2seconds(three1secondSMIsamplesretained).
  Baseline745andfinalreplay745nowoneworker/receiptpershape,retainsvalidwork
  acrossshortwindows. Benchmarkscriptunchangedsincevalidfirst2smoke,
  so those areautomaticallyreused bywatcher.

No productionkernels orformaltunedCSV modified. Originalplan frozen source/input
hashes unchanged. AST/pyflakes/diffchecks pass. Previousscriptrevisions SHAarchived.

Next: inspectlivewatcher. WhenallowedGPUfree,inspectnextsmokelog immediately.
Finishnormal9shape smoke,private10shape smoke,baseline745,parametersearch,
allsplits,legalMfusion,final745replay. If unresolved,continuefrom measuredgaps.
Private winners require parameterizedproductionintegration+newpublicAPI745run.
Do not markgoalcompletefromCPUbuilds orpartialnumericalchecks.
