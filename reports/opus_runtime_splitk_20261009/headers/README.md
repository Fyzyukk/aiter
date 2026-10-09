The five new C++ headers implement runtime global split-K for three fine LDS
geometries and six register geometries. Fine producers have two static output
modes; register producers share compute/local reduction before selecting BF16
direct output or FP32 workspace output. All counts use one balanced-partition
helper and one runtime reducer per vector/block geometry. Existing headers
remain byte-identical.

The final aggregate device compile and host syntax check passed with Clang 23.
The aggregate instantiates fourteen entry points: six fine producer output
modes, six register producers, and vector widths four/sixteen for the reducer.
Every device entry has zero scratch, VGPR spills, and SGPR spills. This is an
offline compilation result; numerical and performance GPU validation were not
run.

The isolated CPU check exercises the actual partition helper for total K128
tiles zero through 128 and split counts one through sixteen: 2,064 global
cases and 17,544 nested local WaveK-two cases. It checks LDS bounds for 6,192
fine launches, actual production B layout expressions, A producer/consumer
byte identities, and local register partial storage. The isolated executable
was linked with a plain C++ driver, and its dynamic dependencies were checked
before execution. It has no HIP, HSA, CUDA, or Torch library dependency.

An initial CPU check was linked and executed using the HIP driver, which
automatically added `libamdhip64.so.7`. That run made no HIP API calls in the
check, but loaded the HIP library and therefore violated the library-load
restriction. Its artifacts are retained (`cpu_header_check`, `cpu_run.log`,
`initial_hip_linked_cpu_dynamic.txt`); the final receipt records this incident
without claiming zero session library loads. The later isolated CPU run
passed after dependency inspection (`isolated_cpu_dynamic.txt`,
`isolated_cpu_run.log`). A plain C++ attempt failed to parse device-only Opus
expressions and was not executed (`plain_cpu_compile.log`).

`receipt.json` describes the final verification and incident. `manifest.json`
freezes all files under this directory except the manifest itself. Production
source dependencies are copied under `frozen/`, with their hashes recorded in
`source_inputs.json`. Run `python3 reproduce.py NEW_OUTPUT_DIRECTORY` to repeat
the frozen compile and CPU checks; it refuses to overwrite an existing output
directory and checks CPU dynamic dependencies before execution.
