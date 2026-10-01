# S1 startup preflight and runtime-profile boundary

Status: implemented for the S1 candidate and covered by focused WSL development
tests. Native Windows process containment, installed-service behavior, CUDA
behavior, and every sealed acceptance case remain `NOT_RUN`.

## Startup order

The service process performs profile selection without importing Torch or any
other ML backend. `select_profile` reads only installed Torch package metadata.
An omitted profile chooses the exact CPU or CUDA build for the current release
OS; missing or unknown Torch selects the OS CPU profile with an unavailable
backend reason. An explicit OS or build mismatch raises
`BACKEND_VERSION_MISMATCH` before the caller may publish the storage-root lock.
The immutable `ProfileSelection` is the device authority for the instance.

The launcher then runs these bounded operations before lock publication:

1. `run_gpu_query` invokes only the fixed native-Windows or WSL `nvidia-smi`
   path with the two fixed query arguments, a 10-second deadline, and a 4,096
   byte stream ceiling.
2. `run_preflight_child` invokes the selected interpreter followed by exactly
   `-I -m llm_foundations_companion.preflight_main`, with the RUN-002
   allowlisted environment, a 30-second deadline, and 65,536-byte stdout and
   stderr ceilings.
3. The parent service performs the 4,096-byte write, digest readback, delete,
   and absence checks represented by `StorageProbeResult`. The child performs
   no storage, model, dataset, network, inference, user-code, or shell work.
4. `require_startup_allowed` is called before any root lock or listener is
   published. A failed CUDA profile raises a stable reason and includes the CPU
   launch command; it never starts a CPU instance. A CPU profile remains
   startable with a missing or mismatched backend so the reader and corrective
   instructions remain available.
5. `build_preflight_receipt` combines the six P00 rows, observed dependency
   versions, storage observations, and CUDA result without recording an
   interpreter path or probe filename. Receipt registration is owned by the
   storage layer after a writable probe.

The fixed child imports Torch, Transformers, PEFT, Accelerate, and Safetensors;
checks their exact profile pins; allocates the selected device; and, for CUDA,
applies the device, architecture, memory, and numeric checks. The parent applies
the complete seven-step CUDA decision order, beginning with the separate GPU
query and Windows-driver rule.

## Process and output containment

The bounded runner uses no shell and starts a new WSL process group. Deadline or
stream failure signals only that retained group; completion also closes the
group boundary so a child cannot leave a descendant behind. On Windows, the
runner creates a per-child Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
retains its handle, and fails closed if configuration or assignment fails.
Closing the handle terminates the owned process tree. The runner never scans by
name and never accepts a PID from child output.

Both callers enforce their byte limits independently of the runner's flags.
Malformed UTF-8, duplicate JSON keys, non-finite JSON, additional properties,
timeouts, nonzero exits, and oversized output become a closed synthetic
`BACKEND_MISSING` child result. They do not raise an unhandled startup error or
enable a capability.

## S1 capability boundary

`capabilities_from_preflight` exposes all 15 operation capability keys required
by RuntimeInfo. Every key remains false with `CAPABILITY_UNAVAILABLE` in S1.
When the backend is missing or mismatched, the same closed keys remain false
with `BACKEND_MISSING` or `BACKEND_VERSION_MISMATCH`. `require_capability`
rejects such work before scheduler insertion.

## Verification boundary

Focused tests cover exact metadata selection, invalid OS/profile pairs, fixed
GPU invocation and hardware thresholds, exact child invocation and schema,
malformed and oversized output, the seven CUDA decisions and first-failure
order, CPU-reader startup with an unavailable backend, failed-CUDA startup,
receipt schema/digest behavior, and closed false S1 capabilities. Real WSL
process tests cover the exact 4,096 and 65,536-byte boundaries, over-limit
termination, the deadline, launch failure, and owned-descendant cleanup. The
Windows Job Object path is source-reviewed but requires a native Windows run;
this note does not claim it passed. AC-RUN-020, AC-RUN-024, AC-RUN-025, full
installed service startup, HTTP readback, CUDA, accessibility, and product
acceptance remain `NOT_RUN`.
