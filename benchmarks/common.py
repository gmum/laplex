"""Shared measurement protocol for the LAPLEX benchmark reproductions.

Every figure script builds a list of *series* (method x mode x metric), sweeps
one size axis and hands each point to :class:`Sweep`, which implements the
stopping rule used by every figure:

    A series is measured at every size until it fails to fit in device memory
    (a real ``OutOfMemoryError`` / allocator failure).  The first failing size
    is recorded with status ``OOM``; the series is not attempted at any larger
    size.  There is no wall-clock or VRAM *budget* in the final protocol.

``--time-limit-ms`` exists only to iterate quickly while developing: a series
whose single probe call exceeds the limit is stopped with status ``TIMEOUT``,
which the plots draw differently from ``OOM``.  Final runs must not set it.

Timing follows App. D of the paper: ``repeats`` independent runs of ``iters``
timed calls after ``warmup`` calls, with device synchronisation around the
timed region.  All repeats are stored; figures pick best-of-repeats (App. D)
or mean +- std (Figs. 1/15).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RESULTS = HERE / "results"
CACHE = HERE / ".cache"

os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# The stopping rule asks whether a computation fits in device memory, not whether the caching
# allocator can find a contiguous cached block: without this, dense fwd+bwd at n = 2^16 (B = 1,
# ~64 GiB) fails on an 80 GB H100 with ~16 GiB "reserved but unallocated".
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

MiB = 1024 ** 2

OK, OOM, TIMEOUT, ERROR, LIMIT = "ok", "OOM", "TIMEOUT", "ERROR", "LIMIT"
REF_OOM = "REF_OOM"       # accuracy: the fp64 reference does not fit, so no error can be computed
INT32_MAX = 2 ** 31 - 1   # element-index range of the custom CUDA kernels (int indexing)


def int32_limit(elems_of):
    """Op.limit for custom kernels: stop once a tensor would exceed int32 element indexing."""
    def limit(x):
        e = elems_of(x)
        return f"{e} elements per tensor > int32 indexing of the custom kernels" if e > INT32_MAX else None
    return limit


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
@dataclass
class Protocol:
    warmup: int = 10
    iters: int = 30
    repeats: int = 3
    time_limit_ms: float | None = None


def make_parser(description: str, *, min_log: int, max_log: int,
                repeats: int = 3, warmup: int = 10, iters: int = 30,
                device: str = "cuda") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["run", "plot", "all"], default="all")
    p.add_argument("--min-log", type=int, default=min_log, help="smallest size = 2**min_log")
    p.add_argument("--max-log", type=int, default=max_log, help="largest size = 2**max_log")
    p.add_argument("--repeats", type=int, default=repeats)
    p.add_argument("--warmup", type=int, default=warmup)
    p.add_argument("--iters", type=int, default=iters)
    p.add_argument("--time-limit-ms", type=float, default=None,
                   help="DEVELOPMENT ONLY: stop a series whose probe call is slower "
                        "than this (status TIMEOUT). Leave unset for final runs.")
    p.add_argument("--quick", action="store_true",
                   help="development preset: warmup=2 iters=5 repeats=1 time-limit=2000 ms "
                        "and a '-quick' suffix on the results tag")
    p.add_argument("--device", default=device, choices=["cuda", "cpu"])
    p.add_argument("--tag", default=None,
                   help="results sub-directory (default: derived from the device name)")
    p.add_argument("--tags", nargs="*", default=None,
                   help="plot stage: result tags to overlay (default: --tag / current device)")
    p.add_argument("--seed", type=int, default=0, help="base seed; repeat r uses seed+r")
    p.add_argument("--ylim-json", nargs="*", default=[],
                   help="plot stage: <stem>.yrange.json of other versions; y ranges become their union")
    p.add_argument("--paper", action="store_true",
                   help="plot stage: version for the manuscript, <stem>_paper.{pgf,pdf}: the published "
                        "figure size and a y range fitted to the data (see plotting.py)")
    return p


def protocol_from_args(args) -> Protocol:
    if args.quick:
        return Protocol(warmup=min(args.warmup, 2), iters=min(args.iters, 5), repeats=1,
                        time_limit_ms=args.time_limit_ms or 2000.0)
    return Protocol(args.warmup, args.iters, args.repeats, args.time_limit_ms)


# ---------------------------------------------------------------------------
# Device / environment
# ---------------------------------------------------------------------------
def device_tag(device: str) -> str:
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
        for key, tag in [("H100", "H100"), ("A100", "A100"), ("4090", "RTX4090"),
                         ("5090", "RTX5090"), ("2080 Ti", "RTX2080Ti"), ("2080", "RTX2080"),
                         ("3080", "RTX3080")]:
            if key in name:
                return tag
        return name.replace("NVIDIA", "").strip().replace(" ", "")
    return "CPU-" + socket.gethostname()


def setup_extensions_dir():
    """Separate torch_extensions cache per GPU architecture and torch version.

    ``load_inline`` keys its cache on the source only, so a kernel built for
    sm_90 would be reused (and fail) on sm_75 without this.
    """
    if "TORCH_EXTENSIONS_DIR" in os.environ or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability(0)
    tv = torch.__version__.split("+")[0]
    cv = (torch.version.cuda or "nocuda").replace(".", "")
    path = CACHE / "torch_extensions" / f"sm{major}{minor}-torch{tv}-cu{cv}"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(path)


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def environment_info(device: str) -> dict:
    info = dict(host=socket.gethostname(), python=platform.python_version(),
                torch=torch.__version__, cuda=torch.version.cuda,
                torch_threads=torch.get_num_threads(),
                slurm_job=os.environ.get("SLURM_JOB_ID"),
                container=os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"),
                laplex_git=git_rev(PROJECT),
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0),
                    capability=".".join(map(str, torch.cuda.get_device_capability(0))),
                    gpu_total_mib=total / MiB, gpu_free_mib_at_start=free / MiB)
        try:
            info["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,memory.used,utilization.gpu",
                 "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
            info["other_gpu_processes"] = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            pass
    else:
        info["cpu"] = platform.processor() or platform.machine()
    return info


def warn_if_gpu_busy(info: dict):
    """Timings and the OOM point are only meaningful on an otherwise idle GPU."""
    if info.get("gpu_free_mib_at_start", 1e18) < 0.95 * info.get("gpu_total_mib", 0):
        print(f"WARNING: only {info['gpu_free_mib_at_start']:.0f} of {info['gpu_total_mib']:.0f} MiB "
              f"free on the GPU at start; other processes: {info.get('other_gpu_processes')!r}. "
              "OOM points and timings will not be representative.", flush=True)


# ---------------------------------------------------------------------------
# Low-level measurement
# ---------------------------------------------------------------------------
def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def clear(device: str):
    gc.collect()
    if device == "cuda":
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


_OOM_MARKERS = ("out of memory", "alloc_failed", "can't allocate memory",
                "cannot allocate memory", "not enough memory")
_FATAL_MARKERS = ("illegal memory access", "unspecified launch failure",
                  "device-side assert", "misaligned address")


def classify_error(e: BaseException) -> str:
    if isinstance(e, torch.OutOfMemoryError):
        return OOM
    msg = str(e).lower()
    if any(m in msg for m in _OOM_MARKERS):
        return OOM
    return ERROR


def is_fatal(e: BaseException) -> bool:
    """CUDA errors after which the context is unusable."""
    return any(m in str(e).lower() for m in _FATAL_MARKERS)


class TimeLimitExceeded(Exception):
    pass


def time_ms(fn, proto: Protocol, device: str, grad: bool) -> float:
    """Mean wall-clock time of one call in ms over ``proto.iters`` calls."""
    ctx = torch.enable_grad if grad else torch.no_grad
    with ctx():
        # The probe is the first warm-up call; it also enforces the dev time limit.
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        probe = (time.perf_counter() - t0) * 1e3
        if proto.time_limit_ms is not None and probe > proto.time_limit_ms:
            raise TimeLimitExceeded(f"probe call took {probe:.0f} ms > {proto.time_limit_ms:.0f} ms")
        for _ in range(max(proto.warmup - 1, 0)):
            fn()
        sync(device)
        t0 = time.perf_counter()
        for _ in range(proto.iters):
            fn()
        sync(device)
        return (time.perf_counter() - t0) / proto.iters * 1e3


def peak_mem_mib(fn, device: str, grad: bool, input_bytes: int) -> tuple[float, float, float]:
    """Peak memory of one call in MiB: (inputs + extra peak of the call, raw peak, baseline).

    The reported value counts the call's inputs plus everything it allocates
    on top of what was already allocated before the call.  Allocations that
    merely persist from earlier calls (cuBLAS workspaces, other series) are
    excluded; in the original scripts they leaked into the raw peak, e.g. the
    constant ~64 MB floor of LAPLEX in Fig. 15.
    """
    if device != "cuda":
        return float("nan"), float("nan"), float("nan")
    clear(device)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ctx = torch.enable_grad if grad else torch.no_grad
    with ctx():
        out = fn()
    sync(device)
    del out
    peak = torch.cuda.max_memory_allocated()
    return (peak - base + input_bytes) / MiB, peak / MiB, base / MiB


def tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


# ---------------------------------------------------------------------------
# Sweep bookkeeping and the stopping rule
# ---------------------------------------------------------------------------
@dataclass
class Series:
    """One curve: ``method`` in ``mode`` measured as ``metric``."""
    method: str
    mode: str            # fwd | fb | value | ...
    metric: str          # time_ms | mem_mib | rel_l2 | ...

    @property
    def key(self):
        return (self.method, self.mode, self.metric)


@dataclass
class Sweep:
    figure: str
    tag: str
    device: str
    proto: Protocol
    axis: str                       # name of the swept variable (n, k, B, D, ...)
    extra: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    stopped: dict = field(default_factory=dict)   # group -> (status, x, message)

    def __post_init__(self):
        self.out_dir = RESULTS / self.tag / self.figure
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()

    # A *group* is the unit that is stopped together, normally (method, mode):
    # time and memory of the same call share one fate.
    def alive(self, group) -> bool:
        return group not in self.stopped

    def record(self, series: Series, x, repeat, value, status=OK, message="", **extra):
        self.rows.append(dict(figure=self.figure, device=self.tag, method=series.method,
                              mode=series.mode, metric=series.metric, axis=self.axis, x=x,
                              repeat=repeat, value=value, status=status, message=message,
                              **{**self.extra, **extra}))

    def stop(self, group, series_list, x, status, message):
        if group in self.stopped:
            return
        self.stopped[group] = (status, x, message)
        # Values of earlier repeats at this x are discarded: a point is shown only
        # if every repeat fits.
        keys = {s.key for s in series_list}
        self.rows = [r for r in self.rows
                     if not ((r["method"], r["mode"], r["metric"]) in keys and r["x"] == x)]
        for s in series_list:
            self.record(s, x, -1, float("nan"), status, message)
        print(f"    [{'/'.join(map(str, group))}] stopped at {self.axis}={x}: {status} ({message})",
              flush=True)

    def measure(self, group, x, repeat, body):
        """Run ``body()`` (which records its own values) under the stopping rule.

        ``body`` returns the list of Series it covers, so that a failure can be
        attributed to all of them.  Returns True if the group is still alive.
        """
        series_list = getattr(body, "series", [])
        if not self.alive(group):
            return False
        try:
            body()
            return True
        except TimeLimitExceeded as e:
            self.stop(group, series_list, x, TIMEOUT, str(e))
        except (RuntimeError, MemoryError) as e:
            status = OOM if isinstance(e, MemoryError) else classify_error(e)
            msg = str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__
            if status == ERROR:
                traceback.print_exc()
            self.stop(group, series_list, x, status, msg)
            if is_fatal(e):
                self.save()
                raise SystemExit(f"fatal CUDA error at {self.axis}={x}; partial results saved")
        clear(self.device)
        return False

    def mark_all_oom(self, groups_with_series, x, message, status=OOM):
        """Inputs shared by all series did not fit: every live series stops here."""
        for group, series_list in groups_with_series:
            if self.alive(group):
                self.stop(group, series_list, x, status, message)

    def any_alive(self, groups) -> bool:
        return any(self.alive(g) for g in groups)

    def save(self, meta: dict | None = None):
        import pandas as pd
        pd.DataFrame(self.rows).to_csv(self.out_dir / "raw.csv", index=False)
        m = dict(figure=self.figure, tag=self.tag, protocol=asdict(self.proto), axis=self.axis,
                 stopped={"/".join(map(str, g)): dict(status=s, x=x, message=msg)
                          for g, (s, x, msg) in self.stopped.items()},
                 elapsed_s=round(time.time() - self.t_start, 1), **self.extra)
        if meta:
            m.update(meta)
        path = self.out_dir / "meta.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        old.update(m)
        path.write_text(json.dumps(old, indent=2, default=str))


def timed_body(sweep: Sweep, series_t: Series, series_m: Series | None, make_fn, x, repeat,
               grad: bool, input_bytes: int = 0):
    """Body measuring time (and optionally peak memory) of ``make_fn()``.

    ``make_fn`` is called inside the body so that allocations it makes (for
    example an FFT plan or precomputed spectrum) fall under the stopping rule.
    """
    def body():
        fn = make_fn()
        nbytes = getattr(fn, "input_bytes", input_bytes)   # factories that allocate their own inputs
        t = time_ms(fn, sweep.proto, sweep.device, grad)
        sweep.record(series_t, x, repeat, t)
        if series_m is not None:
            mem, peak, base = peak_mem_mib(fn, sweep.device, grad, nbytes)
            sweep.record(series_m, x, repeat, mem, raw_peak_mib=peak, base_mib=base)
    body.series = [series_t] + ([series_m] if series_m is not None else [])
    return body


def try_inputs(sweep: Sweep, make, x, groups: dict, status=OOM, what="inputs do not fit"):
    """Allocate the inputs shared by all series at size ``x``.

    If they do not fit, every live series is stopped with ``status`` (OOM) at ``x``.
    """
    try:
        return make()
    except (RuntimeError, MemoryError) as e:
        if isinstance(e, MemoryError) or classify_error(e) == OOM:
            sweep.mark_all_oom(groups.items(), x, f"{what}: " + str(e).splitlines()[0][:200], status)
            clear(sweep.device)
            return None
        raise


@dataclass
class Op:
    """A benchmarked operator.  ``fwd``/``fb`` map the inputs dict to a zero-arg call.

    ``uses`` names the input tensors the operator reads (counted in its memory);
    None means all of them.
    """
    name: str
    fwd: object = None
    fb: object = None
    mem: bool = True
    uses: tuple | None = None
    predict_bytes: object = None   # CPU only: x -> bytes the call will need (see cpu_guard)
    limit: object = None           # x -> message if x is outside the implementation's range


def run_timing_sweep(sweep: Sweep, xs, make_inputs, ops, modes=("fwd", "fb"), seed=0, inputs_bytes=None):
    """Sweep ``xs`` measuring time (+ peak memory) of every op/mode until it stops.

    ``inputs_bytes`` (CPU only): x -> peak bytes of ``make_inputs(x)``, predicted as in cpu_guard
    instead of tried, since the cgroup would kill the process.
    """
    groups = {}
    for op in ops:
        for mode in modes:
            if getattr(op, mode) is None:
                continue
            ser = [Series(op.name, mode, "time_ms")]
            if op.mem and sweep.device == "cuda":
                ser.append(Series(op.name, mode, "mem_mib"))
            groups[(op.name, mode)] = ser
    for x in xs:
        if not sweep.any_alive(groups):
            break
        if sweep.device == "cpu" and inputs_bytes is not None:
            need, avail = inputs_bytes(x), cpu_available_bytes()
            if need > 0.9 * avail:
                sweep.mark_all_oom(groups.items(), x, f"inputs predicted: need ~{need / 2**30:.1f} GiB, "
                                                      f"{avail / 2**30:.1f} GiB available")
                break
        t0 = time.time()
        for r in range(sweep.proto.repeats):
            inputs = try_inputs(sweep, lambda: make_inputs(x, seed + r), x, groups)
            if inputs is None:
                break
            for op in ops:
                for mode in modes:
                    g = (op.name, mode)
                    if g not in groups or not sweep.alive(g):
                        continue
                    ser = groups[g]
                    msg = op.limit(x) if op.limit else None
                    if msg:
                        sweep.stop(g, ser, x, LIMIT, msg)
                        continue
                    if not cpu_guard(sweep, op, g, ser, x):
                        continue
                    factory = getattr(op, mode)
                    sweep.measure(g, x, r, timed_body(
                        sweep, ser[0], ser[1] if len(ser) > 1 else None,
                        lambda: factory(inputs), x, r, grad=(mode == "fb"),
                        input_bytes=tensor_bytes(
                            [inputs[k] for k in op.uses] if op.uses else inputs)))
            del inputs
            clear(sweep.device)
        print_point(sweep, x, time.time() - t0)
        sweep.save()
    return groups


def _rss_bytes() -> int:
    try:
        return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def cpu_available_bytes() -> int:
    """Memory this process may still allocate: the smallest of the cgroup limits it can see (root
    and its own cgroup, v1 or v2), the job's --mem (SLURM_MEM_PER_NODE) minus what this process
    holds, and MemAvailable.  --mem matters on nodes where the container shows neither cgroup: the
    node's free memory is then far above the job's limit and the cgroup kills the process."""
    paths = [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")]
    try:
        for line in open("/proc/self/cgroup"):
            _, ctrl, path = line.strip().split(":", 2)
            if ctrl == "":
                paths.append((f"/sys/fs/cgroup{path}/memory.max", f"/sys/fs/cgroup{path}/memory.current"))
            elif "memory" in ctrl.split(","):
                paths.append((f"/sys/fs/cgroup/memory{path}/memory.limit_in_bytes",
                              f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes"))
    except (OSError, ValueError):
        pass
    avail = []
    for lim, cur in paths:
        try:
            v = open(lim).read().strip()
            if v != "max" and int(v) < 1 << 60:
                avail.append(int(v) - int(open(cur).read()))
        except (OSError, ValueError):
            pass
    mem = os.environ.get("SLURM_MEM_PER_NODE", "")
    if mem.isdigit():
        avail.append(int(mem) * 2 ** 20 - _rss_bytes())
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            avail.append(int(line.split()[1]) * 1024)
    return min(avail) if avail else 1 << 62


def cpu_guard(sweep: Sweep, op: Op, group, series_list, x) -> bool:
    """On CPU an allocation beyond the cgroup limit kills the process instead of
    raising, so the OOM point is *predicted* there: a series whose estimated
    footprint exceeds 90% of the available memory is stopped with OOM."""
    if sweep.device != "cpu" or op.predict_bytes is None:
        return True
    need, avail = op.predict_bytes(x), cpu_available_bytes()
    if need > 0.9 * avail:
        sweep.stop(group, series_list, x, OOM,
                   f"predicted: needs ~{need / 2**30:.1f} GiB, {avail / 2**30:.1f} GiB available")
        return False
    return True


def set_threads(n: int | None):
    if n:
        torch.set_num_threads(n)
    print(f"torch threads: {torch.get_num_threads()}", flush=True)


def run_accuracy_sweep(sweep: Sweep, xs, make_problem, methods: dict, seed_of=lambda x, r: r):
    """Relative l2 / l_inf error of each method against ``problem['Y_ref']`` (fp64).

    ``make_problem(x, seed)`` builds the inputs and the reference; if that does
    not fit (in practice the fp64 reference, the inputs are O(Bn)), no error can
    be computed and every series stops with REF_OOM.  A method that does not fit
    stops with OOM.  ``methods`` maps a name to ``f(problem) -> Y``.
    """
    from impls import rel_errors
    groups = {(m, "value"): [Series(m, "value", "rel_l2"), Series(m, "value", "rel_linf")]
              for m in methods}
    for x in xs:
        if not sweep.any_alive(groups):
            break
        t0 = time.time()
        for r in range(sweep.proto.repeats):
            prob = try_inputs(sweep, lambda: make_problem(x, seed_of(x, r)), x, groups,
                              status=REF_OOM, what="reference does not fit")
            if prob is None:
                break
            for m, f in methods.items():
                def body(m=m, f=f):
                    with torch.no_grad():
                        Y = f(prob)
                    l2, linf = rel_errors(Y, prob["Y_ref"])
                    sweep.record(groups[(m, "value")][0], x, r, l2)
                    sweep.record(groups[(m, "value")][1], x, r, linf)
                body.series = groups[(m, "value")]
                sweep.measure((m, "value"), x, r, body)
            del prob
            clear(sweep.device)
        print_point(sweep, x, time.time() - t0)
        sweep.save()
    return groups


def print_point(sweep: Sweep, x, elapsed):
    parts = []
    for r in sweep.rows:
        if r["x"] == x and r["repeat"] == 0 and r["status"] == OK:
            unit = {"time_ms": "ms", "mem_mib": "MiB"}.get(r["metric"], "")
            parts.append(f"{r['method']}/{r['mode']}/{r['metric'].split('_')[0]}="
                         f"{r['value']:.4g}{unit}")
    lx = int(round(np.log2(x))) if x > 0 else x
    print(f"  {sweep.axis}=2^{lx}: " + "  ".join(parts) + f"   [{elapsed:.1f}s]", flush=True)


def fb_step(fn, *tensors):
    """One forward+backward call: fresh leaves, ``L = sum(fn(...))``, backward."""
    def step():
        leaves = [t.detach().requires_grad_(True) for t in tensors]
        y = fn(*leaves)
        y.sum().backward()
        return y.detach()
    return step


def log_range(args):
    return list(range(args.min_log, args.max_log + 1))


def resolve_tag(args) -> str:
    if args.tag:
        return args.tag
    tag = device_tag(args.device)
    return tag + ("-quick" if args.quick else "")


def init(args):
    """Common start of every script; returns (protocol, tag, environment info)."""
    if args.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("CUDA requested but not available")
        setup_extensions_dir()
        try:   # in case torch was imported before common.py set PYTORCH_CUDA_ALLOC_CONF
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: could not enable expandable segments: {e}", flush=True)
    proto = protocol_from_args(args)
    tag = resolve_tag(args)
    info = environment_info(args.device) if args.stage != "plot" else {}
    if info:
        print(json.dumps(info, indent=1, default=str), flush=True)
        warn_if_gpu_busy(info)
    print(f"protocol: {asdict(proto)}  tag: {tag}", flush=True)
    return proto, tag, info


def load_results(figure: str, tags):
    import pandas as pd
    frames = []
    for t in tags:
        f = RESULTS / t / figure / "raw.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
        else:
            print(f"  (no results for tag {t}: {f})")
    if not frames:
        raise SystemExit(f"no results to plot for {figure}")
    return pd.concat(frames, ignore_index=True)


def plot_tags(args):
    if args.tags:
        return args.tags
    if args.tag:
        return [args.tag]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("plot stage without a GPU: pass --tags (see results/)")
    return [resolve_tag(args)]


def main(parser, run, plot):
    args = parser.parse_args()
    if getattr(args, "paper", False):
        import plotting
        plotting.PAPER = True
    if args.stage in ("run", "all"):
        run(args)
    if args.stage in ("plot", "all"):
        plot(args)


def figure_dir(figure: str, tags) -> Path:
    """Plots of a single tag go next to its data; overlays go to results/combined."""
    d = RESULTS / tags[0] / figure if len(tags) == 1 else RESULTS / "combined" / figure
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_summary(figure: str, tag: str, summary: dict):
    path = RESULTS / tag / figure / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
