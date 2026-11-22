# Vendored LAPLEX implementations

Copies of the code that `impls.py` imports besides the released `../comparison_experiments/`
(`--impl public`, `public-raw`, `nocuda` and the CPU figures), so that `benchmarks/` runs from
this repository alone. Each implementation has its own directory because several of them
ship a module of the same name (`prefix_suffix`, `prefix_suffix_dl`, `kerlap_ops`);
`impls.py` lets one process import only one of them.

| Directory | `--impl` | Origin | Changes |
|---|---|---|---|
| `fused/` | `fused` (default) | `kerlap_fused.py`, `prefix_suffix_dl.py` of `fused_orig/` | `prefix_suffix_dl.py`: scan grid `(tiles, B)` when there are more than 65535 tiles (lifts the n < 2^26 cap); marked `repro:` in the CUDA source |
| `fused_orig/` | `fused-orig` | internal `KerLap/implementation_cuda/faster/` (2026-05-03), the code of the published Figs. 1/15 and Tab. 3 | none |
| `lib2026/` | `lib2026`, `public-fix`, `lib2026-auto`; `pure` in `../last_interval_test.py` | internal `2026-LAPLEX` library (2026-05-18): `cuda_kernel.py`, `prefix_suffix.py`, `pure.py`, `LICENSE` | none; its `merged_events.py` is not copied (see below) |
| `merged_events.py` | `merged` | `2026-LAPLEX/merged_events.py` | the missing closing `"""` of `_CUDA_SRC` added (the original does not parse) |
| `kerlap/` | `dl` | internal `KerLap/implementation_cuda/` (2026-04-26): `kerlap_ops.py`, `prefix_suffix_dl.py` | none |
| `kerlap_opt/` | `dl-opt` | internal `KerLap/implementation_cuda/optimization/` (2026-07-01): `kerlap_ops.py`, `prefix_suffix_dl.py` | none |
| `mix/` | `mix-fused` | internal `LAPLEX multi/laplex_mix_fused.py` | none |

`cuda_kernel.py` tries `from merged_events import merged_K_at_x` for large batches and falls
back to its own scan on any exception. With the original, unparsable `merged_events.py` next
to it this always fell back, so `lib2026/` leaves it out to keep that behaviour.
