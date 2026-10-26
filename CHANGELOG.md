# Changelog

## [Unreleased]

### Fixed — reproducibility on machines other than the original one

- `tests/phase1_t11_rf_vs_langevin.py` no longer hardcodes `/root/EBM-splats`
  for `sys.path` and its results file; both are now derived from the script
  location. Previously the benchmark crashed with `ImportError` (geometry
  module) and `PermissionError` (results path) for any user other than the
  original machine.
- Hugging Face Hub dataset names updated to their current namespaced forms
  (`wikitext` → `Salesforce/wikitext`, `glue` → `nyu-mll/glue`,
  `snli` → `stanfordnlp/snli`) in the Phase 1 tests, `pglf/trainer.py`,
  `src/ebm/data_loader.py` and `experiments/pglf_quick_test.py`. Bare names
  are rejected by current `huggingface_hub`
  ("Repository id must be 'namespace/name'").
- Phase 2 tests (`phase2_energy_guided.py`, `phase2_composition.py`) no longer
  require a local `D:/datasets/ebm/tinystories_train.txt`: they download the
  public TinyStories validation split from the Hub automatically (cached in
  `tests/.tinystories_cache/`, ~19 MB) or accept `EBM_TINYSTORIES_PATH`.
  `src/ebm/data_loader.py` dataset roots are now overridable via
  `EBM_DATASETS_DIR` (the old `D:/` default remains only as the last
  fallback).
- CI now runs the Phase 1 T1.1 benchmark end-to-end on every push/PR
  (fresh install, CPU-only torch, public data) and asserts the RF verdict, so
  the reproducibility guarantee above is enforced, not just documented.
- `docs/PHASE1_RESULTS.md`: documented run-to-run variance for the T1.1
  speedup (committed run: 29.4x; 2026-10-26 rerun on the same RTX 3090:
  15.4x, same qualitative verdict; expect ~15-30x).

## [v2.1.0] - 2026-10-25

### Fixed
- **`SplatStorage.find_neighbors` semantic bug**: the torch-stock path selected
  the k splats with LARGEST squared distance (the FARTHEST splats) via
  `torch.topk` default `largest=True`, contradicting its own docstring
  ("nearest neighbors") and the legacy faiss path (`IndexFlatIP`, highest
  inner product = nearest on unit vectors). Both paths now return the k
  NEAREST splats in ascending-distance order.
- GPU kernel `find_neighbors_triton` (mm_topk backend) and the fused Triton
  kernel updated to the same nearest semantics (packed-key sign inversion;
  padding columns now forced to +inf so they can never enter the top-k —
  previously they relied on distance ordering staying below the cut).
- KNN kernel tests now assert the top-k SET exactly (stable under nearest
  selection: cut gap ~3e-4) and the internal ORDER within a 1e-5 near-tie
  band, because d2 gaps inside the top-k (~1e-6) are below fp32 GEMM noise
  and the order therefore depends on which GEMM computed d2.

### Changed
- Splat-energy / score / Langevin values computed through `find_neighbors`
  change accordingly; numbers published before 2.1.0 were measured with the
  old (farthest-splat) behaviour and remain archived as-is.

## [v2.0.0] - 2026-07-04

### Added
- GitHub Actions CI workflow (tests: `pytest tests/test_core.py` on Python 3.10-3.12)
- PyPI Trusted Publisher auto-release workflow
- Full test suite

### Fixed
- Invalid build-backend (setuptools.backends._legacy:_Backend)
