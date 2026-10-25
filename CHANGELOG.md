# Changelog

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
