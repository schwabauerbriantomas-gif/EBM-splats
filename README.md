# EBM-Splats

> **Status: ACTIVE RESEARCH** — Phase 1 empirical tests complete (July 2026). PGLF embedding approach descartado. EBM generador con Rectified Flow en exploración.

Energy-Based Model con Gaussian Splats en hiperesfera 640D. Explora representaciones distribucionales para espacios latentes, sampling por Langevin dynamics y Rectified Flow.

## Fases del proyecto

### Phase 1: EBM + PGLF (Abril 2026) — Descartado

EBM con splats gaussianos como atractores en S^639 + PGLF (proyección sobre MiniLM con loss contrastivo).

**Resultado:** PGLF degradó MiniLM en STS-B (-4.7%). La proyección sobre embeddings pre-entrenados siempre destruye la geometría.

### Phase 1 Empírica: Tests de descarte (Julio 2026)

3 tests empíricos para descartar o confirmar alternativas. RTX 3090, datos reales.

| Test | Hipótesis | Resultado | Veredicto |
|------|-----------|-----------|-----------|
| PGLF Grid (14 configs) | ¿Alguna config supera MiniLM? | 0/14 superan baseline (0.8672) | **DESCARTADO** |
| OOD Detection | ¿Energía EBM detecta OOD? | AUROC=1.0 pero NN=0.999 | **SIN VENTAJA** |
| RF vs Langevin | ¿RF soluciona bottleneck de velocidad? | 24-29x más rápido, mejor calidad | **CONFIRMADO** |

**Hallazgo clave:** El argumento de "200 pasos Langevin por token" ya no aplica. Rectified Flow con 1-2 pasos produce mejores samples que Langevin con 200 pasos, 24x más rápido.

### Phase 2: EBM Generador con RF — En exploración

Nuevo enfoque: EBM como generador que aprende su propio espacio latente (no como capa sobre modelos pre-entrenados), con sampling por Rectified Flow.

## Estructura del repo

```
├── config.py              # EBMConfig — configuración V2
├── energy.py              # EnergyFunction — splats + geom + comp energy
├── geometry.py            # Operaciones Riemannianas (exp_map, log_map, tangent)
├── splats.py              # SplatStorage — gaussianas direccionales
├── score_network.py       # ScoreNetwork — denoising score matching
├── langevin.py            # Langevin dynamics (underdamped)
├── decoder.py             # MoE decoder (S^639 → vocab)
├── context_hierarchy.py   # Contexto jerárquico (local/medium/global)
├── model.py               # EBMModel — integración completa
├── train_rectified_flow.py # Rectified Flow sampler (SPEC 3)
├── pglf/                  # PGLF (archivado — descartado empíricamente)
├── tests/
│   ├── phase1_t11_rf_vs_langevin.py    # Test RF vs Langevin
│   ├── phase1_t12_pglf_grid.py         # Test PGLF grid search
│   ├── phase1_t13_ood_energy.py        # Test OOD detection
│   └── t1*_results.jsonl               # Resultados raw
├── docs/
│   └── PHASE1_RESULTS.md  # Reporte completo Fase 1
└── benchmark_results/     # Benchmarks previos
```

## Resultados detallados

Ver [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md) para el reporte completo de tests empíricos.

## Tech Stack

- Python, PyTorch (CUDA 12.4, RTX 3090)
- sentence-transformers, HuggingFace datasets
- Rust (M2M integration via HTTP)

## Licencia

Apache-2.0
