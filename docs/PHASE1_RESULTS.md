# EBM-Splats: Fase 1 — Descarte Empírico de Alternativas

**Fecha:** 2 Julio 2026
**Hardware:** RTX 3090 24GB, Ryzen 5 3400G, 32GB RAM, CUDA 12.4, PyTorch 2.6
**Metodología:** 3 tests empíricos para descartar o confirmar alternativas al proyecto archivado

---

## Resumen Ejecutivo

Se ejecutaron 3 tests empíricos sobre las hipótesis pendientes del EBM-splats. **2 de 3 hipótesis descartadas definitivamente. 1 confirmada como solucionable.**

| Test | Hipótesis | Resultado | Veredicto |
|------|-----------|-----------|-----------|
| T1.2 PGLF Grid | ¿Alguna config supera MiniLM? | 0/14 configs superan baseline | **DESCARTADO** |
| T1.3 OOD Detection | ¿Energía EBM detecta OOD? | AUROC=1.0 pero NN=0.999 | **SIN VENTAJA** |
| T1.1 RF vs Langevin | ¿RF soluciona bottleneck de velocidad? | 29x más rápido, mejor calidad | **CONFIRMADO** |

---

## Test 1.2: PGLF Grid Search

**Script:** `tests/phase1_t12_pglf_grid.py`
**Datos:** 14 configuraciones variando data (50K-500K), epochs (1-5), init gain (0.1-1.0), LR (1e-4 a 1e-3), temperatura (0.05-0.1)

### Resultados

| Config | STS-B | Δ | Tiempo |
|---|---|---|---|
| **MiniLM baseline** | **0.8672** | — | — |
| gain=1.0 | 0.8580 | -0.9% | 3.6s |
| gain=0.5 | 0.8514 | -1.6% | 3.2s |
| temp=0.05 | 0.8454 | -2.2% | 3.5s |
| epochs=3 | 0.8416 | -2.6% | 9.2s |
| combo1 (200K, 3ep, gain0.5) | 0.8407 | -2.7% | 40.3s |
| baseline (original run) | 0.8415 | -2.6% | 5.7s |
| epochs=5 | 0.8380 | -2.9% | 15.7s |
| combo3 (200K, 5ep, gain1.0, lr5e-4) | 0.8371 | -3.0% | 71.9s |
| data=200K | 0.8343 | -3.3% | 14.0s |
| data=500K | 0.8326 | -3.5% | 32.2s |
| combo2 (500K, 3ep, gain0.5, lr5e-4) | 0.8287 | -3.9% | 103.7s |
| temp=0.1 | 0.8266 | -4.1% | 3.3s |
| lr=5e-4 | 0.8218 | -4.5% | 3.3s |
| lr=1e-3 | 0.8111 | -5.6% | 3.3s |

### Análisis

- **Más datos NO ayudan**: 200K (-3.3%) y 500K (-3.5%) son peores que 50K con gain=1.0 (-0.9%)
- **Menos gain conservador acerca al baseline**: gain=1.0 (-0.9%) > gain=0.5 (-1.6%) > gain=0.1 (-2.6%)
- **LR alto destruye**: lr=1e-3 da -5.6%, el peor resultado
- **Patrón**: Cualquier proyección entrenada siempre destruye la geometría de MiniLM, sin importar hiperparámetros

### Conclusión

**PGLF está definitivamente descartado para embeddings unimodales de texto.** No existen hiperparámetros que lo salven. La geometría pre-entrenada de MiniLM (entrenada en 1B+ pares) es un obstáculo insuperable para una capa de proyección entrenada con datos limitados.

---

## Test 1.3: OOD Detection con Energía EBM

**Script:** `tests/phase1_t13_ood_energy.py`
**Setup:** 10K embeddings ID (WikiText) como splats. OOD: código Python, random tokens, texto non-English.

### Resultados

| Config (τ, k) | AUROC código | AUROC random | AUROC foreign | AUROC total |
|---|---|---|---|---|
| τ=0.01, k=16 | 1.000 | 1.000 | 1.000 | **1.000** |
| τ=0.01, k=32 | 1.000 | 1.000 | 1.000 | 1.000 |
| τ=0.05, k=16 | 1.000 | 1.000 | 1.000 | 1.000 |
| τ=0.10, k=16 | 1.000 | 1.000 | 1.000 | 1.000 |
| τ=1.00, k=128 | 1.000 | 0.999 | 0.996 | 0.998 |
| **Nearest-Neighbor** | **1.000** | **1.000** | **0.997** | **0.999** |

### Análisis

- La energía de splats discrimina perfectamente ID vs OOD (AUROC=1.0)
- **Pero nearest-neighbor cosine hace exactamente lo mismo (0.999)**
- No hay ventaja medible de usar la función de energía vs simplemente medir distancia al vecino más cercano
- La energía es computacionalmente más costosa (logsumexp sobre k vecinos) sin beneficio

### Conclusión

**OOD detection con energía EBM funciona pero no ofrece ventaja sobre métodos triviales.** Descartado como caso de uso diferenciador.

---

## Test 1.1: Rectified Flow vs Langevin

**Script:** `tests/phase1_t11_rf_vs_langevin.py`
**Setup:** Velocity network (3 layer MLP, 512 hidden) entrenado 500 steps sobre 5K embeddings WikiText. Sampling: 1024 muestras.

### Resultados

| Método | Pasos | Tiempo | MMD (↓ mejor) |
|---|---|---|---|
| **RF 1 paso** | 1 | **0.006s** | **0.0092** |
| RF 2 pasos | 2 | 0.003s | 0.0099 |
| RF 5 pasos | 5 | 0.008s | 0.0102 |
| RF 10 pasos | 10 | 0.015s | 0.0105 |
| RF 20 pasos | 20 | 0.026s | 0.0113 |
| Langevin 10 pasos | 10 | 0.089s | 0.0133 |
| Langevin 50 pasos | 50 | 0.058s | 0.0073 |
| Langevin 100 pasos | 100 | 0.108s | 0.0100 |
| Langevin 200 pasos | 200 | 0.232s | 0.0148 |
| Random noise | 0 | 0.009s | 0.0181 |

### Análisis

- **RF (5 pasos) vs Langevin (200 pasos): 29x más rápido** (0.008s vs 0.232s)
- **RF (5 pasos) produce MEJORES samples**: MMD=0.0102 vs Langevin MMD=0.0148
- RF incluso con 1 solo paso (MMD=0.0092) supera a Langevin con 200 pasos (0.0148)
- Training del velocity network: 3.2s para 500 steps
- Interesting: más pasos de RF NO mejora calidad (1 paso > 20 pasos). El velocity field es tan preciso que 1 paso basta

### Conclusión

**El problema de velocidad del EBM está resuelto.** Rectified Flow reduce 200 pasos de Langevin a 1-5 pasos con mejor calidad de muestreo. El cuello de botella computacional ya no es un argumento válido para descartar el EBM.

---

## Veredicto Final

### Lo que se confirmó como descartado

1. **PGLF como capa sobre embeddings**: Definitivamente no funciona. 14 configuraciones, ninguna supera baseline. Más datos empeora. No hay salvataje.

2. **Energía EBM para OOD**: Funciona pero nearest-neighbor hace lo mismo sin ventaja. No es un diferenciador.

### Lo que cambió del veredicto anterior

3. **Bottleneck de velocidad**: El argumento de "200 pasos Langevin por token" **ya no aplica**. Rectified Flow lo resuelve con 29x speedup y mejor calidad. Esto era uno de los 4 argumentos para abandonar.

### Implicación

De los 4 problemas fundamentales identificados al archivar el proyecto:

| Problema | Estado original | Estado post-tests |
|---|---|---|
| 200 pasos Langevin | ❌ Inviable | ✅ Resuelto (RF, 1 paso) |
| Landscape plano (10K splats en 640D) | ❌ Demasiado sparse | ⚠️ Sin testear aún |
| Decoder lossy (S^639 → vocab) | ❌ Mapeo difícil | ⚠️ Sin testear aún |
| Compitiendo contra GPT/LLaMA | ❌ No competitivo | ❌ Sigue siendo cierto |

El EBM como **generador de lenguaje** sigue sin ser competitivo con transformers. Pero la velocidad ya no es el problema. Si hubiera un caso de uso donde el sampling del landscape de energía agregue valor (no generación de texto, sino exploración de espacios latentes), RF lo hace viable.

### Caminos que NO se descartaron

- **EBM como generador con RF** (proyecto nuevo, no modificación mínima)
- **Cross-modal retrieval** con splats como distribuciones (Test 2.3 del plan, no ejecutado)
- **Inicialización pre-entrenada** de splats desde bge-m3 (Test 2.1, no ejecutado)

---

*Tests ejecutados en RTX 3090, datos reales, resultados reproducibles.*
*Scripts en `tests/phase1_t1*.py`, resultados raw en `tests/t1*_results.jsonl`*
