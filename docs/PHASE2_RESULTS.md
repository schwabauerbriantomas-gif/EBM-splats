# Phase 2: Energy-Guided Generation con Rectified Flow

**Fecha:** 2 Julio 2026
**Hardware:** RTX 3090 24GB, PyTorch 2.6, CUDA 12.4

---

## Hipótesis

Manipular la energía de splats específicas permite controlar qué tipo de contenido se genera. Boost energy hacia tópico A + suppress energy de tópico B debería producir samples que retrieval textos de A.

## Setup

1. **Data:** 10K TinyStories embebidos con MiniLM-L6-v2 (384D, normalizados S^383)
2. **Clusters:** 50 tópicos via KMeans, seleccionados 2 bien separados (sim=0.27)
   - Topic A (cluster 25): historias sobre parques, pájaros, niños jugando (126 stories)
   - Topic B (cluster 27): historias con moralejas explícitas (57 stories)
3. **Splats:** 50 centros de cluster como splat centers
4. **RF:** VelocityNet (3-layer MLP, 512 hidden), entrenado 1000 steps (5.9s)
5. **Sampling:** 500 muestras por condición, 2 pasos RF, decode via nearest-neighbor retrieval
6. **Guidance:** boost/suppress direction toward/away from cluster centers en espacio tangente

## Resultados

| Guidance Scale | Boost A (% en A) | Boost B (% en B) | Baseline (% en A) |
|---|---|---|---|
| 0.0 (baseline) | — | — | 0.6% |
| 0.5 | 62.8% | 99.2% | — |
| **1.0** | **99.6%** | **100.0%** | — |
| **2.0** | **100.0%** | **100.0%** | — |
| 5.0 | 4.4% | 8.2% | — |

## Análisis

### Funciona
- **gs=1.0 a 2.0: control perfecto** (99.6-100% de samples en el tópico objetivo)
- Desde baseline 0.6% → 100% con un solo boost direction
- El control funciona en ambas direcciones (boost A y boost B)
- RF con 2 pasos + guidance = sampling instantáneo

### Limitaciones
- **gs=5.0 colapsa**: demasiado guidance empuja los samples fuera del manifold de datos
- **Decode es retrieval**: no genera texto nuevo, recupera textos existentes del corpus
- **2 tópicos testeados**: falta validar con más pares y tópicos más sutiles
- **Corpus pequeño**: 10K stories limita la diversidad del retrieval

### Próximos pasos necesarios
1. **Decoder neural**: entrenar un decoder que mapee samples → texto (no retrieval)
2. **Composición**: boost A + boost B simultáneo → ¿genera contenido en la intersección?
3. **Control continuo**: guidance_scale como variable continua → interpolación entre tópicos
4. **Más tópicos**: validar con 10+ clusters, no solo 2
5. **Corpus más grande**: 100K+ stories para mejor cobertura

## Veredicto

**La generación con control fino vía energía EBM + RF funciona.** Es el primer resultado positivo del proyecto. La combinación de energy landscape + Rectified Flow permite controlar qué región del espacio semántico se samplea, en tiempo real, con precisión del 100%.

El siguiente paso crítico es reemplazar el decode por retrieval con un decoder neural real, para que el sistema genere texto nuevo en vez de recuperar existente.
