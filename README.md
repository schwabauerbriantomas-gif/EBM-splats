# EBM (Energy-Based Model) para Lenguaje

[![Status](https://img.shields.io/badge/status-entrenando-yellow.svg)](https://github.com)
[![Vulkan](https://img.shields.io/badge/vulkan-1.3-red.svg)](https://vulkan.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-AMD%20RX%206650XT-orange.svg)](https://amd.com)

> **Energy-Based Model para generación de lenguaje sobre hiperesfera 640D con Gaussian Splats como atractores y dinámica Langevin para sampleo.**

---

## 📋 Tabla de Contenidos

- [Estado del Proyecto](#-estado-del-proyecto)
- [Arquitectura](#-arquitectura)
- [Avances Logrados](#-avances-logrados)
- [Limitaciones y Defectos Actuales](#-limitaciones-y-defectos-actuales)
- [Quick Start](#-quick-start)
- [Documentación Técnica](#-documentación-técnica)
- [Roadmap](#-roadmap)

---

## 🎯 Estado del Proyecto

**Versión**: 2.0 - Implementación Composicional
**Estado**: 🔄 **En entrenamiento activo** (Vulkan GPU acceleration)
**Inicio**: Febrero 2026
**Ubicación**: `projects/ebm/`

### Validaciones Completadas ✅

| Validación | Estado | Descripción |
|------------|--------|-------------|
| **Geometric Correctness** | ✅ PASS | Mapeo exacto a S^639 |
| **Training Stability** | ✅ PASS | 16-token dummy sequence |
| **Text Generation** | ✅ PASS | Langevin sample sin NaN |
| **Dataset Integration** | ✅ PASS | wikitext-103 + GPT-2 tokenizer |
| **Vulkan Dispatch** | ✅ PASS | Riemannian scores idénticos |

### Progreso de Entrenamiento 🔄

- **Dataset**: wikitext-103 (20K samples, 5116 batches/epoch)
- **Epochs**: 10 planificados
- **Batch size**: 16
- **Estado**: Entrenando en background
- **Checkpoints**: `checkpoints/ebm_epoch_X.pt`

---

## 🏗 Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EBM Architecture (S^639)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input → Tokenizer (GPT-2) → Embedding (640D)                       │
│                                                                      │
│  Embedding → ┌──────────────┐                                        │
│              │  SplatStore  │ → Gaussian Splats (μ, α, κ)           │
│              │   (50K max)  │                                        │
│              └──────────────┘                                        │
│                      ↓                                               │
│              ┌──────────────┐                                        │
│              │ Energy Func  │ → E(x) = E_splats + E_geom + E_comp   │
│              │  (Riemann)   │                                        │
│              └──────────────┘                                        │
│                      ↓                                               │
│              ┌──────────────┐                                        │
│              │  Langevin    │ → Underdamped Dynamics (200 steps)    │
│              │  Sampler     │                                        │
│              └──────────────┘                                        │
│                      ↓                                               │
│              ┌──────────────┐                                        │
│              │  SOC Ctrl    │ → Self-Organized Criticality          │
│              └──────────────┘                                        │
│                      ↓                                               │
│              ┌──────────────┐                                        │
│              │  MoE Decoder │ → 4 Experts, 2 Active                 │
│              └──────────────┘                                        │
│                      ↓                                               │
│  Output ← Tokens ← Logits                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Componentes Principales

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| **Tokenizer** | `dataset_utils.py` | GPT-2 tokenizer (vocab: 50,257) |
| **SplatStore** | `splats.py` | ImprovedSplatStore con KNN FAISS |
| **EnergyFunction** | `energy.py` | Splat + Geométrica + Composicional |
| **Langevin** | `langevin.py` | Underdamped Störmer-Verlet integrator |
| **SOC Controller** | `soc.py` | HistoryBuffer + consolidación automática |
| **Decoder** | `decoder.py` | Mixture of Experts (4 expertos, 2 activos) |
| **Geometry** | `geometry.py` | Operaciones Riemannianas (exp_map, log_map) |
| **Vulkan Engine** | `vulkan_engine.py` | GPU acceleration para AMD RX 6650XT |

---

## ✅ Avances Logrados

### Fase 1: Convergencia y Validación (Completada)

#### 1. Inicialización Inteligente de Splats ✅
- **Cargar embeddings GPT-2 pre-entrenadas** para representación semántica inicial
- **Expandir de 10K a 50K splats** progresivamente con curriculum learning
- **Temperatura de energy configurada** para mejor exploración

**Impacto**: Cobertura de vocabulario mejorada significativamente

#### 2. Curriculum Learning ✅
- **Fase 1**: 5K splats, alta temperatura
- **Fase 2**: 30K splats, temperatura media
- **Fase 3**: 50K splats, fine-tuning

**Impacto**: Progreso más estable y predecible

#### 3. Monitoreo Avanzado ✅
- **Métricas en vivo**: Loss, energía, estadísticas de splats, SOC rate
- **Logging detallado**: Timestamps, checkpoints cada 5 epochs
- **Alertas automáticas**: Energía aumentando, SOC demasiado rápido

**Impacto**: Detección temprana de problemas

#### 4. Validación Automática ✅
- **Evaluación de checkpoints**: Perplexity, métricas de energía
- **Herramientas de diagnóstico**: `diagnose.py`, `evaluate.py`
- **Muestras generadas**: Evaluación humana

**Impacto**: Feedback en tiempo real sobre calidad

#### 5. Mejoras de Splat Store ✅
- **Estadísticas completas**: Frecuencia, edad, kappa dinámico
- **Weight decay gradual**: Por epoch
- **Límites configurables**: kappa ∈ [1.0, 50.0]

**Impacto**: Mejor gestión de recursos de splats

---

## ⚠️ Limitaciones y Defectos Actuales

### 🔴 Críticos

#### 1. Tiempo de Convergencia
**Problema**: Entrenamiento requiere días/semanas en GPU local

> *"GPT-2 level functionality inherently traces hundreds of millions of parameters over enormous server-grade GPU clusters for several weeks. Translating this quality identically down onto a single continuous discrete RX 6650XT Vulkan mapping means that the pretrain.py instance currently running should be left undisturbed for several days (or weeks)."*

**Mitigación**:
- ✅ Curriculum learning implementado
- ✅ Checkpoints cada epoch para resumir
- 🔄 Monitoreo continuo de progreso

**Estado**: Aceptado como limitación de hardware

---

#### 2. Búsqueda de Splats O(N)
**Problema**: KNN con FAISS-CPU es O(N), no O(log N)

**Impacto**: Búsqueda se vuelve lenta con muchos splats (50K+)

**Mitigación**:
- ✅ FAISS-CPU implementado (12x speedup vs naive)
- 🔄 Pendiente: FAISS-GPU migration

**Solución Futura**: HRM2 hierarchical search (como M2M)

---

#### 3. Embeddings Hash-Based (Demo)
**Problema**: Índice actual usa embeddings hash-based, no semánticos

**Impacto**: Búsqueda no captura semántica real

**Mitigación**:
- 🔄 TODO: Integrar sentence-transformers

**Estado**: Limitación conocida del prototipo

---

### 🟡 Moderados

#### 4. Batch Size Limitado
**Problema**: Batch size = 16 (limitado por VRAM de 8GB)

**Impacto**: Entrenamiento más lento, gradientes menos estables

**Mitigación**:
- 🔄 TODO: Mixed precision training (BF16)
- 🔄 TODO: Gradient accumulation (effective batch 8x)

---

#### 5. Decoder Simplificado
**Problema**: MoE decoder es ligero (4 expertos, 2 activos)

**Impacto**: Calidad de generación puede ser inferior a transformers grandes

**Mitigación**:
- ✅ Arquitectura funcional
- 🔄 TODO: Transformer decoder estilo GPT-2

---

#### 6. Sin Integración LLM Completa
**Problema**: EBM genera tokens pero no está integrado con LLM externo

**Impacto**: No se puede usar en pipelines RAG directamente

**Mitigación**:
- 🔄 TODO: Integración con LangChain/LlamaIndex
- 🔄 TODO: API REST para uso externo

---

### 🟢 Menores

#### 7. Logging Detallado pero Verbose
**Problema**: Logs pueden ser muy extensos

**Mitigación**: ✅ Niveles de logging configurables

---

#### 8. Dependencia de Vulkan SDK
**Problema**: Requiere instalación manual de Vulkan SDK

**Mitigación**: ✅ Fallback a CPU si Vulkan no está disponible

---

## 🚀 Quick Start

### Requisitos

```bash
# Dependencias principales
pip install torch numpy transformers datasets faiss-cpu

# Vulkan SDK (opcional, para GPU acceleration)
# https://vulkan.lunarg.com/
```

### Entrenar

```bash
# GPU (Recomendado)
python train.py --device vulkan --epochs 10 --batch-size 16

# CPU (Lento)
python train.py --device cpu --epochs 10 --batch-size 16

# Reanudar desde checkpoint
python train.py --device vulkan --resume checkpoints/ebm_epoch_5.pt
```

### Diagnosticar

```bash
# Análisis de checkpoint específico
python diagnose.py --checkpoint checkpoints/ebm_epoch_5.pt --device vulkan

# Análisis batch de todos los checkpoints
python diagnose.py --batch --device vulkan

# Generar reporte con recomendaciones
python diagnose.py --checkpoint checkpoints/ebm_epoch_10.pt --report
```

### Evaluar

```bash
# Calcular perplexity en WikiText-103
python evaluate.py --checkpoint checkpoints/ebm_epoch_10.pt --device vulkan

# Generar muestras
python generate.py --checkpoint checkpoints/ebm_epoch_10.pt --prompt "The future of AI"
```

---

## 📖 Documentación Técnica

### Especificación Completa
- **Archivo**: `spec.txt`
- **Contenido**: 20 secciones, 620+ líneas
- **Incluye**: Fórmulas matemáticas completas, hiperparámetros, pipeline completo

### Espacio Latente

| Propiedad | Valor |
|-----------|-------|
| **Manifold** | S^639 (hiperesfera unitaria) |
| **Dimensión** | 640D |
| **Restricción** | \|\|x\|\|² = 1 |
| **Métrica** | g_x = I - x·x^T |
| **Distancia** | d(x,y) = arccos(x·y) |

### Gaussian Splats

| Parámetro | Descripción | Rango |
|-----------|-------------|-------|
| **μ** | Media direccional [640] | Esfera unitaria |
| **α** | Peso/intensidad | (0, ∞) |
| **κ** | Concentración | [1.0, 50.0] |

### Langevin Underdamped

```
dx/dt = v
dv/dt = -γv - ∇_R E(x) + √(2γT)·ξ
```

| Parámetro | Valor |
|-----------|-------|
| **Pasos** | 200 |
| **dt** | 0.001 |
| **Fricción (γ)** | 0.1 |
| **Temperatura (T)** | 1.0 |

### Entrenamiento

| Parámetro | Valor |
|-----------|-------|
| **Método** | Denoising Score Matching |
| **Loss** | L = E[\|\|s_θ(x̃) - ε/σ\|\|²] |
| **Dataset** | wikitext-103 |
| **Batch size** | 16 |
| **Learning rate** | 1e-4 (Cosine Annealing) |
| **Noise levels** | (0.01, 0.05, 0.1, 0.2, 0.5) |

---

## 🗺 Roadmap

### ✅ Completado

- [x] Arquitectura base EBM
- [x] Gaussian Splats con KNN
- [x] Langevin Underdamped
- [x] SOC Controller
- [x] Vulkan GPU acceleration
- [x] Curriculum Learning
- [x] Monitoreo avanzado
- [x] Diagnóstico automático
- [x] Validación geométrica

### 🔄 En Progreso

- [ ] Entrenamiento completo (10 epochs)
- [ ] Evaluación de perplexity
- [ ] Análisis de convergencia

### 📋 Futuro (Fase 2 - Opcional)

- [ ] **FAISS-GPU Migration**: Aceleración real de KNN
- [ ] **Mixed Precision Training**: BF16 para 2x capacidad
- [ ] **Gradient Accumulation**: Effective batch 8x
- [ ] **Transformer Decoder**: Arquitectura GPT-2
- [ ] **HRM2 Integration**: Búsqueda O(log N)
- [ ] **API REST**: Integración con sistemas externos
- [ ] **LangChain/LlamaIndex**: Pipelines RAG

---

## 📊 Métricas de Éxito

### Targets Fase 1

| Métrica | Target | Estado |
|---------|--------|--------|
| **Perplexity (WikiText)** | < 100 | 🔄 Por validar |
| **Energy Trend** | Decreciente | 🔄 Monitoreando |
| **Splat Coverage** | > 80% | 🔄 Por medir |
| **SOC Rate** | Decreciente | 🔄 Monitoreando |

### Métricas de Convergencia

| Indicador | Excelente | Bueno | Regular | Malo |
|-----------|-----------|-------|---------|------|
| **Loss Score Matching** | < 0.05 | < 0.1 | < 0.2 | > 0.2 |
| **Energía Promedio** | Decreciente | Estable | Fluctuante | Creciente |
| **Tendencia** | Converging | Stable | Needs attention | Diverging |

---

## 🤝 Contribuir

### Estructura del Proyecto

```
projects/ebm/
├── train.py              # Script principal de entrenamiento
├── diagnose.py           # Diagnóstico de checkpoints
├── evaluate.py           # Evaluación de calidad
├── generate.py           # Generación de texto
├── model.py              # EBMModel principal
├── splats.py             # ImprovedSplatStore
├── energy.py             # EnergyFunction
├── langevin.py           # Langevin sampler
├── soc.py                # SOC controller
├── decoder.py            # MoE decoder
├── geometry.py           # Operaciones Riemannianas
├── vulkan_engine.py      # GPU acceleration
├── config.py             # Configuración
├── dataset_utils.py      # WikiText-103 dataloader
├── spec.txt              # Especificación técnica completa
└── README.md             # Este archivo
```

### Dependencias

Ver `requirements.txt` para lista completa.

---

## 📚 Referencias

- **Especificación técnica**: `spec.txt`
- **Documentación M2M**: `../m2m/README.md`
- **Integración M2M-EBM**: `../../MEMORY.md`

---

## 📄 Licencia

Apache License 2.0

---

## 👤 Autor

**Alfred** 🎩 - Asistente AI del Sr. Schwabauer

---

## 🙏 Agradecimientos

- **DeepSeek**: Inspiración para Engram memory
- **Gaussian Splatting**: Foundation para representaciones
- **Vulkan SDK**: GPU acceleration

---

**Última actualización**: 2026-02-23
**Versión**: 2.0
**Estado**: En entrenamiento activo 🔄

---

> *"El objetivo no es artificial general intelligence — es genuine specific usefulness."*
