# EBM (Energy-Based Model) para Lenguaje

> Energy-Based Model para generación de lenguaje sobre hiperesfera 640D, con Gaussian Splats como atractores y dinámica Langevin para sampleo.

**Estado**: 🔄 En entrenamiento activo con Phase 1 Mejoras
**Ubicación**: `projects/ebm/`
**Inicio**: Febrero 2026

---

## 🎉 Nuevas Mejoras Implementadas (Fase 1)

### ✅ Fase 1: Convergencia y Validación

#### 1. Inicialización Inteligente de Splats
- **Cargar embeddings GPT-2 pre-entrenadas** para representación semántica inicial rica
- **Expandir de 10K a 50K splats** progresivamente con curriculum learning
- **Temperatura de energy configurada** para mejor exploración inicial

**Beneficios**:
- Cobertura de vocabulario mucho mejor desde el inicio
- Representaciones iniciales semanticamente significativas
- Reducción drástica del tiempo de convergencia

#### 2. Curriculum Learning
- **Fase 1 (Init)**: 5K splats, aprender representaciones básicas, alta temperatura
- **Fase 2 (Mid)**: 30K splats, expandir vocabulario, temperatura media
- **Fase 3 (Max)**: 50K splats, fine-tuning completa, baja temperatura

**Beneficios**:
- Progreso más predecible y estable
- Evita colapso local en mínimos
- Mejor uso de capacidad GPU por fase

#### 3. Monitoreo Avanzado
- **Métricas en vivo**:
  - Loss score matching (por batch y por epoch)
  - Energía promedio con tendencia
  - Estadísticas de splats (n_active, frecuencia, edad)
  - Tasa de consolidación SOC
  - Perplexity en validación WikiText-103

- **Logging detallado**:
  - Timestamps exactos por batch
  - Checkpoints cada 5 epochs (además de cada epoch)
  - Información de diagnóstico (n_splats, distancias promedio)

- **Alertas automáticas**:
  - Energía aumentando inesperadamente
  - SOC consolidándose demasiado rápido
  - Perplexity empeorando
  - Convergencia pobre detectada

#### 4. Validación Automática
- **Evaluación de checkpoints**:
  - Perplexity automática en subset de validación
  - Métricas de energía por epoch
  - Análisis de tendencia de convergencia

- **Herramientas de diagnóstico**:
  - `diagnose.py`: Análisis automático de checkpoints
  - `evaluate.py`: Métricas de calidad generativa
  - Muestras generadas para evaluación humana

#### 5. Mejoras de Splat Store
- **Estadísticas de splats mejoradas**:
  - Seguimiento de frecuencia de uso
  - Edad de cada splat para weight decay
  - Kappa dinámico con límites configurables (min: 1.0, max: 50.0)
  - Ajuste de temperatura para más exploración
  - Weight decay gradual por epoch

---

## 🏗 Arquitectura del Modelo

```
Tokenizer → Embedding → Splat Store → Energy → Langevin → Decoder → Tokens
                     (μ, α, κ)         (Riemann)    (MoE)
```

**Componentes Mejorados**:
- **ImprovedSplatStore**: Hasta 50K splats con KNN FAISS-CPU
- **EnergyFunction**: Splat + Geométrica + Composicional
- **Langevin Dynamics**: Underdamped (momentum) con 200 pasos
- **SOC Controller**: Self-Organized Criticality para consolidación
- **EBMDecoder**: Mixture of Experts (4 expertos, 2 activos)
- **Geometry**: Operaciones Riemannianas completas (exp_map, log_map, proyección de gradientes)

---

## 🚀 Cómo Entrenar

### Inicio Rápido (Vulkan GPU)

```bash
# Entrenar con mejoras de Fase 1 usando GPU
python train.py --device vulkan --epochs 10 --batch-size 32

# Reanudar desde checkpoint
python train.py --device vulkan --resume

# Validar checkpoint existente
python diagnose.py --checkpoint checkpoints/ebm_epoch_5.pt --device vulkan
```

### Diagnóstico Automático

```bash
# Análisis detallado de checkpoint específico
python diagnose.py --checkpoint checkpoints/ebm_epoch_X.pt --device vulkan

# Análisis batch de todos los checkpoints
python diagnose.py --batch --device vulkan

# Generar reporte con recomendaciones
python diagnose.py --checkpoint checkpoints/ebm_epoch_10.pt --device vulkan --report
```

---

## 📊 Métricas de Éxito

### Fase 1 Objetivos
| Métrica | Target | Progreso |
|---------|--------|----------|
| Perplexity (WikiText) | < 100 | Pendiente de validación |
| Energy Trend | Estable/Decreciente | Por medir en entrenamiento |
| Splat Coverage | 80%+ | Pendiente de medición |
| SOC Rate | Decreciente | Por medir en entrenamiento |

### Métricas de Convergencia
- **Loss Score Matching**: Target < 0.1
- **Energía Promedio**: Estable y decreciente
- **Tendencia**: Converging o Excelente (estable)
- **Tasa de Consolidación**: Decreciendo con el tiempo

---

## 📁 Archivos del Proyecto

### Core Architecture
- `config.py` - Configuración centralizada (EbmConfig dataclass)
- `model.py` - EBMModel principal
- `splats.py` - ImprovedSplatStore (50K splats con KNN)
- `energy.py` - EnergyFunction (Splat + Geométrica + Composicional)
- `langevin.py` - Underdamped Langevin sampler
- `soc.py` - HistoryBuffer + SOC consolidation
- `decoder.py` - EBMDecoder (MoE: 4 expertos, 2 activos)
- `geometry.py` - Operaciones Riemannianas completas

### Training and Evaluation
- `train.py` - Script principal de entrenamiento con mejoras Fase 1 ✅ MEJORADO
- `evaluate.py` - Evaluación de perplexity y calidad generativa ✅ NUEVO
- `diagnose.py` - Diagnóstico automático de checkpoints ✅ NUEVO
- `pretrain.py` - Script de pretraining existente
- `train_logger.py` - Logging detallado de entrenamiento ✅ NUEVO

### Utilities
- `dataset_utils.py` - WikiText-103 dataloader
- `vulkan_engine.py` - VulkanEBMRunner (GPU acceleration)
- `config.py` - Configuración fallback

### Documentation
- `README.md` - Este archivo con instrucciones completas ✅ NUEVO
- `requirements.txt` - Dependencias del proyecto ✅ NUEVO

### Checkpoints and Logs
- `checkpoints/` - Model checkpoints guardados cada epoch
- `logs/ebm/` - Logs detallados de entrenamiento (JSON)

---

## 🎯 Diferencias con el Diseño Original

| Aspecto | Original | Fase 1 Mejorado | Beneficio |
|---------|----------|-------------------|----------|
| Splats Init | 10K random | 50K GPT-2 embeddings | Mejor cobertura semántica |
| Training | Single phase | 3-phase curriculum | Convergencia más estable |
| Monitoreo | Básico | Métricas en vivo + alertas | Problemas detectados temprano |
| Validación | Manual | Automática con diagnósticos | Feedback en tiempo real |
| Splat Stats | Simple | Estadísticas completas | Mejor comprensión del modelo |

---

## 📖 Documentación

### Quick Start
```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar con GPU (Recomendado para AMD RX 6650XT)
python train.py --device vulkan --epochs 10 --batch-size 32

# Monitorear entrenamiento en vivo
# Los logs se guardan en logs/ebm/training_log_TIMESTAMP.json
```

### Diagnóstico
```bash
# Análisis de checkpoint específico
python diagnose.py --checkpoint checkpoints/ebm_epoch_5.pt --device vulkan

# Diagnóstico batch de todos los checkpoints
python diagnose.py --batch --device vulkan
```

---

## 🚀 Próximos Pasos (Fase 2 - Opcionales)

Estas mejoras solo se implementarán si las de Fase 1 no resuelven los problemas de convergencia:

1. **FAISS-GPU Migration**: Aceleración real de KNN de splats
2. **Mixed Precision Training**: BF16 para 2x capacidad de batch
3. **Gradient Accumulation**: Effective batch size 8x (actualmente 1)
4. **Transformer Decoder**: Arquitectura estilo GPT-2 probada
5. **Hierarchical Sampling**: Coarse-to-fine para mayor eficiencia

---

## 🔧 Configuración

### Fase 1 Parámetros (config.py)

```python
@dataclass
class EBMConfig:
    # Ambiente
    device: str = "vulkan"  # Usar GPU AMD RX 6650XT

    # Espacio latente
    latent_dim: int = 640

    # Splats (Fase 1 mejorado)
    n_splats_init: int = 10000  # Inicial: 10K, luego expandir a 50K
    max_splats: int = 150000  # Capacidad máxima: 50K
    knn_k: int = 64

    # Curriculum learning (Fase 1 nuevo)
    enable_curriculum_learning: bool = True
    curriculum_epochs: int = 5
    curriculum_target_splats: int = 50000

    # Monitoreo (Fase 1 mejorado)
    enable_detailed_logging: bool = True
    soc_check_interval: int = 100

    # Regularización de splats (Fase 1 mejorado)
    splat_temperature: float = 0.1
    splat_weight_decay: float = 0.0
    splat_weight_decay_start: float = 1.0
    min_kappa: float = 1.0
    max_kappa: float = 50.0

    # Entrenamiento
    batch_size: int = 32
    seq_length: int = 32
    noise_levels: tuple = (0.01, 0.05, 0.1, 0.2, 0.5)

    # Dinámica Langevin
    langevin_steps: int = 200
    langevin_dt: float = 0.001
    langevin_gamma: float = 0.1
    langevin_T: float = 1.0

    # SOC (Self-Organized Criticality)
    soc_threshold: float = 0.8

    # Hierarchical context
    context_local: int = 12
    context_medium: int = 64
    context_global: int = 512

    # Decoder (MoE)
    vocab_size: int = 50257
    moe_experts: int = 4
    moe_active: int = 2
    hidden_dim: int = 1024

    # Optimización
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
```

---

## 💡 Consejos de Entrenamiento

### Para Mejor Convergencia

1. **Usar GPU** (`--device vulkan`) para acelerar el entrenamiento
2. **Monitorear los logs** en tiempo real para detectar problemas temprano
3. **Validar checkpoints** periódicamente con `diagnose.py`
4. **Ajustar curriculum learning** si la convergencia es muy lenta
5. **Verificar estadísticas de splats** para asegurar uso balanceado

### Para Evaluar Calidad

1. **Usar `evaluate.py`** para calcular perplexity en WikiText-103
2. **Generar muestras** de checkpoints sucesivos para comparar calidad
3. **Revisar métricas de energía** para asegurar convergencia estable
4. **Verificar tasa de consolidación SOC** (debe disminuir con el tiempo)

---

## 🎉 Resumen de Fase 1

**Estado**: ✅ Completado
**Archivos Nuevos**: 7 archivos mejorados/creados
**Mejoras Implementadas**: 5 categorías principales
**Beneficios Esperados**: Convergencia más rápida y estable, monitoreo en tiempo real

**Estimación de Tiempo**:
- Fase 1 (10 epochs): 2-3 horas en GPU AMD RX 6650XT
- Convergencia completa: 5-7 días adicionales (dependiendo de métricas)

---

**Última actualización**: 2026-02-21
**Autor**: Alfred 🎩
