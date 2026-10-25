# Bench notes — CPU optimization round (branch opt/performance)

## Method
- Script: `bench_cpu.py` (seeds fijos: `torch.manual_seed(42)`, `np.random.seed(42)`; FM
  re-seeded por llamada porque muestrea ruido/t internamente). Todos los workloads
  usan dim 640, 2000 splats activos, knn_k=64, Langevin B=8 x 100 steps.
- Timing: `time.perf_counter`, n repeticiones, se reporta mediana + min/max.
- Comparación head-to-head en la misma sesión/máquina (baseline = commit 26dd4cb
  corrido desde worktree temporal vía PYTHONPATH; optimizado = worktree final),
  para eliminar drift de máquina entre sesiones.
- Equivalencia numérica: `check_equiv` (script externo) compara bit a bit los
  outputs seedados de: índices KNN, energy forward, score, sample_langevin
  completo (100 steps), y HypersphereFlowMatching.sample. Resultado: **idénticos
  bit a bit** en todos los casos, antes y después de cada cambio.

## Resultados head-to-head (misma sesión; mediana, n=20/10/5/3 según workload)

| workload             | baseline   | optimizado  | speedup |
|----------------------|------------|-------------|---------|
| langevin_sampling    | 1966 ms    | 1106 ms     | **1.78x** |
| score_forward        | 26.7 ms    | 15.5 ms     | 1.73x   |
| find_neighbors       | 6.7 ms     | 0.01 ms (*) | memoizado |
| energy_forward       | 14.6 ms    | 13.7 ms     | 1.07x   |
| flow_matching_sample | 144.0 ms   | 131.1 ms    | 1.10x (parcialmente ruido) |
| score_network        | 11.2 ms    | 11.8 ms     | 0.95x — SIN mejora (código no tocado; ruido) |
| flow_matching_loss   | 13.4 ms    | 18.4 ms     | 0.73x — SIN mejora (dentro del ruido entre corridas 13-20 ms en ambas versiones) |

Rangos entre corridas: langevin baseline 1955-2312 ms, optimizado 1081-1290 ms.

(*) find_neighbors/energy_forward/score_forward tras la optimización miden el
estado estacionario *con cache* (la 2a..n-ésima rep usa la memoización LRU-1, que
es exactamente el patrón del loop de Langevin). Llamadas frías medidas aparte
(proceso fresco, n=1): find_neighbors B=64 frío ≈ 8.5 ms ≈ baseline (NEUTRO:
`torch.cdist` ya elegía el path mm con B>25; la ganancia fría real está en el
régimen B chico — B=8 de Langevin — donde caía al kernel broadcast lento, más la
eliminación de conversiones numpy muertas y el `import faiss` fallido por llamada).

## Qué se cambió (commits)
1. `4a91815` perf(splats): import faiss resuelto 1 vez; conversiones numpy
   eliminadas del path sin faiss (~25 MB/step de memcpy muerto); cdist→mm.
2. `daa6017` perf(splats): memoización LRU-1 de find_neighbors → 3 cómputos KNN
   por paso de Langevin en vez de 4.
3. `be38258` perf(splats): cdist → d²=‖x‖²+‖μ‖²-2x·μ (un sgemm): B=8 pasa de
   ~3.2 ms a ~0.3 ms por llamada.
4. `6304c42` perf(pglf): freqs de SinusoidalTimeEmbedding precomputadas;
   fast_non_dominated_sort vectorizado (diferencial-testeado vs HEAD: fronts
   idénticos en 9 casos con ties/duplicados/NaN).
5. `5dd983c` lint: 0 errores ruff nuevos vs baseline.

## Caveats honestos
- El path con faiss instalado se preservó igual (solo se cachea el import), pero
  NO fue ejercitado: faiss no está instalado en este entorno.
- Memoización: escrituras externas directas vía `param.data` (fuera de los
  métodos de la librería) no bumpan `_version` de torch y no invalidan la cache.
  Los mutadores de la librería (`add_splat`, `normalize`) y cualquier in-place
  (optimizer steps) sí invalidan (contador explícito + `_version`).
- El swap cdist→matmul puede reordenar near-ties dentro del redondeo float;
  verificado bit a bit en los workloads seedados (no ocurrió ningún swap).
- score_network y flow_matching_loss NO mostraron mejora medible: reportado tal
  cual, sin números inventados.
- Machine: WSL2, 8 CPUs, torch 2.14.1+cpu, numpy 2.4.6, python 3.11.15.
