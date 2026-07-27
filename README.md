# Learning AI

**Python 3.12 requerido** — Para TensorFlow, usar entorno `tfenv` con `conda activate tfenv`.

## Instalación

```bash
make i
```

## Ejecutar scripts Python

```bash
make run file=basics/01_perceptron_mse.py
```

## Estructura

```
├── basics/                     # Scripts Python básicos (01–11)
│   ├── 01_perceptron_mse.py    # Solución analítica con least squares
│   ├── 02_perceptron_gd.py     # Entrenamiento con gradient descent
│   ├── 03_monolayer_nn.py      # Red monocapa con activación
│   ├── 04_bilayer_nn.py        # Red bicapa con capa oculta
│   ├── 05_generic_network.py   # Red genérica N capas
│   ├── 06_pytorch_xor.py       # XOR con PyTorch
│   ├── 07_tensorflow_xor.py    # XOR con TensorFlow
│   ├── 08_keras_xor.py         # XOR con Keras
│   ├── 09_sklearn_xor.py       # XOR con sklearn
│   ├── 10_comparison_circle.py # Comparación frameworks (círculo)
│   ├── 11_onehot_embedding.py  # One-hot embedding manual
│   ├── utils.py                # Funciones comunes
│   ├── test.py                 # Tests binarios: OR, AND, XOR
│   ├── test_numeric.py         # Tests numéricos: ADD, MAXMIN, PROD, DIVMOD
│   ├── visualize.py            # Visualización de fronteras
│   └── plots/                  # Gráficos generados por los tests
├── .env.example                # Variables de entorno (HF_TOKEN, LD_LIBRARY_PATH)
├── .vscode/settings.json       # Configuración VS Code
├── notebooks/
│   ├── nlp/                    # Notebooks de NLP (16–25+)
│   │   ├── 16_word2vec_usage.ipynb
│   │   ├── 17_word2vec_next_word.ipynb
│   │   ├── 18_rnn_manual.ipynb
│   │   ├── 19_rnn_gradient_clipping.ipynb
│   │   ├── 20_rnn_bahdanau_attention.ipynb
│   │   ├── 21_self_attention.ipynb
│   │   ├── 21_v2_self_attention_window32.ipynb
│   │   ├── 21_v3_self_attention_trainable.ipynb
│   │   ├── 22_mini_transformer.ipynb
│   │   ├── 22_v2_mini_transformer_moredata_trainable.ipynb
│   │   ├── 23_mini_transformer_warmup_ls.ipynb
│   │   ├── 23_v2_mini_transformer_v4.ipynb
│   │   ├── 23_v3_mini_transformer_v4_teacherforcing.ipynb
│   │   ├── 24_v2_santiago2.8Mparams_5Mchars_mini_transformer_colab.ipynb
│   │   ├── 24_v3_santiago2.8Mparams_20Mchars_mini_transformer_colab.ipynb
│   │   ├── 24_v4_santiago2.8Mparams_20Mchars_mixedFP16.ipynb
│   │   ├── 26_ezeiza9Mparams_50Mchars.ipynb
│   │   ├── 25_cuda_test.ipynb
│   │   ├── nlp_lib/            # Librería compartida
│   │   │   └── __init__.py     # Word2VecLoader class
│   │   └── checkpoints/        # Pesos guardados por modelo
│   │       └── ezeiza/         # Mejores pesos de Ezeiza
│   └── myWord2Vec/             # Embeddings Word2Vec por versión
│       ├── v1/                 # 11–15 original (gaianet/london, vocab 3K)
│       ├── v2/                 # 15_v2 (gaianet/london, vocab 8K)
│       ├── v3/                 # 15_v3 (wikitext-103, vocab 10K, dim 128)
│       │   └── README.md       # Detalles de implementación
│       └── v4/                 # 15_v4 (wikitext-103, vocab 9K, dim 128, sin filtro)
├── Makefile
├── requirements.txt
└── README.md
```

## Progresión NLP

| Notebook | Modelo | Params | Dim | Window | Layers | Vocab | Chars | Test Acc |
|----------|--------|:-----:|:---:|:------:|:------:|:-----:|:-----:|:--------:|
| 16 | Word2Vec básico | — | 128 | — | — | v1 | 100K | — |
| 17 | Word2Vec + predict | — | 128 | 3 | — | v1 | 100K | — |
| 18 | RNN vanilla | 128 | — | 5 | 1 RNN | v2 | — | 0.403 |
| 19 | RNN + gradient clipping | 128 | — | 64 | 1 RNN | v2 | — | 0.420 |
| 20 | RNN + Bahdanau Attention | 128 | — | 5 | 1 RNN+Att | v2 | — | 0.575 |
| 21 | Self-Attention manual | ~130K | — | 5 | 1 SA | v2 | — | 0.405 |
| 21_v2 | Self-Attention (window=32) | ~130K | — | 32 | 1 SA | v2 | — | 0.104 |
| 21_v3 | Self-Attention (trainable emb) | ~8M | — | 5 | 1 SA | v2 | — | 0.749 |
| 22 | Mini Transformer (causal+last) | ~2.5M | 128 | 5 | 3 SA+FFN | v2 | — | 0.641 |
| 22_v2 | Mini Transformer (trainable emb) | ~8M | 128 | 5 | 3 SA+FFN | v2 | — | **0.755** |
| 23 | + warmup + label smoothing | ~8M | 128 | 5 | 3 SA+FFN | v3 | 50K | **?** |
| 23_v2 | + Word2Vec v4 | ~8M | 128 | 5 | 3 SA+FFN | v4 | 50K | 0.171 |
| 23_v3 | + Teacher Forcing + W32 | ~8M | 128 | 32 | 3 SA+FFN | v4 | 50K | **0.342** |
| 24 | + 6L post-norm (stuck) | ~3.5M | 128 | 32 | 6 SA+FFN | v4 | 20M | 0.090 |
| **24_v2** | **Santiago 5M (4L pre-norm)** | **2.8M** | **128** | **32** | **4 SA+FFN** | **v4** | **5M** | **0.372** |
| **24_v3** | **Santiago 20M** | **2.8M** | **128** | **32** | **4 SA+FFN** | **v4** | **20M** | **~0.37** |
| 24_v4 | Santiago 20M mixedFP16 | 2.8M | 128 | 32 | 4 SA+FFN | v4 | 20M | NaN (crash) |
| **26_ezeiza** | **Ezeiza 9.1M** | **9.1M** | **320** | **64** | **6 SA+FFN** | **v4** | **50M** | **?** (running) |

## Hallazgos clave

1. **Embedding trainable es el factor dominante** — 21→21_v3 (+0.344), 22→22_v2 (+0.114)
2. **GlobalAvgPooling + causal mask es destructivo** — último token (22) supera al avg de 0.103 a 0.641
3. **RNN + Bahdanau Attention** — 0.575 con solo 128 params entrenables
4. **Teacher forcing rompe el plateau** — 0.171 (23_v2, last-token) → 0.342 (23_v3, teacher forcing + W32)
5. **Post-norm + 6 capas sin clipnorm diverge** — accuracy plana en 0.09 desde epoch 1. Warmup no ayuda.
6. **Pre-norm + clipnorm destraba profundidad** — 24_v2 con 4 capas pre-norm + Adam(clipnorm=1.0) mejora estable sin warmup.
7. **LR constante > warmup para modelos chicos** — warmup fue contraproducente con 128-dim y 6 capas. LR constante (0.001) funciona mejor.
8. **v4 embeddings** — wikitext-103 con vocab 9K, dim 128, sin filtro `len>1`. Word2VecLoader soporta multi-versión.
9. **Word2VecLoader multi-versión** — `Word2VecLoader(version='v4')` resuelve path relativo a `__file__`, no CWD.
10. **Overlapping windows con stride=1 maximiza datos** — cada char aparece en W ventanas; el modelo aprende de todas las posiciones sin costo extra de memoria.
11. **mixed_float16 en modelos <10M params causa NaN** — ReLU + -1e9 en float16 → inf + (-inf) = NaN. LossScaleOptimizer no puede rescatar forward pass corrupto. float32 es más simple y cabe en 6GB.
12. **LossScaleOptimizer manual = kernel crash** — TF 2.20+ auto-wrap con mixed_float16. Hacerlo manual duplica el wrapper y silencia el crash.
13. **Cache en disco (/tmp) con .cache() reduce I/O** — datasets de 50M chars con shuffle+batch se benefician de cachear el pipeline preprocesado.
14. **Más datos > más parámetros (hasta saturación)** — Santiago 5M→20M chars subió accuracy de ~0.25 a ~0.37 con mismos 2.8M params. El límite de saturación está en ~10-15× los params en chars.
15. **Depth Delusion: width > depth** — Para ~8M params, 6L×320d supera a 8L×256d. El gradiente se desvanece en capas profundas con poca dimensión.

## Proximidad a Transformer Base

- Santiago (2.8M) está **23×** más lejos de Transformer Base (65M) que de una RNN simple
- Ezeiza (9.1M) es **7×** más cerca de Transformer Base que Santiago
- Próximo salto Schipol (18M) pondrá el modelo a **28%** del camino a Transformer Base

## Memory bottleneck

`keras.utils.to_categorical()` convierte etiquetas enteras a one-hot (N, vocab_size). Con vocab=3904 y N=28K son ~424 MB; con N=110K ya son ~1.6 GB y la RAM se agota.

**Solución:** usar `SparseCategoricalCrossentropy()` en vez de `CategoricalCrossentropy()` + `to_categorical()`. Las etiquetas siguen siendo enteros y ocupan ~N×4 bytes (~112 KB para 28K). El trade‑off es que `SparseCategoricalCrossentropy()` **no soporta `label_smoothing`** en TF 2.21.

## GPU (CUDA en WSL2)

RTX 3060 (6 GB VRAM) con NVIDIA driver 555.97, CUDA 12.5.

**Requisito:** `LD_LIBRARY_PATH` debe incluir el `lib/` del conda env `tfenv`. Configuración en `.env.example`.

| Plataforma | GPU | Tiempo/epoch (8781 steps, window=32) |
|:-----------|:---:|:------------------------------------:|
| CPU local | — | ~73 min |
| WSL2 RTX 3060 | sí (con fix) | ~3 min |
| Colab T4 | sí | ~3 min |

**Problema conocido:** `WARMUP_STEPS` + post-norm con 6+ capas causa estancamiento. Pre-norm + `clipnorm` + LR constante lo resuelve.

## Roadmap: De MicroLM a Transformer Base

Nombres de aeropuertos/ciudades que marcan el avance geográfico hacia el paper "Attention Is All You Need" (65M params).

### Fase 0: Fundamentos

| Notebook | Descripción |
|----------|-------------|
| 16 | Word2Vec básico |
| 17 | Word2Vec + predicción |
| 18–20 | RNN manual + gradient clipping + Bahdanau attention |
| 21–21_v3 | Self-attention desde cero |
| 22–23_v3 | Mini transformers |

### Fase 1: MicroLMs

| Notebook | Params | Dim | Layers | FF | Heads | Window | Batch | Chars | Tiempo |
|----------|--------|-----|--------|-----|-------|--------|-------|-------|--------|
| 24_v2_santiago2.8M_5Mchars | 2.8M | 128 | 4 | 128 | 4 | 32 | 128 | 5M | ~15 min |
| 24_v3_santiago2.8M_20Mchars | 2.8M | 128 | 4 | 128 | 4 | 32 | 128 | 20M | ~1 h |
| 24_v4_santiago2.8M_20Mchars_mixedFP16 | 2.8M | 128 | 4 | 128 | 4 | 32 | 128 | 20M | crash (NaN) |
| **26_ezeiza9.1M_50Mchars** | **9.1M** | **320** | **6** | **640** | **8** | **64** | **64** | **50M** | **~3 h/epoch** |

### Fase 2: SLMs (próximos)

| Modelo | Código | Params | Dim | Layers | FF | Heads | Window | Batch | Chars | VRAM | Tiempo/epoch |
|--------|--------|--------|-----|--------|-----|-------|--------|-------|-------|------|-------------|
| **Schipol** | 26_schipol18M_80Mchars | **~18M** | **480** | **6** | **960** | **8** | 64 | 32 | 80M | ~2GB | ~6 h |
| **Eindhoven** | 27_eindhoven38M_120Mchars | **~38M** | **576** | **8** | **2304** | **8** | 64 | 16 | 120M | ~3.5GB | ~12 h |
| Transformer Base | — | 65M | 768 | 8 | 3072 | 12 | 128 | 16 | 200M | ~5GB | ~20 h |

### Depth Delusion (arXiv 2601.20994, Ene 2026)

Paper clave: "The Depth Delusion: Why Transformers Should Be Wider, Not Deeper".

**Hallazgos:**
- Width escala 2.8× más rápido que depth: D* ∝ C^0.12, W* ∝ C^0.34
- Critical depth: D_crit ∝ W^0.44. Pasado ese punto, más capas empeora el loss
- A escala 7B: 32L×4096W (6.92B) supera a 64L×2816W (7.08B) por 0.12 nats

**Implicación práctica:** para un presupuesto de parámetros fijo, priorizar width sobre depth. Ezeiza se diseñó con esta filosofía: 6 layers + 320 dim en vez de 8 layers + 256 dim.
