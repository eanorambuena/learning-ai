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
│   ├── nlp/                    # Notebooks de NLP (16–25)
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
│   │   ├── 24_mini_transformer_v4_teacherforcing.ipynb
│   │   ├── 24_v2_mini_transformer_v4_teacherforcing.ipynb
│   │   ├── 24_v3_mini_transformer_v4_teacherforcing.ipynb
│   │   ├── 25_cuda_test.ipynb
│   │   ├── nlp_lib/            # Librería compartida
│   │   │   └── __init__.py     # Word2VecLoader class
│   │   └── ...
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

| Notebook | Modelo | Window | Layers | Vocab | Test Acc |
|----------|--------|:-----:|:------:|:-----:|:--------:|
| 18 | RNN vanilla | 5 | 1 RNN | v2 | 0.403 |
| 19 | RNN + gradient clipping | 64 | 1 RNN | v2 | 0.420 |
| 20 | RNN + Bahdanau Attention | 5 | 1 RNN+Att | v2 | 0.575 |
| 21 | Self-Attention manual | 5 | 1 SA | v2 | 0.405 |
| 21_v2 | Self-Attention (window=32) | 32 | 1 SA | v2 | 0.104 |
| 21_v3 | Self-Attention (trainable emb) | 5 | 1 SA | v2 | 0.749 |
| 22 | Mini Transformer (causal+last) | 5 | 3 SA+FFN | v2 | 0.641 |
| 22_v2 | Mini Transformer (trainable emb) | 5 | 3 SA+FFN | v2 | **0.755** |
| 23 | Mini Transformer (+warmup+LS) | 5 | 3 SA+FFN | v3 | **?** |
| 23_v2 | Mini Transformer (Word2Vec v4) | 5 | 3 SA+FFN | v4 | 0.171 |
| 23_v3 | + Teacher Forcing + W32 | 32 | 3 SA+FFN | v4 | **0.342** |
| 24 | + 6L post-norm (stuck) | 32 | 6 SA+FFN | v4 | 0.090 |
| **24_v2** | **+ 4L pre-norm + clipnorm** | **32** | **4 SA+FFN** | **v4** | **0.372↑** |
| 24_v3 | copia de 24_v2 | 32 | 4 SA+FFN | v4 | ? |

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
