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
├── notebooks/
│   ├── nlp/                    # Notebooks de NLP (16–23)
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
│   │   ├── nlp_lib/            # Librería compartida
│   │   │   └── __init__.py     # Word2VecLoader class
│   │   └── ...
│   └── myWord2Vec/             # Embeddings Word2Vec por versión
│       ├── v1/                 # 11–15 original (gaianet/london, vocab 3K)
│       ├── v2/                 # 15_v2 (gaianet/london, vocab 8K)
│       └── v3/                 # 15_v3 (wikitext-103, vocab 10K, dim 128)
│           └── README.md       # Detalles de implementación
├── Makefile
├── requirements.txt
└── README.md
```

## Progresión NLP

| Notebook | Modelo | Window | Layers | Test Acc |
|----------|--------|:-----:|:------:|:--------:|
| 18 | RNN vanilla | 5 | 1 RNN | 0.403 |
| 19 | RNN + gradient clipping | 64 | 1 RNN | 0.420 |
| 20 | RNN + Bahdanau Attention | 5 | 1 RNN+Att | 0.575 |
| 21 | Self-Attention manual | 5 | 1 SA | 0.405 |
| 21_v2 | Self-Attention (window=32) | 32 | 1 SA | 0.104 |
| 21_v3 | Self-Attention (trainable emb) | 5 | 1 SA | 0.749 |
| 22 | Mini Transformer (causal+last) | 5 | 3 SA+FFN | 0.641 |
| 22_v2 | Mini Transformer (trainable emb) | 5 | 3 SA+FFN | **0.755** |
| 23 | Mini Transformer (+warmup+LS) | 5 | 3 SA+FFN | **?** |

## Hallazgos clave

1. **Embedding trainable es el factor dominante** — 21→21_v3 (+0.344), 22→22_v2 (+0.114)
2. **GlobalAvgPooling + causal mask es destructivo** — último token (22) supera al avg de 0.103 a 0.641
3. **RNN + Bahdanau Attention** — 0.575 con solo 128 params entrenables
4. **v3 embeddings** — wikitext-103 con vocab 10K, dim 128. Ver [`notebooks/myWord2Vec/v3/README.md`](notebooks/myWord2Vec/v3/README.md) para detalles.
