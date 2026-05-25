# Learning AI

**Python 3.12 requerido** - Para TensorFlow, usar entorno `tfenv` con `conda activate tfenv`.

Proyectos y ejercicios para aprender IA y machine learning desde cero, implementando redes neuronales con Numba.

## Instalación

```bash
make i
```

## Ejecutar

```bash
make run file=01_perceptron_mse.py   # Perceptrón con solución cerrada (least squares)
make run file=02_perceptron_gd.py    # Perceptrón con gradient descent
make run file=03_monolayer_nn.py     # Red neuronal monocapa con activación
make run file=04_bilayer_nn.py        # Red neuronal bicapa (capas ocultas)
```

## Estructura de Archivos

### Implementaciones (creadas por el usuario)
- `01_perceptron_mse.py` - Solución analítica con least squares
- `02_perceptron_gd.py` - Entrenamiento con gradient descent
- `03_monolayer_nn.py` - Red monocapa con activación (sigmoid/identity)
- `04_bilayer_nn.py` - Red bicapa con capa oculta (generado con asistencia AI)

### Utilities y Tests
- `utils.py` - Funciones comunes: forward, activations, mean_reduce, apply_gradients
- `test.py` - Tests binarios: OR, AND, XOR
- `test_numeric.py` - Tests numéricos: ADD, MAXMIN, PROD, DIVMOD
- `visualize.py` - Visualización de fronteras de decisión (generado con asistencia AI)

## Data Format

Filas = samples, columnas = features.

```
X = [[0, 0],     target = [[0, 0],
     [0, 1],              [1, 1],
     [1, 0],              [1, 1],
     [1, 1]]              [1, 1]]
```

## Tests Incluidos

### Binarios (sigmoid)
- **OR** - cualquier input 1 → output 1
- **AND** - ambos inputs 1 → output 1
- **XOR** - inputs diferentes → output 1 (igual → 0)

### Numéricos (identity)
- **ADD** - [a+b, a-b]
- **MAXMIN** - [max(a,b), min(a,b)]
- **PROD** - [a*b, a//b]
- **DIVMOD** - [a//b, a%b]

### Clasificación
- **CIRCLE** - puntos dentro vs fuera de un círculo

## Visualizaciones

Los tests generan gráficos en `plots/` mostrando:
- Frontera de decisión (regiones de clasificación)
- Puntos de entrenamiento coloreados por clase

```bash
ls plots/
```

## Aprendizajes Clave

### Monolayer (03)
- Una sola capa = transformación lineal
- **No puede resolver XOR** - solo dibuja fronteras rectas
- Funciona para OR, AND (problemas linealmente separables)

### Bilayer (04)
- Dos capas = transformación no-lineal
- **Puede resolver XOR** - puede dibujar fronteras curvas
- Puede resolver problemas no-linealmente separables (circulos, etc.)
- La capa oculta permite representar funciones más complejas

### RNNs Vanilla - Vanishing & Exploding Gradients (18)

**Problema de diseño fundamental en RNNs vanilla:**

Durante backpropagation through time (BPTT), el gradiente se multiplica por la matriz de pesos `Wh` para cada paso temporal:
```
dL/dW = dL/dh * dh/dW₁ * dW₁/dW₂ * ... (n veces, donde n = seq_length)
```

Si seq_length = 256 palabras:
- **Vanishing Gradient**: Si derivada de tanh < 1, se multiplica 256 veces → gradiente → 0 (no aprende)
- **Exploding Gradient**: Si valores de Wh > 1, se multiplica 256 veces → gradiente → ∞ (pesos explotan)

**Por qué ocurre:**
1. `tanh` devuelve valores en [-1, 1], derivada máx = 0.25
2. Si Wh ≈ 0.95, después de 256 pasos: 0.95²⁵⁶ ≈ 0 (vanishing)
3. Si Wh ≈ 1.05, después de 256 pasos: 1.05²⁵⁶ ≈ ∞ (exploding)

**Síntomas observados en Notebook 18:**
- train_acc crece muy lentamente (vanishing): 0.09 → 0.28 en 20 epochs
- Regularización sola no lo resuelve (es un problema estructural, no de overfitting)

**Soluciones implementadas en Notebook 19:**
1. **Gradient Clipping** - Limita magnitud de gradientes durante actualización
2. **Truncated BPTT** - Limita backprop a últimos N pasos (no todos los 256)
3. **LeakyReLU** - Sustituye tanh por activación con derivada > 0 siempre

### Notebook 19 — RNN + gradient clipping (window=64)

**Problema identificado (original):** Usaba `step=5`, generando solo 4,718 secuencias — 5x menos que los demás notebooks.

**Corrección:** Cambiado a `step=1` → 23,587 secuencias. Resultados actualizados:

| Métrica | 18 (baseline, w=5) | 19 (step=5, w=64) | 19 corregido (step=1, w=64) |
|---------|:----------------:|:-----------------:|:--------------------------:|
| Secuencias | 23,646 | 4,718 | **23,587** |
| Épocas entrenadas | ~68 | ~11 | **~73** |
| Test accuracy (tanh) | 0.403 | 0.105 | **0.420** |
| Test accuracy (LeakyReLU) | — | 0.105 | **0.112** |

**Conclusiones:**
1. El fracaso inicial de 19 fue por `step=5` (pocos datos), no por la ventana de 64
2. Con datos suficientes, gradient clipping + truncated BPTT SÍ funciona (0.420 vs 0.403)
3. LeakyReLU sigue siendo peor que tanh para este caso (0.112)
4. La RNN con window=64 logra 0.420, pero aún por debajo de attention (0.575)

**Lección:** Las micro-optimizaciones ayudan si hay datos suficientes, pero attention sigue siendo superior.

### Notebook 20 — RNN + Attention

**Salto cualitativo:** En vez de parchar la RNN, cambiamos la arquitectura. Attention permite que el modelo "mire" todos los estados ocultos `h_1..h_T` y decida cuáles son relevantes mediante pesos aprendidos.

**Cómo funciona Bahdanau Attention:**
```
h_1..h_T  = estados ocultos de cada paso (los guardamos TODOS)
score_i   = v^T * tanh(W_att * h_i + W_att * h_T)  # relevancia de h_i
alpha_i   = softmax(score_i)                         # normalización a [0,1]
c         = sum(alpha_i * h_i)                       # context vector ponderado
output    = softmax(Dense([c; h_T]))                  # predice con contexto + final
```

**Ventaja:** La atención añade ~5% más parámetros pero permite que el gradiente fluya directamente a cualquier paso, sin depender de la multiplicación recurrente.

| Notebook | Modelo | Test Accuracy | Mejora vs 18 |
|----------|--------|---------------|--------------|
| 18 | RNN vanilla | 0.403 | — |
| 19 | RNN + clipping | 0.105 | -74% |
| **20** | **RNN + Attention** | **0.575** | **+43%** |

**Resultado:** Atención (20) supera significativamente a la RNN vanilla (18). Pasa de 40.3% a 57.5% — una mejora de +17 puntos porcentuales. Esto confirma que el **salto arquitectónico** (añadir atención) es más efectivo que las micro-optimizaciones de gradiente (19).

### Notebook 21 — Self-Attention Manual (sin RNN, window=5)

**Motivación:** Si atención sobre RNN ya da +43%, ¿qué pasa si eliminamos la RNN por completo? Self-attention (Transformer) procesa toda la secuencia en paralelo, sin dependencia secuencial.

**Arquitectura (manual, sin built-in de Keras):**
```
Input (5 tokens) -> Embedding -> + Positional Encoding
  -> Q=Wq(x), K=Wk(x), V=Wv(x)
  -> scores = Q·K^T / sqrt(d_k) -> softmax -> att_weights·V
  -> Concat heads -> LayerNorm + residual
  -> GlobalAveragePooling -> Dense -> softmax
```

**Por qué manual en vez de `layers.MultiHeadAttention`:**
- Más rápido para secuencias cortas (sin overhead interno)
- Didáctico: se ve exactamente el scaled dot-product
- Fácil extraer att_weights para heatmaps

**Incluye:** Heatmaps de atención (matriz 5×5), balance recibida/emitida, comparación entre contextos.

### Notebook 22 — Self-Attention con window=32

**Motivación:** Notebook 19 intentó window=64 con RNN y falló (10.5% con step=5). Self-attention NO sufre vanishing gradient, debería escalar a ventanas grandes.

**Resultado:** Fracaso — **0.104**. Overfitting temprano (época 7 de 100). train_acc subía, val_acc caía.

**Causa:** 1 sola capa de self-attention no tiene suficiente capacidad para 32 tokens. El modelo memoriza el training set pero no generaliza.

**Lección:** Self-attention sola no escala a ventanas grandes. Se necesitan **múltiples capas apiladas** + **feed-forward networks** por capa — como un Transformer real.

### Progresión completa (18 → 22)

| Notebook | Modelo | Window | step | Test Accuracy |
|----------|--------|:-----:|:----:|:------------:|
| 18 | RNN vanilla | 5 | 1 | 0.403 |
| 19 | RNN + gradient clipping | 64 | 5 (original) | 0.105 |
| 19 | RNN + gradient clipping | 64 | 1 (corregido) | 0.420 |
| 20 | RNN + Bahdanau Attention | 5 | 1 | **0.575** |
| 21 | Self-Attention manual | 5 | 1 | 0.405 |
| 22 | Self-Attention manual | 32 | 1 | 0.104 |
| **23** | **Mini Transformer (3 capas + FFN)** | **32** | **1** | **pendiente** |

### Notebook 23 — Mini Transformer (Stacked SA + FFN + Causal Mask)

**Motivación:** Notebook 22 (1 capa self-attention, window=32) fracasó (0.104). Ahora apilamos 3 bloques Transformer completos.

**Arquitectura (diagrama incluido en notebook):**
```
Input → Embedding + PosEncoding
  → [Block 1: Self-Attention(causal) → Add&Norm → FFN → Add&Norm]
  → [Block 2: Self-Attention(causal) → Add&Norm → FFN → Add&Norm]
  → [Block 3: Self-Attention(causal) → Add&Norm → FFN → Add&Norm]
  → GlobalAveragePooling → Dense → softmax
```

**Nuevo respecto a notebooks anteriores:**
1. **Stacked layers** (3 bloques) — profundidad real
2. **FFN por capa** — `Dense(128, ReLU) → Dense(64)` en cada bloque
3. **Causal mask** — atención triangular (lower triangular), cada token solo ve previos
4. **Diagrama de arquitectura** — generado con matplotlib, similar al paper

**Lo que ya tenemos (coincide con Transformer real):**
- Positional Encoding sinusoidal ✓
- Multi-Head Self-Attention (4 heads) ✓
- Residual connections + LayerNorm ✓
- FFN por capa ✓
- Causal masking ✓
- Dropout ✓

**Lo que falta (para un Transformer completo):**
- Label smoothing
- Learning rate warmup
- 6+ capas (vs 3)
- Embeddings entrenables (vs frozen Word2Vec)

## Decisión de Framework

Al implementar redes más complejas (04, 05), el código se volvió muy largo y difícil de mantener.
Evaluamos 5 opciones:

1. **Numba (manual)** - Lo que usamos hasta ahora
   - ✅ Aprendimos el fondo (forward, backprop, gradientes)
   - ❌ Solo CPU, debugging difícil, código muy largo

2. **PyTorch** - La elección final ⭐
   - ✅ Pythonico, fácil de debuggear, GPU automático
   - ✅ Dynamic graphs, documentación excelente
   - ✅ 75% de papers de investigación lo usan
   - ✅ Mayor empleabilidad

3. **TensorFlow/Keras**
   - ❌ Muy abstracto - "no aprendés nada"
   - ✅ Ecosistema de production

4. **JAX**
   - ✅ Similar a Numba, auto-differentiation
   - ❌ Requiere skills de functional programming

5. **scikit-learn**
   - ❌ Muy abstracto, poco flexible

**Conclusión:** Después de 01-03 (manual) y 04-05 (AI generated), migramos a PyTorch para seguir aprendiendo de forma más práctica.

## PyTorch vs TensorFlow (08, 07)

Mismo problema (XOR), misma arquitectura (2-8-1), mismo learning rate (0.1):

| Framework | Epochs | Loss Final | Predicciones |
|-----------|--------|------------|--------------|
| TensorFlow (07) | 10,000 | 0.0063 | [0.05, 0.93, 0.92, 0.09] |
| PyTorch (06) | 10,000 | 0.0152 | [0.11, 0.90, 0.89, 0.09] |

**TensorFlow fue ~5x más rápido** con mismas condiciones. Posibles razones:
- Inicialización de pesos por defecto diferente
- Implementación de SGD internamente distintas
- Precisión numérica entre frameworks

## Comparación Completa (10)

Problema CIRCLE (clasificación círculo interno vs externo), misma arquitectura (2-16-1), mismos hiperparámetros:

| Framework | Accuracy Final |
|-----------|----------------|
| TensorFlow legacy | **100%** |
| Keras | 66.67% |
| PyTorch | 66.67% |
| sklearn | 66.67% |

**Aprendizaje clave: La inicialización de pesos importa**
- TF legacy usa `random_normal` por defecto → converge bien
- PyTorch usa Kaiming uniform → se estanca
- Keras usa Glorot uniform → se estanca
- sklearn tiene su propia inicialización → se estanca

Para redes pequeñas (2 capas), la inicialización "correcta" (Kaiming/Glorot) puede ser peor que random_normal simple.

## Contribuciones

- Archivos de implementación (01-03, test.py, test_numeric.py, utils.py): usuario
- visualize.py, 04, 05: generados con asistencia AI
- 06: PyTorch, 07: TensorFlow

## Embeddings y Word2Vec (notebooks 11-15)

Sección dedicada a embeddings de palabras y Word2Vec:

### Datasets

- **11** - Dataset local (texto hardcodeado)
- **12** - TensorFlow Embedding Layer
- **13** - Word2Vec eficiente (texto local, rápido)
- **14** - Word2Vec con HuggingFace (`gaianet/london`) - dataset público, rápido
- **15** - Word2Vec con Negative Sampling (`gaianet/london`) - más eficiente

### Dataset gaianet/london ⭐

Dataset público de HuggingFace que contiene artículos sobre Londres. Es muy rápido de cargar y perfecto para experimentar con Word2Vec.

```python
from datasets import load_dataset
ds = load_dataset("gaianet/london", split="train")
texts = [row['text'] for row in ds][:10000]
```

Ventajas:
- Descarga rápida (~MB)
- Texto en inglés, vocabulario rico
- No requiere filtrado por idioma
- Ideal para demos y experimentos