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