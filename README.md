# Learning AI

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

## Contribuciones

- Archivos de implementación (01-03, test.py, test_numeric.py, utils.py): usuario
- visualize.py y 04_bilayer_nn.py: generados con asistencia AI