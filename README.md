# Learning AI

Proyectos y ejercicios para aprender IA y machine learning.

## Data Format

Rows are samples, columns are features. eg:

    | 0 0 |              | 0 0 |
X = | 0 1 |     target = | 1 1 |
    | 1 0 |              | 1 1 |
    | 1 1 |              | 1 1 |

OR:   target = [[0,0], [1,1], [1,1], [1,1]]
AND:  target = [[0,0], [0,0], [0,0], [1,1]]
XOR:  target = [[0,0], [1,1], [1,1], [0,0]]

## Setup

```bash
make i
```

## Run

```bash
make run file=01_perceptron_mse.py
make run file=02_perceptron_gd.py
make run file=03_multilayer_perceptron.py
```

## Estructura

- `utils.py` - Funciones comunes
- `test.py` - Tests para OR, AND, XOR
- `01_perceptron_mse.py` - Perceptrón con solución cerrada (least squares)
- `02_perceptron_gd.py` - Perceptrón con gradient descent
- `03_multilayer_perceptron.py` - Perceptrón con activación sigmoid
- `Makefile` - Comandos de build
- `requirements.txt` - Dependencias Python