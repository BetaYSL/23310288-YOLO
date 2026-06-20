# Proyecto de Visión Artificial – Detección de Celulares con YOLOv8

## Integrante

DIEGO BETANCOURT MÉNDEZ 6F 23310288

---

# Descripción del Proyecto

Este proyecto consiste en el desarrollo de un sistema de visión artificial basado en el modelo **YOLOv8 (You Only Look Once)** para la detección automática de teléfonos celulares en imágenes.

El objetivo principal es demostrar el funcionamiento del entrenamiento, ajuste y aplicación de un modelo de detección de objetos utilizando aprendizaje profundo y procesamiento de imágenes.

El desarrollo fue realizado utilizando:

* Python
* Google Colab
* Roboflow
* YOLOv8 (Ultralytics)
* PyTorch

---

# Objetivos

## Objetivo General

Desarrollar un modelo de visión artificial capaz de detectar teléfonos celulares mediante técnicas de aprendizaje profundo.

---

## Objetivos Específicos

* Construir un conjunto de datos personalizado.
* Etiquetar imágenes utilizando Roboflow.
* Entrenar un modelo YOLOv8.
* Evaluar el desempeño del modelo.
* Realizar inferencia sobre nuevas imágenes.

---

# Herramientas Utilizadas

| Herramienta  | Función                   |
| ------------ | ------------------------- |
| Python       | Desarrollo del sistema    |
| Google Colab | Entrenamiento             |
| Roboflow     | Preparación del dataset   |
| YOLOv8       | Detección de objetos      |
| OpenCV       | Procesamiento de imágenes |
| PyTorch      | Motor de entrenamiento    |

---

# Dataset Utilizado

Dataset personalizado generado mediante Roboflow.

## Clase Entrenada

```text
celular
```

## Configuración utilizada

```text
Epochs: 30
Resolución: 640 × 640
Modelo: YOLOv8 Nano
```

---

# Resultados del Entrenamiento

Resultados obtenidos durante el entrenamiento:

```text
Precisión (P): 98.5%
Recall (R): 78.3%
mAP50: 91.2%
mAP50–95: 81.7%
```

Interpretación:

El modelo logró detectar teléfonos celulares con un nivel alto de precisión considerando el tamaño del conjunto de datos utilizado.

---

# Estructura del Repositorio

```plaintext
ProyectoVisionArtificial/
│
├── notebook/
│      EntrenamientoYOLO_DeteccionCelulares.ipynb
│
├── modelo/
│      best.pt
│
├── evidencias/
│      evidencia1.jpg
│      evidencia2.jpg
│      evidencia3.jpg
│
├── requirements.txt
│
└── README.md
```

---

# Instalación

Instalar dependencias:

```bash
pip install ultralytics
```

---

# Ejecución

Ejecutar el notebook en Google Colab.

Posteriormente:

```bash
Entrenar → Generar best.pt → Ejecutar inferencia
```

---

# Código Fuente

## 1. Instalación de Dependencias

```python
!pip install ultralytics
```

---

## 2. Importación del Modelo

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("Modelo cargado correctamente")
```

---

## 3. Carga del Dataset

```python
from google.colab import files

uploaded = files.upload()
```

---

## 4. Descompresión

```python
!unzip "CelularesYOLO.v3i.yolov8.zip"
```

---

## 5. Verificación

```python
!ls
```

---

## 6. Verificación GPU

```python
import torch

print(torch.cuda.is_available())
```

---

## 7. Entrenamiento

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="/content/data.yaml",
    epochs=30,
    imgsz=640
)
```

---

## 8. Cargar Modelo Entrenado

```python
modelo = YOLO(
"/content/runs/detect/train-3/weights/best.pt"
)
```

---

## 9. Cargar Imagen

```python
from google.colab import files

files.upload()
```

---

## 10. Detección

```python
from PIL import Image

resultado = modelo(
"/content/imagen_prueba.jpg"
)

for r in resultado:

    imagen = Image.fromarray(
        r.plot()[...,::-1]
    )

    display(imagen)
```

---
