<div align="center">

# ☕ Machine Learning Supervisado — Café, Sueño & Salud

### Dashboard Interactivo de Predicción de Calidad del Sueño

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)

</div>

---

## 🎯 Objetivo

Aplicar un **pipeline completo de Machine Learning supervisado** desde la limpieza de datos hasta la evaluación de modelos predictivos, para predecir la **calidad del sueño** (`Sleep_Quality`) a partir de variables de consumo de café y métricas de salud.

---

## 📂 Estructura del Proyecto

```
📁 proyecto_machine_learning/
├── 📊 dashboard.py                    # Dashboard interactivo (Streamlit)
├── 📁 data/
│   └── 📄 coffee_health_dirty.csv     # Dataset original (10,200 registros)
├── 📁 .streamlit/
│   └── ⚙️ config.toml                 # Configuración de tema
├── 📋 requirements.txt                # Dependencias del proyecto
├── 📝 README.md                       # Este archivo
└── 🚫 .gitignore                      # Archivos ignorados por Git
```

---

## 🔄 Pipeline del Proyecto

```mermaid
graph LR
    A[📥 Carga de Datos] --> B[🧹 Limpieza]
    B --> C[📊 EDA]
    C --> D[🤖 Modelado]
    D --> E[📈 Evaluación]
    E --> F[🔮 Predicción]

    style A fill:#8b5cf6,stroke:#6d28d9,color:#fff
    style B fill:#3b82f6,stroke:#2563eb,color:#fff
    style C fill:#06b6d4,stroke:#0891b2,color:#fff
    style D fill:#22c55e,stroke:#16a34a,color:#fff
    style E fill:#eab308,stroke:#ca8a04,color:#fff
    style F fill:#f97316,stroke:#ea580c,color:#fff
```

### Etapas detalladas:

1. **🧹 Limpieza de Datos**
   - Estandarización de variables categóricas (mayúsculas, espacios)
   - Eliminación de duplicados (147 registros)
   - Detección y tratamiento de valores fuera de rango (Age, BMI, Sleep_Hours, Coffee_Intake)
   - Eliminación de filas sin etiqueta `Sleep_Quality` (758 registros)
   - Corrección de valores inválidos en `Smoking`

2. **📊 Análisis Exploratorio (EDA)**
   - Distribución de la variable objetivo (4 clases balanceadas)
   - Histogramas y boxplots de variables numéricas
   - Mapa de correlaciones de Pearson
   - Análisis bivariado: café y horas de sueño vs calidad del sueño

3. **🤖 Modelado (Aprendizaje Supervisado)**
   - **Regresión Logística** con GridSearchCV (C ∈ {0.1, 1, 10})
   - **Random Forest** con GridSearchCV (max_depth, min_samples_leaf)
   - Validación cruzada estratificada (5 folds)
   - Preprocesamiento integrado en pipelines (imputación + escalado + encoding)

4. **📈 Evaluación & Comparación**
   - Métricas: Accuracy, Precision, Recall, F1 (macro y weighted)
   - Matrices de confusión interactivas
   - Radar chart comparativo
   - Importancia de características (Random Forest)

5. **🔮 Predicción Interactiva**
   - Formulario con sliders para ingresar datos de un paciente
   - Predicción en tiempo real con ambos modelos
   - Visualización de probabilidades por clase

---

## 📊 Dataset

**`coffee_health_dirty.csv`** — 10,200 registros × 14 variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `Age` | Numérica | Edad del individuo |
| `Gender` | Categórica | Género (Male/Female) |
| `Country` | Categórica | País de residencia |
| `Coffee_Intake` | Numérica | Tazas de café por día |
| `Caffeine_mg` | Numérica | Cafeína consumida (mg) |
| `Sleep_Hours` | Numérica | Horas de sueño por noche |
| `Sleep_Quality` | **Objetivo** | Calidad del sueño (Excellent/Good/Fair/Poor) |
| `BMI` | Numérica | Índice de Masa Corporal |
| `Heart_Rate` | Numérica | Frecuencia cardíaca |
| `Stress_Level` | Categórica | Nivel de estrés (Low/Medium/High) |
| `Physical_Activity_Hours` | Numérica | Horas de actividad física semanal |
| `Occupation` | Categórica | Tipo de ocupación |
| `Smoking` | Binaria | Fumador (0/1) |
| `Alcohol_Consumption` | Binaria | Consumo de alcohol (0/1) |

---

## 🚀 Ejecución Local

### Requisitos previos
- Python 3.10 o superior

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/Wilmeee70/proyecto-ml-sueno.git
cd proyecto-ml-sueno

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar el dashboard

```bash
streamlit run dashboard.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube este repositorio a **GitHub**
2. Ve a [share.streamlit.io](https://share.streamlit.io/)
3. Conecta tu cuenta de GitHub
4. Selecciona:
   - **Repository:** tu repositorio
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
5. Click en **Deploy** 🚀

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Uso |
|------------|-----|
| **Python** | Lenguaje principal |
| **Streamlit** | Framework del dashboard interactivo |
| **Pandas / NumPy** | Manipulación y análisis de datos |
| **Plotly** | Visualizaciones interactivas |
| **Scikit-Learn** | Modelos de Machine Learning |
| **Google Colab** | Desarrollo inicial del pipeline |

---

<div align="center">

**Proyecto de Machine Learning Supervisado** · Análisis de Salud & Calidad del Sueño

</div>
