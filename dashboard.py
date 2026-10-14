# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Dashboard Interactivo — Machine Learning Supervisado
  Análisis y predicción de calidad del sueño a partir de
  hábitos de consumo de café y métricas de salud
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Ruta base del proyecto (compatible con Streamlit Cloud) ─────
BASE_DIR = Path(__file__).resolve().parent

# ── Configuración de la página ──────────────────────────────────
st.set_page_config(
    page_title="☕ ML Supervisado — Café & Salud",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado premium ───────────────────────────────────
st.markdown("""
<style>
    /* ── Fuente global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    /* ── Fondo principal ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.25);
    }

    /* ── Tarjetas métricas ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.10));
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(139, 92, 246, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #a5b4fc !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.72rem !important;
        letter-spacing: 0.8px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
    }

    /* ── Expanders ── */
    details[data-testid="stExpander"] {
        background: rgba(30, 30, 60, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 14px;
        backdrop-filter: blur(8px);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(139, 92, 246, 0.08);
        border-radius: 10px 10px 0 0;
        border: 1px solid rgba(139, 92, 246, 0.15);
        color: #c4b5fd;
        font-weight: 500;
        padding: 10px 22px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.25) !important;
        color: #f1f5f9 !important;
        border-bottom: 2px solid #8b5cf6 !important;
    }

    /* ── DataFrames ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Título hero ── */
    .hero-title {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .hero-title h1 {
        background: linear-gradient(90deg, #c084fc, #818cf8, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-title p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Glassmorphism card ── */
    .glass-card {
        background: rgba(30, 30, 60, 0.55);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 16px;
    }
    .glass-card h3 {
        color: #c4b5fd;
        margin-top: 0;
        font-weight: 600;
    }
    .glass-card p, .glass-card li {
        color: #cbd5e1;
        line-height: 1.7;
    }

    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    .badge-purple { background: rgba(139,92,246,0.22); color: #c4b5fd; }
    .badge-blue   { background: rgba(59,130,246,0.22);  color: #93c5fd; }
    .badge-green  { background: rgba(34,197,94,0.22);   color: #86efac; }
    .badge-amber  { background: rgba(245,158,11,0.22);  color: #fcd34d; }

    /* ── Dividers ── */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.4), transparent);
        margin: 30px 0;
    }

    /* ── Custom footer ── */
    .custom-footer {
        text-align: center;
        padding: 30px 20px 20px 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(139,92,246,0.2);
    }
    .custom-footer p {
        color: #64748b;
        font-size: 0.78rem;
        margin: 3px 0;
    }
    .custom-footer a {
        color: #8b5cf6;
        text-decoration: none;
    }

    /* ── Hide Streamlit branding but keep sidebar toggle visible ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    header[data-testid="stHeader"] [data-testid="stToolbar"] {
        visibility: hidden;
    }
    header[data-testid="stHeader"] [data-testid="stDecoration"] {
        display: none;
    }
    /* Mantener la navegación visible para que no dependa del estado guardado en el navegador. */
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        flex: 0 0 300px !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* ── Menú de Navegación Lateral Estilizado ── */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 5px 0;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 10px 14px;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin-bottom: 0px;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(139, 92, 246, 0.22);
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateX(4px);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.35), rgba(99, 102, 241, 0.25)) !important;
        border-color: #8b5cf6 !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.35) !important;
    }
    div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #f1f5f9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE LIMPIEZA (reproducen exactamente el Colab)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV original."""
    return pd.read_csv(ruta)


@st.cache_data(show_spinner=False)
def limpiar_datos(df: pd.DataFrame):
    """
    Pipeline de limpieza — reproduce paso a paso el código Colab.
    Devuelve: (df_limpio, log) donde log es un diccionario con métricas
    del proceso de limpieza.
    """
    log = {}
    log["registros_originales"] = len(df)
    log["nulos_originales"] = df.isna().sum().to_dict()
    log["duplicados_originales"] = int(df.duplicated().sum())

    df_c = df.copy()

    # 1. Estandarizar columnas de texto ──────────────────────────
    df_c["Sleep_Quality"] = df_c["Sleep_Quality"].str.strip().str.capitalize()
    df_c["Gender"] = df_c["Gender"].str.strip()
    df_c["Stress_Level"] = df_c["Stress_Level"].str.strip().str.capitalize()

    # 2. Eliminar duplicados ─────────────────────────────────────
    filas_antes = len(df_c)
    df_c = df_c.drop_duplicates()
    log["duplicados_eliminados"] = filas_antes - len(df_c)

    # 3. Valores fuera de rango → NaN ───────────────────────────
    edad_inv = (df_c["Age"] < 10) | (df_c["Age"] > 120)
    bmi_inv = df_c["BMI"] > 60
    sueno_inv = (df_c["Sleep_Hours"] < 2) | (df_c["Sleep_Hours"] > 20)
    cafe_neg = df_c["Coffee_Intake"] < 0

    log["edad_fuera_rango"] = int(edad_inv.sum())
    log["bmi_extremo"] = int(bmi_inv.sum())
    log["sueno_fuera_rango"] = int(sueno_inv.sum())
    log["cafe_negativo"] = int(cafe_neg.sum())

    df_c.loc[edad_inv, "Age"] = np.nan
    df_c.loc[bmi_inv, "BMI"] = np.nan
    df_c.loc[sueno_inv, "Sleep_Hours"] = np.nan
    df_c.loc[cafe_neg, "Coffee_Intake"] = np.nan

    # 4. Eliminar filas sin etiqueta ─────────────────────────────
    filas_antes = len(df_c)
    df_c = df_c.dropna(subset=["Sleep_Quality"]).copy()
    log["filas_sin_etiqueta"] = filas_antes - len(df_c)

    # 5. Smoking: solo 0/1 ──────────────────────────────────────
    smoking_inv = ~df_c["Smoking"].isin([0, 1])
    log["smoking_invalido"] = int(smoking_inv.sum())
    df_c.loc[smoking_inv, "Smoking"] = np.nan

    log["registros_limpios"] = len(df_c)
    log["nulos_finales"] = df_c.isna().sum().to_dict()

    return df_c, log


# ═══════════════════════════════════════════════════════════════
# FUNCIONES DE ML (reproducen exactamente el Colab)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def entrenar_modelos(_df_limpio):
    """Entrena Logistic Regression y Random Forest con GridSearchCV."""
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    from sklearn.base import clone

    X = _df_limpio.drop(columns=["Sleep_Quality"])
    y = _df_limpio["Sleep_Quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Columnas
    binarias = ["Smoking", "Alcohol_Consumption"]
    numericas = [c for c in X_train.select_dtypes(include="number").columns if c not in binarias]
    categoricas = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

    # Preprocesador
    proceso_numerico = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler())
    ])
    proceso_categorico = Pipeline([
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocesador = ColumnTransformer([
        ("numericas", proceso_numerico, numericas),
        ("categoricas", proceso_categorico, categoricas),
        ("binarias", SimpleImputer(strategy="most_frequent"), binarias)
    ])

    validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression ──
    pipeline_lr = Pipeline([
        ("preprocesamiento", clone(preprocesador)),
        ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])
    busqueda_lr = GridSearchCV(
        pipeline_lr,
        param_grid={"modelo__C": [0.1, 1, 10]},
        scoring="f1_macro", cv=validacion, n_jobs=-1
    )
    busqueda_lr.fit(X_train, y_train)

    # ── Random Forest ──
    pipeline_rf = Pipeline([
        ("preprocesamiento", clone(preprocesador)),
        ("modelo", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1
        ))
    ])
    busqueda_rf = GridSearchCV(
        pipeline_rf,
        param_grid={
            "modelo__max_depth": [10, None],
            "modelo__min_samples_leaf": [1, 3]
        },
        scoring="f1_macro", cv=validacion, n_jobs=-1
    )
    busqueda_rf.fit(X_train, y_train)

    # ── Evaluación ──
    clases = ["Excellent", "Good", "Fair", "Poor"]
    modelos_info = {}

    for nombre, busqueda in [("Regresión Logística", busqueda_lr), ("Random Forest", busqueda_rf)]:
        est = busqueda.best_estimator_
        pred = est.predict(X_test)

        metricas = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision macro": precision_score(y_test, pred, average="macro", zero_division=0),
            "Recall macro": recall_score(y_test, pred, average="macro", zero_division=0),
            "F1 macro": f1_score(y_test, pred, average="macro", zero_division=0),
            "Precision weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1 weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        }

        report = classification_report(y_test, pred, labels=clases, digits=4, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, pred, labels=clases)

        modelos_info[nombre] = {
            "best_params": busqueda.best_params_,
            "cv_score": busqueda.best_score_,
            "metricas": metricas,
            "report": report,
            "confusion_matrix": cm,
            "predicciones": pred,
            "clases": clases,
        }

    # Datos de splits
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "X_train_nulls": X_train.isna().sum().to_dict(),
        "X_test_nulls": X_test.isna().sum().to_dict(),
        "numericas": numericas,
        "categoricas": categoricas,
        "binarias": binarias,
    }

    # Feature importances (Random Forest)
    rf_model = busqueda_rf.best_estimator_
    feature_names = numericas.copy()
    ohe = rf_model.named_steps["preprocesamiento"].named_transformers_["categoricas"].named_steps["codificacion"]
    cat_features = ohe.get_feature_names_out(categoricas).tolist()
    feature_names += cat_features + binarias
    importances = rf_model.named_steps["modelo"].feature_importances_

    return modelos_info, split_info, feature_names, importances


# ═══════════════════════════════════════════════════════════════
# FOOTER REUTILIZABLE
# ═══════════════════════════════════════════════════════════════

def mostrar_footer():
    """Muestra un footer profesional al final de cada página."""
    st.markdown("""
    <div class="custom-footer">
        <p style="font-size: 0.9rem; color: #c4b5fd; font-weight: 600;">☕ Machine Learning Supervisado — Análisis de Salud & Calidad del Sueño</p>
        <p style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">Pipeline integral: Limpieza de Datos · EDA · Regresión Logística & Random Forest · Simulador Predictivo</p>
        <p style="margin-top: 8px; color: #475569; font-size: 0.75rem;">Desarrollado con Streamlit, Plotly, Pandas & Scikit-Learn</p>
    </div>
    """, unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════
# CARGAR DATOS
# ═══════════════════════════════════════════════════════════════

DATA_PATH = BASE_DIR / "data" / "coffee_health_dirty.csv"

try:
    df_original = cargar_datos(str(DATA_PATH))
except FileNotFoundError:
    st.error(f"No se encontró el archivo `{DATA_PATH}`. Asegúrate de que existe en el repositorio.")
    st.stop()

df_limpio, log_limpieza = limpiar_datos(df_original)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 15px 0;">
        <span style="font-size: 3rem;">☕</span>
        <h2 style="color: #c4b5fd; margin: 5px 0 0 0; font-size: 1.15rem; font-weight: 700;">
            ML Supervisado
        </h2>
        <p style="color: #64748b; font-size: 0.78rem; margin-top: 3px;">
            Café & Calidad del Sueño
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    OPCIONES_PAGINAS = [
        "🏠 Inicio",
        "🧹 Limpieza de Datos",
        "📊 EDA — Análisis Exploratorio",
        "🤖 Modelos de ML",
        "📈 Evaluación & Comparación",
        "🔮 Predicción Interactiva",
    ]

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = OPCIONES_PAGINAS[0]

    def set_nav_page(target_page):
        st.session_state["current_page"] = target_page

    pagina = st.radio(
        "**Navegación**",
        OPCIONES_PAGINAS,
        index=OPCIONES_PAGINAS.index(st.session_state["current_page"]) if st.session_state["current_page"] in OPCIONES_PAGINAS else 0,
        key="nav_radio",
        label_visibility="collapsed",
    )
    if pagina != st.session_state["current_page"]:
        st.session_state["current_page"] = pagina

    st.markdown("""
    <div class="glass-card" style="padding: 16px;">
        <h3 style="font-size: 0.85rem; margin-bottom: 8px;">⚙️ Especificaciones Técnicas</h3>
        <p style="font-size: 0.75rem; margin: 3px 0; color: #cbd5e1;">
            <strong>Muestras:</strong> 10,200 registros<br>
            <strong>Target:</strong> Sleep_Quality (4 clases)<br>
            <strong>Modelos:</strong> LogReg, Random Forest<br>
            <strong>Validación:</strong> Stratified 5-Fold CV
        </p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  1.  INICIO
# ═══════════════════════════════════════════════════════════════

if pagina == "🏠 Inicio":
    st.markdown("""
    <div class="hero-title">
        <h1>☕ Café, Sueño & Machine Learning</h1>
        <p>Pipeline completo de aprendizaje supervisado: desde la limpieza de datos
        hasta la evaluación de modelos predictivos</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # KPIs rápidos
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Registros originales", f"{log_limpieza['registros_originales']:,}")
    col2.metric("✅ Registros limpios", f"{log_limpieza['registros_limpios']:,}")
    col3.metric("🗑️ Filas eliminadas", f"{log_limpieza['registros_originales'] - log_limpieza['registros_limpios']:,}")
    col4.metric("📐 Variables", f"{df_original.shape[1]}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Descripción del proyecto
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="glass-card">
            <h3>🎯 Objetivo del Proyecto</h3>
            <p>Aplicar un <strong>pipeline completo de Machine Learning supervisado</strong>
            para predecir la <strong>calidad del sueño</strong> (Sleep_Quality) a partir
            de variables como consumo de café, edad, IMC, actividad física, ocupación,
            nivel de estrés y más.</p>
            <p>Se implementan dos modelos de clasificación multiclase:
            <strong>Regresión Logística</strong> y <strong>Random Forest</strong>,
            comparando su rendimiento con validación cruzada estratificada.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3>📂 Dataset: Coffee Health</h3>
            <p>El conjunto de datos <strong>coffee_health_dirty.csv</strong> contiene
            información de <strong>10,200 personas</strong> sobre hábitos de consumo
            de café y métricas de salud.</p>
            <p><strong>Variable objetivo:</strong>
            <span class="badge badge-purple">Sleep_Quality</span> con 4 clases:</p>
            <p>
                <span class="badge badge-green">Excellent</span>
                <span class="badge badge-blue">Good</span>
                <span class="badge badge-amber">Fair</span>
                <span class="badge badge-purple" style="background:rgba(239,68,68,0.22);color:#fca5a5;">Poor</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Pipeline visual con navegación interactiva
    st.markdown("""
    <div class="glass-card">
        <h3>🔄 Pipeline del Proyecto</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0;">Haz clic en cualquiera de las fases para ir directamente a su sección:</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    pasos = [
        ("🧹", "Limpieza", "🧹 Limpieza de Datos", "Duplicados, outliers y estandarización"),
        ("📊", "EDA", "📊 EDA — Análisis Exploratorio", "Distribuciones, correlaciones y gráficos"),
        ("🤖", "Modelos", "🤖 Modelos de ML", "Entrenamiento de LogReg y Random Forest"),
        ("📈", "Evaluación", "📈 Evaluación & Comparación", "Métricas, ROC y matrices de confusión"),
        ("🔮", "Predicción", "🔮 Predicción Interactiva", "Simulador en tiempo real"),
    ]
    for col, (emoji, titulo, target_page, desc) in zip(cols, pasos):
        with col:
            st.markdown(f"""
            <div style="text-align:center; background: rgba(139,92,246,0.08);
                        border: 1px solid rgba(139,92,246,0.2);
                        border-radius: 14px 14px 0 0; padding: 16px 10px 10px 10px;">
                <span style="font-size: 2rem;">{emoji}</span>
                <p style="color: #c4b5fd; font-weight: 700; font-size: 0.88rem; margin: 6px 0 2px 0;">{titulo}</p>
                <p style="color: #94a3b8; font-size: 0.72rem; margin-bottom: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Ir a {titulo} →", key=f"btn_nav_{titulo}", use_container_width=True):
                st.session_state["current_page"] = target_page
                st.rerun()

    # Variables del dataset
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-bottom: 20px;">
        <h3 style="font-size: 1.35rem; font-weight: 700; color: #c4b5fd; margin-bottom: 8px;">
            📋 Diccionario y Variables del Conjunto de Datos <span style="font-size: 0.95rem; font-weight: 400; color: #a5b4fc;">(14 columnas)</span>
        </h3>
        <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0;">
            A continuación se presenta el desglose estructurado de todas las características disponibles en el dataset, clasificadas por su tipo de dato, presencia de nulos y rol en el modelado.
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🔢 Numéricas Continuas", "6", help="Age, Coffee_Intake, Caffeine_mg, Sleep_Hours, BMI, Physical_Activity_Hours")
    m2.metric("🔤 Categóricas", "4", help="Gender, Country, Stress_Level, Occupation")
    m3.metric("🔘 Binarias", "2", help="Smoking, Alcohol_Consumption")
    m4.metric("🎯 Variable Objetivo", "Sleep_Quality", help="4 categorías ordinales")

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    dict_vars = [
        {"Variable": "Age", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Edad del participante en años", "Nulos (%)": f"{(df_original['Age'].isna().mean()*100):.1f}%"},
        {"Variable": "Gender", "Tipo": "Categórica", "Rol": "Predictor", "Descripción": "Género (Male, Female, Non-binary, Other)", "Nulos (%)": f"{(df_original['Gender'].isna().mean()*100):.1f}%"},
        {"Variable": "Country", "Tipo": "Categórica", "Rol": "Predictor", "Descripción": "País de residencia del individuo", "Nulos (%)": f"{(df_original['Country'].isna().mean()*100):.1f}%"},
        {"Variable": "Coffee_Intake", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Número de tazas de café consumidas al día", "Nulos (%)": f"{(df_original['Coffee_Intake'].isna().mean()*100):.1f}%"},
        {"Variable": "Caffeine_mg", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Miligramos estimados de cafeína diaria", "Nulos (%)": f"{(df_original['Caffeine_mg'].isna().mean()*100):.1f}%"},
        {"Variable": "Sleep_Hours", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Promedio de horas de sueño diarias", "Nulos (%)": f"{(df_original['Sleep_Hours'].isna().mean()*100):.1f}%"},
        {"Variable": "BMI", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Índice de Masa Corporal (IMC)", "Nulos (%)": f"{(df_original['BMI'].isna().mean()*100):.1f}%"},
        {"Variable": "Heart_Rate", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Frecuencia cardíaca en reposo (bpm)", "Nulos (%)": f"{(df_original['Heart_Rate'].isna().mean()*100):.1f}%"},
        {"Variable": "Stress_Level", "Tipo": "Categórica", "Rol": "Predictor", "Descripción": "Nivel de estrés percibido (Low, Medium, High)", "Nulos (%)": f"{(df_original['Stress_Level'].isna().mean()*100):.1f}%"},
        {"Variable": "Physical_Activity_Hours", "Tipo": "Numérica", "Rol": "Predictor", "Descripción": "Horas semanales dedicadas a ejercicio físico", "Nulos (%)": f"{(df_original['Physical_Activity_Hours'].isna().mean()*100):.1f}%"},
        {"Variable": "Occupation", "Tipo": "Categórica", "Rol": "Predictor", "Descripción": "Área profesional / ocupación laboral", "Nulos (%)": f"{(df_original['Occupation'].isna().mean()*100):.1f}%"},
        {"Variable": "Smoking", "Tipo": "Binaria", "Rol": "Predictor", "Descripción": "Hábito de fumar (1: Sí, 0: No)", "Nulos (%)": f"{(df_original['Smoking'].isna().mean()*100):.1f}%"},
        {"Variable": "Alcohol_Consumption", "Tipo": "Binaria", "Rol": "Predictor", "Descripción": "Consumo de alcohol habitual (1: Sí, 0: No)", "Nulos (%)": f"{(df_original['Alcohol_Consumption'].isna().mean()*100):.1f}%"},
        {"Variable": "Sleep_Quality", "Tipo": "Categórica", "Rol": "Target 🎯", "Descripción": "Calidad del sueño (Poor, Fair, Good, Excellent)", "Nulos (%)": f"{(df_original['Sleep_Quality'].isna().mean()*100):.1f}%"},
    ]
    vars_df = pd.DataFrame(dict_vars)

    st.dataframe(
        vars_df,
        column_config={
            "Variable": st.column_config.TextColumn("📌 Variable", width="medium"),
            "Tipo": st.column_config.TextColumn("🏷️ Tipo", width="small"),
            "Rol": st.column_config.TextColumn("⚙️ Rol", width="small"),
            "Descripción": st.column_config.TextColumn("📝 Descripción Detallada", width="large"),
            "Nulos (%)": st.column_config.TextColumn("⚠️ Nulos", width="small"),
        },
        hide_index=True,
        use_container_width=True,
        height=490,
    )

    mostrar_footer()


# ═══════════════════════════════════════════════════════════════
#  2.  LIMPIEZA DE DATOS
# ═══════════════════════════════════════════════════════════════

elif pagina == "🧹 Limpieza de Datos":
    st.markdown("""
    <div class="hero-title">
        <h1>🧹 Limpieza de Datos</h1>
        <p>Proceso completo de depuración: duplicados, outliers, estandarización y valores inválidos</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Resumen de limpieza ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Registros originales", f"{log_limpieza['registros_originales']:,}")
    col2.metric("🗑️ Duplicados eliminados", f"{log_limpieza['duplicados_eliminados']:,}")
    col3.metric("⚠️ Sin etiqueta", f"{log_limpieza['filas_sin_etiqueta']:,}")
    col4.metric("✅ Registros finales", f"{log_limpieza['registros_limpios']:,}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Tabs de limpieza ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔤 Estandarización de texto",
        "📐 Valores fuera de rango",
        "📊 Comparación Nulos",
        "📝 Datos limpios"
    ])

    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h3>🔤 Estandarización de variables categóricas</h3>
            <p>Se normalizaron espacios y mayúsculas en las columnas <code>Sleep_Quality</code>,
            <code>Gender</code> y <code>Stress_Level</code> para unificar categorías inconsistentes.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Antes de limpiar")
            sq_antes = df_original["Sleep_Quality"].value_counts(dropna=False)
            fig_antes = px.bar(
                x=sq_antes.index.astype(str), y=sq_antes.values,
                labels={"x": "Sleep_Quality", "y": "Conteo"},
                title="Sleep_Quality — valores originales",
                color_discrete_sequence=["#f97316"]
            )
            fig_antes.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1"
            )
            st.plotly_chart(fig_antes, use_container_width=True)

        with c2:
            st.markdown("#### Después de limpiar")
            sq_despues = df_limpio["Sleep_Quality"].value_counts(dropna=False)
            fig_despues = px.bar(
                x=sq_despues.index.astype(str), y=sq_despues.values,
                labels={"x": "Sleep_Quality", "y": "Conteo"},
                title="Sleep_Quality — valores estandarizados",
                color_discrete_sequence=["#22c55e"]
            )
            fig_despues.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1"
            )
            st.plotly_chart(fig_despues, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h3>📐 Detección de valores fuera de rango</h3>
            <p>Se marcaron como <code>NaN</code> los valores numéricos que excedían rangos razonables.</p>
        </div>
        """, unsafe_allow_html=True)

        outlier_data = pd.DataFrame({
            "Variable": ["Age", "BMI", "Sleep_Hours", "Coffee_Intake", "Smoking"],
            "Criterio": [
                "< 10 o > 120 años",
                "> 60",
                "< 2 o > 20 horas",
                "< 0 (negativo)",
                "≠ 0 y ≠ 1"
            ],
            "Casos detectados": [
                log_limpieza["edad_fuera_rango"],
                log_limpieza["bmi_extremo"],
                log_limpieza["sueno_fuera_rango"],
                log_limpieza["cafe_negativo"],
                log_limpieza["smoking_invalido"],
            ],
            "Acción": [
                "Convertido a NaN",
                "Convertido a NaN",
                "Convertido a NaN",
                "Convertido a NaN",
                "Convertido a NaN",
            ]
        })

        st.dataframe(outlier_data, use_container_width=True, hide_index=True)

        # Gráfico de outliers
        fig_out = px.bar(
            outlier_data, x="Variable", y="Casos detectados",
            color="Variable",
            color_discrete_sequence=["#ef4444", "#f97316", "#eab308", "#3b82f6", "#8b5cf6"],
            title="Valores fuera de rango detectados por variable"
        )
        fig_out.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            showlegend=False
        )
        st.plotly_chart(fig_out, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Valores nulos: antes vs. después</h3>
        </div>
        """, unsafe_allow_html=True)

        nulos_antes = pd.Series(log_limpieza["nulos_originales"])
        nulos_despues = pd.Series(log_limpieza["nulos_finales"])

        comparacion = pd.DataFrame({
            "Variable": nulos_antes.index,
            "Nulos originales": nulos_antes.values,
            "Nulos tras limpieza": nulos_despues.reindex(nulos_antes.index, fill_value=0).values,
        })

        fig_nulos = go.Figure()
        fig_nulos.add_bar(x=comparacion["Variable"], y=comparacion["Nulos originales"],
                          name="Originales", marker_color="#ef4444")
        fig_nulos.add_bar(x=comparacion["Variable"], y=comparacion["Nulos tras limpieza"],
                          name="Tras limpieza", marker_color="#22c55e")
        fig_nulos.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            title="Comparación de valores nulos"
        )
        st.plotly_chart(fig_nulos, use_container_width=True)

    with tab4:
        st.markdown("#### Vista previa de los datos limpios")
        st.dataframe(df_limpio.head(20), use_container_width=True, hide_index=True)

        st.markdown("#### Resumen estadístico")
        st.dataframe(df_limpio.describe().round(2), use_container_width=True)

    # ── Reporte final ──
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card">
        <h3>📝 Reporte de Limpieza y Preparación</h3>
        <p>El conjunto original contenía <strong>{log_limpieza['registros_originales']:,}</strong> registros.
        Después de la depuración quedaron <strong>{log_limpieza['registros_limpios']:,}</strong>,
        con un total de <strong>{log_limpieza['registros_originales'] - log_limpieza['registros_limpios']:,}</strong>
        filas eliminadas y cero duplicados restantes.</p>
        <p>Se eliminaron <strong>{log_limpieza['filas_sin_etiqueta']}</strong> filas sin la etiqueta
        <code>Sleep_Quality</code>, dado que esta representa la respuesta que el modelo debe aprender.</p>
        <p>Se estandarizaron las variables categóricas y se marcaron como faltantes los valores numéricos inválidos.
        En <code>Smoking</code> se identificaron <strong>{log_limpieza['smoking_invalido']}</strong> valores
        distintos de 0 y 1, que se convirtieron en datos faltantes.</p>
        <p>La imputación se incorporó a los pipelines: <strong>mediana</strong> para las variables numéricas continuas
        y <strong>moda</strong> para las categóricas y binarias.</p>
    </div>
    """, unsafe_allow_html=True)

    mostrar_footer()


# ═══════════════════════════════════════════════════════════════
#  3.  EDA — ANÁLISIS EXPLORATORIO
# ═══════════════════════════════════════════════════════════════

elif pagina == "📊 EDA — Análisis Exploratorio":
    st.markdown("""
    <div class="hero-title">
        <h1>📊 Análisis Exploratorio de Datos</h1>
        <p>Distribuciones, correlaciones y relaciones clave entre variables</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Variable Objetivo",
        "📈 Distribuciones",
        "📦 Boxplots & Outliers",
        "🔥 Correlaciones",
        "☕ Café vs Sueño"
    ])

    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h3>🎯 Distribución de la variable objetivo: Sleep_Quality</h3>
        </div>
        """, unsafe_allow_html=True)

        orden = ["Excellent", "Good", "Fair", "Poor"]
        colores_calidad = {"Excellent": "#22c55e", "Good": "#3b82f6", "Fair": "#eab308", "Poor": "#ef4444"}
        conteos = df_limpio["Sleep_Quality"].value_counts().reindex(orden)

        c1, c2 = st.columns(2)

        with c1:
            fig_bar = px.bar(
                x=conteos.index, y=conteos.values,
                color=conteos.index,
                color_discrete_map=colores_calidad,
                labels={"x": "Calidad del sueño", "y": "Cantidad de personas"},
                title="Distribución de la calidad del sueño",
                text=conteos.values
            )
            fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False,
                yaxis_range=[0, conteos.max() * 1.18]
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            fig_pie = px.pie(
                values=conteos.values,
                names=conteos.index,
                color=conteos.index,
                color_discrete_map=colores_calidad,
                title="Proporción de cada clase",
                hole=0.45
            )
            fig_pie.update_traces(textinfo="label+percent", textfont_size=13)
            fig_pie.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Métricas por clase
        cols = st.columns(4)
        for col, clase in zip(cols, orden):
            pct = conteos[clase] / conteos.sum() * 100
            col.metric(f"{'🟢' if clase=='Excellent' else '🔵' if clase=='Good' else '🟡' if clase=='Fair' else '🔴'} {clase}", f"{conteos[clase]:,}", f"{pct:.1f}%")

    with tab2:
        st.markdown("""
        <div class="glass-card">
            <h3>📈 Distribución de variables numéricas</h3>
            <p>Histogramas de las principales variables numéricas del dataset limpio.</p>
        </div>
        """, unsafe_allow_html=True)

        columnas_hist = ["Age", "Coffee_Intake", "Caffeine_mg", "Sleep_Hours",
                         "BMI", "Heart_Rate", "Physical_Activity_Hours"]

        col_sel = st.multiselect(
            "Selecciona las variables a visualizar:",
            columnas_hist,
            default=columnas_hist,
            key="hist_cols"
        )

        if col_sel:
            n_cols = min(3, len(col_sel))
            n_rows = (len(col_sel) + n_cols - 1) // n_cols

            fig_hist = make_subplots(rows=n_rows, cols=n_cols,
                                     subplot_titles=col_sel,
                                     vertical_spacing=0.12,
                                     horizontal_spacing=0.08)

            colors = ["#8b5cf6", "#3b82f6", "#06b6d4", "#22c55e", "#eab308", "#f97316", "#ef4444"]

            for i, col_name in enumerate(col_sel):
                r = i // n_cols + 1
                c = i % n_cols + 1
                data = df_limpio[col_name].dropna()
                fig_hist.add_trace(
                    go.Histogram(x=data, nbinsx=30, marker_color=colors[i % len(colors)],
                                 opacity=0.85, name=col_name, showlegend=False),
                    row=r, col=c
                )

            fig_hist.update_layout(
                height=350 * n_rows,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                title_text="Distribución de variables numéricas"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="glass-card">
            <h3>📦 Boxplots — Detección de valores atípicos</h3>
        </div>
        """, unsafe_allow_html=True)

        columnas_box = ["Age", "Coffee_Intake", "Caffeine_mg", "Sleep_Hours",
                        "BMI", "Heart_Rate", "Physical_Activity_Hours"]

        fig_box = make_subplots(rows=2, cols=4,
                                subplot_titles=columnas_box + [""],
                                vertical_spacing=0.15)

        colors_box = ["#8b5cf6", "#3b82f6", "#06b6d4", "#22c55e", "#eab308", "#f97316", "#ef4444"]

        for i, col_name in enumerate(columnas_box):
            r = i // 4 + 1
            c = i % 4 + 1
            fig_box.add_trace(
                go.Box(y=df_limpio[col_name].dropna(), name=col_name,
                       marker_color=colors_box[i], showlegend=False),
                row=r, col=c
            )

        fig_box.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            title_text="Valores atípicos en las variables numéricas"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.markdown("""
        <div class="glass-card">
            <h3>🔥 Mapa de correlaciones</h3>
            <p>Correlación de Pearson entre variables numéricas continuas.</p>
        </div>
        """, unsafe_allow_html=True)

        columnas_corr = ["Age", "Coffee_Intake", "Caffeine_mg", "Sleep_Hours",
                         "BMI", "Heart_Rate", "Physical_Activity_Hours"]

        corr = df_limpio[columnas_corr].corr()

        fig_hm = px.imshow(
            corr.round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Mapa de calor — Correlaciones",
            aspect="auto"
        )
        fig_hm.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Top correlaciones
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_pairs = corr.where(mask).stack().reset_index()
        corr_pairs.columns = ["Variable 1", "Variable 2", "Correlación"]
        corr_pairs["Abs"] = corr_pairs["Correlación"].abs()
        corr_pairs = corr_pairs.sort_values("Abs", ascending=False).drop(columns="Abs")

        st.markdown("#### Top correlaciones")
        st.dataframe(corr_pairs.head(10).reset_index(drop=True), use_container_width=True, hide_index=True)

    with tab5:
        st.markdown("""
        <div class="glass-card">
            <h3>☕ Relación: Café y Horas de sueño vs Calidad del sueño</h3>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            fig_cafe = px.box(
                df_limpio, x="Sleep_Quality", y="Coffee_Intake",
                category_orders={"Sleep_Quality": ["Excellent", "Good", "Fair", "Poor"]},
                color="Sleep_Quality",
                color_discrete_map=colores_calidad,
                title="Consumo de café según calidad del sueño",
                labels={"Coffee_Intake": "Consumo de café", "Sleep_Quality": "Calidad del sueño"}
            )
            fig_cafe.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False
            )
            st.plotly_chart(fig_cafe, use_container_width=True)

        with c2:
            fig_sleep = px.box(
                df_limpio, x="Sleep_Quality", y="Sleep_Hours",
                category_orders={"Sleep_Quality": ["Excellent", "Good", "Fair", "Poor"]},
                color="Sleep_Quality",
                color_discrete_map=colores_calidad,
                title="Horas de sueño según calidad del sueño",
                labels={"Sleep_Hours": "Horas de sueño", "Sleep_Quality": "Calidad del sueño"}
            )
            fig_sleep.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False
            )
            st.plotly_chart(fig_sleep, use_container_width=True)

        # Scatter interactivo
        st.markdown("#### 🔬 Scatter Plot interactivo")
        var_x = st.selectbox("Eje X", columnas_hist, index=1, key="scatter_x")
        var_y = st.selectbox("Eje Y", columnas_hist, index=3, key="scatter_y")

        fig_scatter = px.scatter(
            df_limpio, x=var_x, y=var_y,
            color="Sleep_Quality",
            color_discrete_map=colores_calidad,
            category_orders={"Sleep_Quality": ["Excellent", "Good", "Fair", "Poor"]},
            opacity=0.6,
            title=f"{var_x} vs {var_y} por Calidad del sueño"
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            height=500
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    mostrar_footer()


# ═══════════════════════════════════════════════════════════════
#  4.  MODELOS DE ML
# ═══════════════════════════════════════════════════════════════

elif pagina == "🤖 Modelos de ML":
    st.markdown("""
    <div class="hero-title">
        <h1>🤖 Modelos de Machine Learning</h1>
        <p>Entrenamiento con GridSearchCV y validación cruzada estratificada (5 folds)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("🔄 Entrenando modelos... (esto puede tardar un momento la primera vez)"):
        modelos_info, split_info, feature_names, importances = entrenar_modelos(df_limpio)

    # Split info
    col1, col2, col3 = st.columns(3)
    col1.metric("🏋️ Entrenamiento", f"{split_info['train_size']:,} registros")
    col2.metric("🧪 Prueba", f"{split_info['test_size']:,} registros")
    col3.metric("📏 Proporción", "80% / 20%")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Preprocesamiento", "📊 Regresión Logística", "🌲 Random Forest"])

    with tab1:
        st.markdown("""
        <div class="glass-card">
            <h3>⚙️ Pipeline de Preprocesamiento</h3>
            <p>El preprocesamiento se integra directamente en los pipelines de scikit-learn,
            garantizando que no haya fuga de datos entre entrenamiento y prueba.</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <h3>🔢 Variables numéricas ({len(split_info['numericas'])})</h3>
                <p><strong>Pipeline:</strong> SimpleImputer(median) → StandardScaler</p>
                <p>{''.join(f'<span class="badge badge-blue">{v}</span>' for v in split_info['numericas'])}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="glass-card">
                <h3>🔤 Variables categóricas ({len(split_info['categoricas'])})</h3>
                <p><strong>Pipeline:</strong> SimpleImputer(most_frequent) → OneHotEncoder</p>
                <p>{''.join(f'<span class="badge badge-purple">{v}</span>' for v in split_info['categoricas'])}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="glass-card">
                <h3>🔘 Variables binarias ({len(split_info['binarias'])})</h3>
                <p><strong>Pipeline:</strong> SimpleImputer(most_frequent)</p>
                <p>{''.join(f'<span class="badge badge-green">{v}</span>' for v in split_info['binarias'])}</p>
            </div>
            """, unsafe_allow_html=True)

        # Imputación info
        st.markdown("#### 📊 Valores faltantes por conjunto")

        impute_data = []
        for col_name in split_info["X_train_nulls"]:
            train_n = split_info["X_train_nulls"].get(col_name, 0)
            test_n = split_info["X_test_nulls"].get(col_name, 0)
            if train_n > 0 or test_n > 0:
                strat = "Mediana" if col_name in split_info["numericas"] else "Moda"
                impute_data.append({
                    "Variable": col_name,
                    "Faltantes (Train)": train_n,
                    "Faltantes (Test)": test_n,
                    "Estrategia": strat
                })

        if impute_data:
            st.dataframe(pd.DataFrame(impute_data), use_container_width=True, hide_index=True)

    with tab2:
        info_lr = modelos_info["Regresión Logística"]

        st.markdown(f"""
        <div class="glass-card">
            <h3>📊 Regresión Logística</h3>
            <p><strong>Mejor parámetro:</strong> C = {info_lr['best_params']['modelo__C']}</p>
            <p><strong>F1 macro en validación cruzada:</strong> {info_lr['cv_score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{info_lr['metricas']['Accuracy']:.4f}")
        col2.metric("Precision macro", f"{info_lr['metricas']['Precision macro']:.4f}")
        col3.metric("Recall macro", f"{info_lr['metricas']['Recall macro']:.4f}")
        col4.metric("F1 macro", f"{info_lr['metricas']['F1 macro']:.4f}")

        # Reporte por clase
        st.markdown("#### Reporte por clase")
        report_df = pd.DataFrame(info_lr["report"]).T
        report_df = report_df.loc[info_lr["clases"]].round(4)
        st.dataframe(report_df, use_container_width=True)

    with tab3:
        info_rf = modelos_info["Random Forest"]

        st.markdown(f"""
        <div class="glass-card">
            <h3>🌲 Random Forest</h3>
            <p><strong>Mejores parámetros:</strong> max_depth = {info_rf['best_params']['modelo__max_depth']},
            min_samples_leaf = {info_rf['best_params']['modelo__min_samples_leaf']}</p>
            <p><strong>F1 macro en validación cruzada:</strong> {info_rf['cv_score']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{info_rf['metricas']['Accuracy']:.4f}")
        col2.metric("Precision macro", f"{info_rf['metricas']['Precision macro']:.4f}")
        col3.metric("Recall macro", f"{info_rf['metricas']['Recall macro']:.4f}")
        col4.metric("F1 macro", f"{info_rf['metricas']['F1 macro']:.4f}")

        # Reporte por clase
        st.markdown("#### Reporte por clase")
        report_df_rf = pd.DataFrame(info_rf["report"]).T
        report_df_rf = report_df_rf.loc[info_rf["clases"]].round(4)
        st.dataframe(report_df_rf, use_container_width=True)

        # Feature importances
        st.markdown("#### 🏆 Importancia de características (Top 15)")
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importancia": importances
        }).sort_values("Importancia", ascending=True).tail(15)

        fig_imp = px.bar(
            feat_imp, x="Importancia", y="Feature",
            orientation="h",
            color="Importancia",
            color_continuous_scale="Viridis",
            title="Top 15 — Importancia de características (Random Forest)"
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            height=500,
            yaxis_title=""
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    mostrar_footer()


# ═══════════════════════════════════════════════════════════════
#  5.  EVALUACIÓN & COMPARACIÓN
# ═══════════════════════════════════════════════════════════════

elif pagina == "📈 Evaluación & Comparación":
    st.markdown("""
    <div class="hero-title">
        <h1>📈 Evaluación & Comparación de Modelos</h1>
        <p>Matrices de confusión, métricas comparativas y análisis por clase</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("🔄 Cargando resultados de modelos..."):
        modelos_info, split_info, feature_names, importances = entrenar_modelos(df_limpio)

    clases = ["Excellent", "Good", "Fair", "Poor"]

    # ── Matrices de confusión ──
    st.markdown("### 🔲 Matrices de Confusión")

    c1, c2 = st.columns(2)

    for col, (nombre, info) in zip([c1, c2], modelos_info.items()):
        with col:
            cm = info["confusion_matrix"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Clase predicha", y="Clase real", color="Conteo"),
                x=clases, y=clases,
                text_auto=True,
                color_continuous_scale="Blues" if "Logística" in nombre else "Greens",
                title=f"Matriz de Confusión — {nombre}"
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                height=450
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Comparación de métricas ──
    st.markdown("### 📊 Comparación de métricas principales")

    metricas_comp = []
    for nombre, info in modelos_info.items():
        for metrica, valor in info["metricas"].items():
            metricas_comp.append({"Modelo": nombre, "Métrica": metrica, "Valor": valor})

    df_comp = pd.DataFrame(metricas_comp)

    # Filtrar solo macro
    metricas_principales = ["Accuracy", "Precision macro", "Recall macro", "F1 macro"]
    df_main = df_comp[df_comp["Métrica"].isin(metricas_principales)]

    fig_comp = px.bar(
        df_main, x="Métrica", y="Valor", color="Modelo",
        barmode="group",
        color_discrete_map={
            "Regresión Logística": "#8b5cf6",
            "Random Forest": "#22c55e"
        },
        title="Comparación de modelos en el conjunto de prueba",
        text="Valor"
    )
    fig_comp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_comp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        yaxis_range=[0, 1.1],
        height=500
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Radar chart ──
    st.markdown("### 🕸️ Radar de métricas")

    categories = metricas_principales + [metricas_principales[0]]  # cerrar el polígono

    fig_radar = go.Figure()
    colors_radar = {"Regresión Logística": "#8b5cf6", "Random Forest": "#22c55e"}

    for nombre, info in modelos_info.items():
        values = [info["metricas"][m] for m in metricas_principales]
        values.append(values[0])

        fig_radar.add_trace(go.Scatterpolar(
            r=values, theta=categories,
            fill="toself", name=nombre,
            line_color=colors_radar[nombre],
            opacity=0.7
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(139,92,246,0.15)"),
            angularaxis=dict(gridcolor="rgba(139,92,246,0.15)")
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tabla resumen ──
    st.markdown("### 📋 Tabla resumen completa")

    resumen = []
    for nombre, info in modelos_info.items():
        row = {"Modelo": nombre}
        row.update(info["metricas"])
        row["F1 CV (validación)"] = info["cv_score"]
        resumen.append(row)

    df_resumen = pd.DataFrame(resumen).set_index("Modelo")
    st.dataframe(df_resumen.round(4).style.highlight_max(axis=0, color="rgba(34,197,94,0.25)"),
                 use_container_width=True)

    # Mejor modelo
    mejor = max(modelos_info.items(), key=lambda x: x[1]["metricas"]["F1 macro"])
    st.markdown(f"""
    <div class="glass-card" style="border-color: rgba(34,197,94,0.4);">
        <h3>🏆 Mejor modelo: {mejor[0]}</h3>
        <p>Con un <strong>F1 macro de {mejor[1]['metricas']['F1 macro']:.4f}</strong> en el conjunto de prueba
        y <strong>{mejor[1]['cv_score']:.4f}</strong> en validación cruzada,
        <strong>{mejor[0]}</strong> es el modelo con mejor rendimiento global.</p>
    </div>
    """, unsafe_allow_html=True)

    mostrar_footer()


# ═══════════════════════════════════════════════════════════════
#  6.  PREDICCIÓN INTERACTIVA
# ═══════════════════════════════════════════════════════════════

elif pagina == "🔮 Predicción Interactiva":
    st.markdown("""
    <div class="hero-title">
        <h1>🔮 Predicción Interactiva</h1>
        <p>Ingresa los datos de un paciente y obtén la predicción de calidad del sueño</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.spinner("🔄 Cargando modelos..."):
        modelos_info, split_info, feature_names, importances = entrenar_modelos(df_limpio)

    # Formulario de entrada
    st.markdown("""
    <div class="glass-card">
        <h3>📝 Datos del paciente</h3>
        <p>Completa la información para obtener una predicción de la calidad del sueño.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("🎂 Edad", 10, 100, 35)
        gender = st.selectbox("👤 Género", ["Male", "Female"])
        country = st.selectbox("🌍 País", sorted(df_limpio["Country"].dropna().unique()))
        occupation = st.selectbox("💼 Ocupación", sorted(df_limpio["Occupation"].dropna().unique()))

    with c2:
        coffee_intake = st.slider("☕ Consumo de café (tazas/día)", 0.0, 8.0, 2.5, 0.1)
        caffeine_mg = st.slider("💊 Cafeína (mg)", 0.0, 800.0, 238.0, 5.0)
        sleep_hours = st.slider("😴 Horas de sueño", 2.0, 14.0, 6.7, 0.1)
        bmi = st.slider("⚖️ IMC (BMI)", 15.0, 50.0, 24.0, 0.1)

    with c3:
        heart_rate = st.slider("❤️ Frecuencia cardíaca", 50, 110, 70)
        stress_level = st.selectbox("😰 Nivel de estrés", ["Low", "Medium", "High"])
        activity_hours = st.slider("🏃 Actividad física (hrs/semana)", 0.0, 15.0, 7.5, 0.1)
        smoking = st.selectbox("🚬 Fumador", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        alcohol = st.selectbox("🍷 Consumo de alcohol", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.button("🔮 **Predecir calidad del sueño**", use_container_width=True, type="primary"):
        # Crear DataFrame con los datos
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Country": country,
            "Coffee_Intake": coffee_intake,
            "Caffeine_mg": caffeine_mg,
            "Sleep_Hours": sleep_hours,
            "BMI": bmi,
            "Heart_Rate": heart_rate,
            "Stress_Level": stress_level,
            "Physical_Activity_Hours": activity_hours,
            "Occupation": occupation,
            "Smoking": smoking,
            "Alcohol_Consumption": alcohol,
        }])

        # Re-entrenar para obtener modelos completos
        from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.base import clone

        X = df_limpio.drop(columns=["Sleep_Quality"])
        y = df_limpio["Sleep_Quality"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        binarias = ["Smoking", "Alcohol_Consumption"]
        numericas = [c for c in X_train.select_dtypes(include="number").columns if c not in binarias]
        categoricas = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

        proceso_numerico = Pipeline([
            ("imputacion", SimpleImputer(strategy="median")),
            ("escalado", StandardScaler())
        ])
        proceso_categorico = Pipeline([
            ("imputacion", SimpleImputer(strategy="most_frequent")),
            ("codificacion", OneHotEncoder(handle_unknown="ignore"))
        ])
        preprocesador = ColumnTransformer([
            ("numericas", proceso_numerico, numericas),
            ("categoricas", proceso_categorico, categoricas),
            ("binarias", SimpleImputer(strategy="most_frequent"), binarias)
        ])

        # Usar los mejores params ya encontrados
        best_c = modelos_info["Regresión Logística"]["best_params"]["modelo__C"]
        best_depth = modelos_info["Random Forest"]["best_params"]["modelo__max_depth"]
        best_leaf = modelos_info["Random Forest"]["best_params"]["modelo__min_samples_leaf"]

        pipeline_lr = Pipeline([
            ("preprocesamiento", clone(preprocesador)),
            ("modelo", LogisticRegression(max_iter=2000, class_weight="balanced",
                                          random_state=42, C=best_c))
        ])
        pipeline_rf = Pipeline([
            ("preprocesamiento", clone(preprocesador)),
            ("modelo", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42,
                n_jobs=1, max_depth=best_depth, min_samples_leaf=best_leaf
            ))
        ])

        pipeline_lr.fit(X_train, y_train)
        pipeline_rf.fit(X_train, y_train)

        pred_lr = pipeline_lr.predict(input_data)[0]
        pred_rf = pipeline_rf.predict(input_data)[0]

        # Probabilidades
        proba_lr = pipeline_lr.predict_proba(input_data)[0]
        proba_rf = pipeline_rf.predict_proba(input_data)[0]
        clases_lr = pipeline_lr.classes_
        clases_rf = pipeline_rf.classes_

        st.markdown("### 🎯 Resultados de la predicción")

        c1, c2 = st.columns(2)

        emoji_map = {"Excellent": "🟢", "Good": "🔵", "Fair": "🟡", "Poor": "🔴"}
        color_map = {"Excellent": "#22c55e", "Good": "#3b82f6", "Fair": "#eab308", "Poor": "#ef4444"}

        with c1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-color: {color_map.get(pred_lr, '#8b5cf6')}40;">
                <h3>📊 Regresión Logística</h3>
                <p style="font-size: 3rem; margin: 10px 0;">{emoji_map.get(pred_lr, '⚪')}</p>
                <p style="font-size: 1.6rem; font-weight: 700; color: {color_map.get(pred_lr, '#c4b5fd')};">{pred_lr}</p>
            </div>
            """, unsafe_allow_html=True)

            fig_prob_lr = px.bar(
                x=clases_lr, y=proba_lr,
                color=clases_lr.tolist(),
                color_discrete_map=color_map,
                labels={"x": "Clase", "y": "Probabilidad"},
                title="Probabilidades — Reg. Logística"
            )
            fig_prob_lr.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                showlegend=False, yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_prob_lr, use_container_width=True)

        with c2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border-color: {color_map.get(pred_rf, '#8b5cf6')}40;">
                <h3>🌲 Random Forest</h3>
                <p style="font-size: 3rem; margin: 10px 0;">{emoji_map.get(pred_rf, '⚪')}</p>
                <p style="font-size: 1.6rem; font-weight: 700; color: {color_map.get(pred_rf, '#c4b5fd')};">{pred_rf}</p>
            </div>
            """, unsafe_allow_html=True)

            fig_prob_rf = px.bar(
                x=clases_rf, y=proba_rf,
                color=clases_rf.tolist(),
                color_discrete_map=color_map,
                labels={"x": "Clase", "y": "Probabilidad"},
                title="Probabilidades — Random Forest"
            )
            fig_prob_rf.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1",
                showlegend=False, yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_prob_rf, use_container_width=True)

    mostrar_footer()
