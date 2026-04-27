# 🛡️ Brand Risk & Propensity Radar

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📋 Descripción del Proyecto
Este proyecto es una herramienta analítica avanzada diseñada para predecir el **riesgo de boicot de marca** en el contexto sociopolítico de Estados Unidos. Utilizando datos de la encuesta **ANES 2024 (American National Election Studies)**, el sistema permite a las marcas y consultoras de comunicación simular escenarios y entender la propensión de diferentes segmentos de la población a realizar acciones de boicot o buycot basadas en su identidad política y valores sociales.

El núcleo del proyecto es un modelo de **Machine Learning (Random Forest)** optimizado para tratar datos desbalanceados, integrado en una interfaz interactiva que permite la toma de decisiones basada en datos.

## 🚀 Características Principales
- **Simulador de Escenarios en Tiempo Real:** Interfaz interactiva construida con Streamlit para ajustar variables demográficas y políticas.
- **Inteligencia Artificial Explicable (XAI):** Integración de valores **SHAP** para desglosar exactamente qué factores (interés político, polarización, agenda climática) están impulsando el riesgo en cada predicción.
- **Arquitectura de Pipeline Profesional:** Procesamiento de datos robusto que maneja la codificación y el escalado de forma automática.
- **Enfoque de Negocio:** Clasificación del riesgo en niveles (Bajo, Moderado, Alto) con recomendaciones estratégicas automáticas.

## 🧠 Metodología y Stack Técnico
- **Lenguaje:** Python 3.12
- **Modelo:** Random Forest Classifier (Optimizado con `imblearn` para balanceo de clases).
- **Explicabilidad:** SHAP (Kernel & Tree Explainers).
- **Despliegue:** Streamlit Community Cloud.
- **Librerías Clave:** `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `numpy`.

## 🛠️ Instalación y Uso Local

Si deseas ejecutar este proyecto localmente, sigue estos pasos:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/boicot-buycot-radar.git](https://github.com/TU_USUARIO/boicot-buycot-radar.git)
   cd boicot-buycot-radar
   
2. **Instalar dependencias:**

    Bash
    pip install -r requirements.txt

3. **Ejecutar la aplicación:**

     Bash
     streamlit run app.py
     
📊 Visualización de la Herramienta
<img width="1275" height="1115" alt="image" src="https://github.com/user-attachments/assets/a02e25fd-38b7-452a-8fb9-b5da6238f1be" />


📈 Impacto en el TFM
Este proyecto forma parte de mi Trabajo de Fin de Máster, demostrando la capacidad de transformar datos sociológicos complejos en herramientas de software funcionales y accionables para el mundo corporativo. Se pone especial énfasis en la transparencia de los algoritmos, permitiendo que los usuarios no técnicos confíen en las predicciones de la IA.

Creado por Fernando Martínez Buitrago - 2026
