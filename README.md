🏷️ Multi-Label Text Classifier
📖 Descripción
Sistema de clasificación de texto con múltiples etiquetas que utiliza Machine Learning para categorizar automáticamente textos según diferentes criterios. Incluye una API REST desarrollada con Flask y una interfaz web intuitiva.
🏗️ Estructura del Proyecto
📦 Proyecto
├── 📁 data/                          # Datos del proyecto
│   ├── challenge-data-18-ago.csv     # Dataset principal
│   └── prueba.xlsx                   # Datos de prueba
├── 📁 src/                           # Código fuente
│   ├── 📁 __pycache__                # Cache de Python
├── 📁 front/                         # Frontend
│   ├── 🌐 index.html                # Interfaz web
│   └── 🎨 styles.css                # Estilos CSS
├── 📁 models/                        # Modelo entrenado
│   ├── 🏷️ text_classifier_labels.pkl # Etiquetas del modelo
│   ├── 🧠 text_classifier_mlb.pkl    # MultiLabelBinarizer
│   ├── 🤖 text_classifier_model.h5   # Modelo de TensorFlow/Keras
│   ├── 📝 text_classifier_vectorizer.pkl # Vectorizador de texto
├── 📊 data_visualization.ipynb       # Análisis y visualización en jupyter para mas comodidad de visualización
├── 🌶️ flask_app.py                  # API REST con Flask
├── 🐍 MultiLabelTextClassifier.py   # Script principal del clasificador
├── 🚫 .gitignore                    # Archivos ignorados por Git
└── 📋 requirements.txt               # Dependencias del proyecto

🚀 Guía de Uso Paso a Paso
1️⃣ Preparación del Entorno
bash# Instalar dependencias
pip install -r requirements.txt
2️⃣ Entrenar el Modelo
bash# Ir al directorio src desde la raíz del proyecto
cd src

# Ejecutar el clasificador para entrenar el modelo
python MultiLabelTextClassifier.py | py MultiLabelTextClassifier.py
esto entrenara el modelo pero en este caso ya esta "se genraran tres imagenes de visualizacion se debe dar en "x" para continuar el proceso
3️⃣ Iniciar la API REST
bash# En la misma consola, ejecutar la aplicación Flask
python flask_app.py
4️⃣ Acceder a la Aplicación Web

Abrir tu navegador web
Ir a la URL que aparece en la consola (por defecto: http://127.0.0.1:5000)
en caso de tener visual studio code intalar dependencia Go Live y dar click e ir a carpeta de front
En la interfaz web:

Título: Ingresa el título del texto a clasificar
Abstract: Ingresa el resumen o contenido del texto
Hacer clic en "Clasificar"



📚 Dependencias Necesarias
📋 requirements.txt
txt# Frameworks Web
Flask==2.3.3
Flask-CORS==4.0.0

# Machine Learning y Data Science
scikit-learn==1.3.0
tensorflow==2.13.0
pandas==2.0.3
numpy==1.24.3

# Procesamiento de Texto
nltk==3.8.1

# Visualización
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0


🎯 Personalización

Modelo: Los modelos entrenados se guardan en la carpeta models/
Datos: Coloca tus datasets en la carpeta data/
Frontend: Personaliza la interfaz editando front/index.html y front/styles.css

📊 Características del Sistema

✅ Clasificación Multi-etiqueta: Asigna múltiples categorías a un texto
✅ API REST: Interfaz programática para integración
✅ Interfaz Web: Frontend amigable para usuarios finales
✅ Visualizaciones: Gráficos de distribución de datos
✅ Modelos Persistentes: Guarda y carga modelos entrenados

🔍 Endpoints de la API
POST /predict
json{
  "title": "Título del texto",
  "abstract": "Contenido o resumen del texto"
}
Respuesta:
json{
  "predictions": ["etiqueta1", "etiqueta2", "etiqueta3"],
  "confidence_scores": [0.85, 0.72, 0.68]
}
🛠️ Solución de Problemas
Error: "No module named..."
bashpip install --upgrade pip
pip install -r requirements.txt
Puerto ocupado
bash# Cambiar puerto en flask_app.py
app.run(host='0.0.0.0', port=5001, debug=True)
📈 Próximas Mejoras

 Soporte para más formatos de datos
 Métricas de evaluación en tiempo real
 Interfaz de administración
 Exportación de resultados


💡 ¿Necesitas ayuda?
Si encuentras algún problema o tienes sugerencias, no dudes en crear un issue en el repositorio.
¡Feliz clasificación! 🎉