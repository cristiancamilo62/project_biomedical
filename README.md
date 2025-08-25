# 🏷️ Multi-Label Text Classifier

## 📖 Descripción
Sistema de clasificación de texto con múltiples etiquetas que utiliza **Machine Learning** para categorizar automáticamente textos según diferentes criterios.  
Incluye una **API REST con Flask** y una **interfaz web** intuitiva.

---

## 🏗️ Estructura del Proyecto

<img width="963" height="390" alt="image" src="https://github.com/user-attachments/assets/7ea1ecac-3c5a-4327-8755-8cc79e5e5d98" />


yaml
Copiar
Editar

---

## 🚀 Guía de Uso Paso a Paso

### 1️⃣ Preparación del entorno
Instalar dependencias:
```bash
pip install -r requirements.txt
2️⃣ Entrenar el modelo
Ir a la carpeta src:

bash
Copiar
Editar
cd src
Ejecutar el clasificador:

bash
Copiar
Editar
python MultiLabelTextClassifier.py
Esto entrenará el modelo (ya está entrenado, pero generará gráficas que debes cerrar con X para continuar).

3️⃣ Iniciar la API REST
bash
Copiar
Editar
python flask_app.py
4️⃣ Acceder a la aplicación web
Abrir navegador en: http://127.0.0.1:5000

O en VS Code instalar Go Live e ir a la carpeta front/

En la interfaz web:
<img width="1878" height="894" alt="image" src="https://github.com/user-attachments/assets/3a7522a3-7f9a-4cac-9bec-7c57116df2e4" />



Título: Ingresa el título del texto a clasificar

Abstract: Ingresa el resumen o contenido

Clic en Clasificar

![image](https://github.com/user-attachments/assets/cb9665e4-bb25-4b58-b253-f1ae070ec44a)



📚 Dependencias Necesarias
📋 requirements.txt

txt
Copiar
Editar
# Frameworks Web
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
📊 Características del Sistema
✅ Clasificación Multi-etiqueta
✅ API REST para integración
✅ Interfaz Web intuitiva
✅ Visualizaciones de datos
✅ Modelos persistentes

🔍 Endpoints de la API
POST /predict
json
Copiar
Editar
{
  "title": "Título del texto",
  "abstract": "Contenido o resumen del texto"
}
Respuesta:

json
Copiar
Editar
{
  "predictions": ["etiqueta1", "etiqueta2"],
  "confidence_scores": [0.85, 0.72]
}
🛠️ Solución de Problemas
Error: No module named ...
👉 Ejecuta:

bash
Copiar
Editar
pip install --upgrade pip
pip install -r requirements.txt
Puerto ocupado
👉 Cambiar en flask_app.py:

python
Copiar
Editar
app.run(host="0.0.0.0", port=5001, debug=True)
📈 Próximas Mejoras
Soporte para más formatos de datos

Métricas de evaluación en tiempo real

Interfaz de administración

Exportación de resultados

💡 ¿Necesitas ayuda?
Si encuentras algún problema o sugerencia, crea un issue en el repositorio.
¡Feliz clasificación! 🎉
