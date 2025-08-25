from flask import Flask, request, jsonify
import json
from MultiLabelTextClassifier import MultiLabelTextClassifier
import os
import traceback
from flask_cors import CORS # Importa CORS

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Variable global para el clasificador
classifier = None

def initialize_classifier():
    """Inicializa el clasificador cargando el modelo preentrenado"""
    global classifier
    try:
        classifier = MultiLabelTextClassifier(model_dir='models')
        classifier.load_model()
        print("Clasificador inicializado exitosamente!")
        return True
    except Exception as e:
        print(f"Error al inicializar el clasificador: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API esté funcionando"""
    return jsonify({
        'status': 'OK',
        'message': 'API funcionando correctamente',
        'model_loaded': classifier is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal para hacer predicciones
    
    Expects JSON:
    {
        "title": "Título del artículo",
        "abstract": "Resumen del artículo",
        "threshold": 0.5  # Opcional, por defecto 0.5
    }
    
    Returns JSON:
    {
        "success": true,
        "data": {
            "etiquetas_predichas": ["etiqueta1", "etiqueta2"],
            "probabilidades": {
                "etiqueta1": 0.85,
                "etiqueta2": 0.73,
                ...
            },
            "umbral_usado": 0.5
        }
    }
    """
    global classifier
    
    if classifier is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no está cargado. Contacte al administrador.'
        }), 500
    
    try:
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se proporcionaron datos JSON'
            }), 400
        
        # Validar campos requeridos
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        threshold = data.get('threshold', 0.5)
        
        if not title and not abstract:
            return jsonify({
                'success': False,
                'error': 'Debe proporcionar al menos un título o resumen'
            }), 400
        
        # Validar threshold
        try:
            threshold = float(threshold)
            if not 0 <= threshold <= 1:
                return jsonify({
                    'success': False,
                    'error': 'El umbral debe estar entre 0 y 1'
                }), 400
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'error': 'El umbral debe ser un número válido'
            }), 400
        
        # Realizar predicción
        resultado = classifier.predict(title, abstract, threshold)
        
        return jsonify({
            'success': True,
            'data': resultado
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Endpoint para hacer predicciones en lote
    
    Expects JSON:
    {
        "articles": [
            {
                "title": "Título 1",
                "abstract": "Resumen 1"
            },
            {
                "title": "Título 2",
                "abstract": "Resumen 2"
            }
        ],
        "threshold": 0.5  # Opcional
    }
    
    Returns JSON:
    {
        "success": true,
        "data": [
            {
                "index": 0,
                "title": "Título 1",
                "prediction": {...}
            },
            ...
        ]
    }
    """
    global classifier
    
    if classifier is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no está cargado. Contacte al administrador.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'articles' not in data:
            return jsonify({
                'success': False,
                'error': 'Debe proporcionar una lista de artículos en el campo "articles"'
            }), 400
        
        articles = data['articles']
        threshold = data.get('threshold', 0.5)
        
        if not isinstance(articles, list):
            return jsonify({
                'success': False,
                'error': 'El campo "articles" debe ser una lista'
            }), 400
        
        if len(articles) > 100:  # Limitar número de artículos por seguridad
            return jsonify({
                'success': False,
                'error': 'Máximo 100 artículos por request'
            }), 400
        
        # Validar threshold
        try:
            threshold = float(threshold)
            if not 0 <= threshold <= 1:
                return jsonify({
                    'success': False,
                    'error': 'El umbral debe estar entre 0 y 1'
                }), 400
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'error': 'El umbral debe ser un número válido'
            }), 400
        
        # Procesar cada artículo
        results = []
        for i, article in enumerate(articles):
            try:
                title = article.get('title', '')
                abstract = article.get('abstract', '')
                
                if not title and not abstract:
                    results.append({
                        'index': i,
                        'title': title,
                        'error': 'Debe proporcionar al menos un título o resumen'
                    })
                    continue
                
                prediction = classifier.predict(title, abstract, threshold)
                results.append({
                    'index': i,
                    'title': title,
                    'prediction': prediction
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'title': article.get('title', ''),
                    'error': f'Error en predicción: {str(e)}'
                })
        
        return jsonify({
            'success': True,
            'data': results,
            'total_processed': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Endpoint para obtener información sobre el modelo"""
    global classifier
    
    if classifier is None:
        return jsonify({
            'success': False,
            'error': 'Modelo no está cargado'
        })
    
    try:
        return jsonify({
            'success': True,
            'data': {
                'etiquetas_disponibles': list(classifier.target_labels),
                'numero_etiquetas': len(classifier.target_labels),
                'modelo_cargado': True
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error al obtener información del modelo: {str(e)}'
        })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Método no permitido'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

if __name__ == '__main__':
    print("Iniciando API Flask...")
    
    # Inicializar clasificador al arrancar la aplicación
    if initialize_classifier():
        print("✓ Clasificador cargado exitosamente")
    else:
        print("✗ Error al cargar el clasificador")
        print("La API seguirá funcionando pero las predicciones fallarán")
    
    # Ejecutar la aplicación
    app.run(
        host='0.0.0.0',  # Permite conexiones desde cualquier IP
        port=5000,       # Puerto por defecto
        debug=True       # Cambiar a False en producción
    )