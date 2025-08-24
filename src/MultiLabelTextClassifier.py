import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import pickle
import os
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

class MultiLabelTextClassifier:
    def __init__(self, model_dir='models'):
        """
        Inicializa el clasificador multilabel
        
        Args:
            model_dir: Directorio donde se guardarán los modelos y componentes
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.mlb = None
        self.target_labels = None
        self.lemmatizer = None
        self.stop_words = None
        self.history = None
        
        # Crear directorio de modelos si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Descargar recursos de NLTK
        self._download_nltk_resources()
        
    def _download_nltk_resources(self):
        """Descarga los recursos necesarios de NLTK"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """
        Preprocesa el texto: limpieza, eliminación de stopwords y lemmatización
        
        Args:
            text: Texto a preprocesar
            
        Returns:
            Texto preprocesado
        """
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return " ".join(words)
    
    def load_and_preprocess_data(self, csv_path, sep=';'):
        """
        Carga y preprocesa los datos desde un archivo CSV
        
        Args:
            csv_path: Ruta al archivo CSV
            sep: Separador del CSV
            
        Returns:
            DataFrame procesado con las características y etiquetas
        """
        print("Cargando datos...")
        df = pd.read_csv(csv_path, sep=sep)
        
        # Rellenar valores nulos
        df['title'] = df['title'].fillna('')
        df['abstract'] = df['abstract'].fillna('')
        
        print(f"Datos cargados: {df.shape}")
        
        # Procesar etiquetas multilabel
        df['group'] = df['group'].apply(lambda x: str(x).split('|'))
        self.mlb = MultiLabelBinarizer()
        y = self.mlb.fit_transform(df['group'])
        self.target_labels = self.mlb.classes_
        
        # Crear columnas para cada etiqueta
        df_labels = pd.DataFrame(y, columns=self.target_labels)
        df = pd.concat([df, df_labels], axis=1)
        
        print(f"Etiquetas encontradas: {list(self.target_labels)}")
        print(f"Número de artículos por clase:\n{df[self.target_labels].sum()}")
        
        return df
    
    def create_visualizations(self, df):
        """
        Crea visualizaciones del análisis exploratorio de datos
        
        Args:
            df: DataFrame con los datos procesados
        """
        print("\nCreando visualizaciones...")
        
        # Distribución de clases
        class_counts = df[self.target_labels].sum()
        plt.figure(figsize=(12, 6))
        class_counts.plot(kind='bar', color='skyblue')
        plt.title('Distribución de Artículos por Clase')
        plt.xlabel('Clase')
        plt.ylabel('Número de Artículos')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'class_distribution.png'))
        
        # Distribución de longitud de textos
        df['text_combined'] = df['title'] + " " + df['abstract']
        df['text_length'] = df['text_combined'].apply(lambda x: len(str(x).split()))
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'], bins=50, kde=True, color='purple')
        plt.title('Distribución de la Longitud de los Textos')
        plt.xlabel('Número de Palabras')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'text_length_distribution.png'))
    
    def prepare_features(self, df, max_features=10000, ngram_range=(1,2)):
        """
        Prepara las características para el entrenamiento
        
        Args:
            df: DataFrame con los datos
            max_features: Número máximo de características TF-IDF
            ngram_range: Rango de n-gramas
            
        Returns:
            Matrices de entrenamiento y prueba, etiquetas
        """
        print("\nPreparando características...")
        
        # Combinar y limpiar texto
        df['text_combined'] = df['title'] + " " + df['abstract']
        df['cleaned_text'] = df['text_combined'].apply(self.preprocess_text)
        
        # Vectorización TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_tfidf = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df[self.target_labels]
        
        # División de datos
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=12
        )
        
        # Convertir a matrices densas para Keras
        X_train_dense = X_train_tfidf.toarray()
        X_test_dense = X_test_tfidf.toarray()
        
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        print(f"Forma de X_train: {X_train_dense.shape}")
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Forma de X_test: {X_test_dense.shape}")
        print(f"Forma de y_test: {y_test.shape}")
        
        return X_train_dense, X_test_dense, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """
        Construye la red neuronal
        
        Args:
            input_shape: Forma de la entrada
            num_classes: Número de clases de salida
            
        Returns:
            Modelo compilado
        """
        print("\nConstruyendo la red neuronal...")
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_shape,),
                  kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.4),
            
            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.4),
            
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.2),
            
            Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.1),
            
            Dense(num_classes, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
    def train_model(self, X_train, y_train, epochs=17, batch_size=32, validation_split=0.1):
        """
        Entrena el modelo
        
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento
            epochs: Número de épocas
            batch_size: Tamaño del lote
            validation_split: Proporción para validación
            
        Returns:
            Historial de entrenamiento
        """
        print("\nEntrenando el modelo...")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo entrenado
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        print("\nEvaluando el modelo...")
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba, average='weighted', multi_class='ovr')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        return metrics, y_pred_proba, y_pred
    
    def fit(self, csv_path, sep=';', **kwargs):
        """
        Método principal para entrenar el modelo completo
        
        Args:
            csv_path: Ruta al archivo CSV con los datos
            sep: Separador del CSV
            **kwargs: Argumentos adicionales para el entrenamiento
        """
        # Cargar y preprocesar datos
        df = self.load_and_preprocess_data(csv_path, sep)
        
        # Crear visualizaciones
        self.create_visualizations(df)
        
        # Preparar características
        X_train, X_test, y_train, y_test = self.prepare_features(df, **kwargs)
        
        # Construir modelo
        self.model = self.build_model(X_train.shape[1], len(self.target_labels))
        
        # Entrenar modelo
        self.train_model(X_train, y_train, **kwargs)
        
        # Evaluar modelo
        metrics, y_pred_proba, y_pred = self.evaluate_model(X_test, y_test)
        
        # Guardar modelo y componentes
        self.save_model()
        
        return metrics
    
    def predict(self, title, abstract, threshold=0.5):
        """
        Realiza predicción para un nuevo artículo
        
        Args:
            title: Título del artículo
            abstract: Resumen del artículo
            threshold: Umbral para clasificación binaria
            
        Returns:
            Diccionario con predicciones y probabilidades
        """
        if self.model is None or self.vectorizer is None or self.mlb is None:
            raise ValueError("El modelo no está entrenado o cargado. Use fit() o load_model() primero.")
        
        # Combinar y preprocesar texto
        texto_combinado = str(title) + " " + str(abstract)
        texto_limpio = self.preprocess_text(texto_combinado)
        
        # Vectorizar
        texto_vectorizado = self.vectorizer.transform([texto_limpio])
        texto_vectorizado_dense = texto_vectorizado.toarray()
        
        # Predecir probabilidades
        prediccion_proba = self.model.predict(texto_vectorizado_dense, verbose=0)[0]
        
        # Aplicar umbral
        prediccion_binaria = (prediccion_proba > threshold).astype(int)
        
        # Obtener etiquetas predichas
        etiquetas_predichas = self.mlb.inverse_transform(prediccion_binaria.reshape(1, -1))[0]
        
        # Crear diccionario de probabilidades
        probabilidades = {}
        for i, label in enumerate(self.target_labels):
            probabilidades[label] = float(prediccion_proba[i])
        
        resultado = {
            'etiquetas_predichas': list(etiquetas_predichas),
            'probabilidades': probabilidades,
            'umbral_usado': threshold
        }
        
        return resultado
    
    def save_model(self, model_name='text_classifier'):
        """
        Guarda el modelo y todos sus componentes
        
        Args:
            model_name: Nombre base para los archivos del modelo
        """
        print(f"\nGuardando modelo en {self.model_dir}...")
        
        # Guardar modelo de Keras
        model_path = os.path.join(self.model_dir, f'{model_name}_model.h5')
        self.model.save(model_path)
        
        # Guardar vectorizador TF-IDF
        vectorizer_path = os.path.join(self.model_dir, f'{model_name}_vectorizer.pkl')
        joblib.dump(self.vectorizer, vectorizer_path)
        
        # Guardar MultiLabelBinarizer
        mlb_path = os.path.join(self.model_dir, f'{model_name}_mlb.pkl')
        joblib.dump(self.mlb, mlb_path)
        
        # Guardar etiquetas objetivo
        labels_path = os.path.join(self.model_dir, f'{model_name}_labels.pkl')
        with open(labels_path, 'wb') as f:
            pickle.dump(self.target_labels, f)
        
        print(f"Modelo guardado exitosamente:")
        print(f"- Modelo Keras: {model_path}")
        print(f"- Vectorizador: {vectorizer_path}")
        print(f"- MultiLabelBinarizer: {mlb_path}")
        print(f"- Etiquetas: {labels_path}")
    
    def load_model(self, model_name='text_classifier'):
        """
        Carga el modelo y todos sus componentes
        
        Args:
            model_name: Nombre base de los archivos del modelo
        """
        print(f"\nCargando modelo desde {self.model_dir}...")
        
        try:
            # Cargar modelo de Keras
            model_path = os.path.join(self.model_dir, f'{model_name}_model.h5')
            self.model = load_model(model_path)
            
            # Cargar vectorizador TF-IDF
            vectorizer_path = os.path.join(self.model_dir, f'{model_name}_vectorizer.pkl')
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Cargar MultiLabelBinarizer
            mlb_path = os.path.join(self.model_dir, f'{model_name}_mlb.pkl')
            self.mlb = joblib.load(mlb_path)
            
            # Cargar etiquetas objetivo
            labels_path = os.path.join(self.model_dir, f'{model_name}_labels.pkl')
            with open(labels_path, 'rb') as f:
                self.target_labels = pickle.load(f)
            
            print("Modelo cargado exitosamente!")
            print(f"Etiquetas disponibles: {list(self.target_labels)}")
            
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo: {str(e)}")

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia del clasificador
    classifier = MultiLabelTextClassifier()
    
    # Entrenar el modelo (descomenta para entrenar)
    metrics = classifier.fit('../data/challenge_data-18-ago.csv')
    print("Métricas de entrenamiento:", metrics)

    # Cargar modelo preentrenado
    # classifier.load_model()
    
    
    # Hacer predicción
    resultado = classifier.predict(
        title="Machine Learning Applications in Healthcare",
        abstract="This paper explores the use of machine learning algorithms in medical diagnosis and treatment planning."
    )
    print("Resultado de predicción:", resultado)
