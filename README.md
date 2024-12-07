Clasificación de Noticias con Modelos de Machine Learning
Este proyecto tiene como objetivo implementar y comparar diferentes arquitecturas de modelos de aprendizaje profundo para la clasificación automática de noticias en múltiples categorías. Se evaluaron modelos como RNN y LSTM combinado con CNN para determinar su desempeño y viabilidad en aplicaciones prácticas de procesamiento de lenguaje natural (NLP).

Estructura del Proyecto
1. Dataset
El dataset utilizado contiene noticias etiquetadas en múltiples categorías, como deportes, economía, política, y tecnología. Incluye información textual y está desbalanceado en términos de representación de clases.

Tamaño del dataset: ~275 noticias.
Atributos principales:
Texto de la noticia: El contenido principal que se usó como entrada para los modelos.
Etiqueta: Categoría a la que pertenece cada noticia.
2. Modelos Implementados
Se implementaron y compararon dos modelos principales:

RNN (Red Neuronal Recurrente):
Arquitectura básica para procesamiento de secuencias.
Resultados moderados debido a limitaciones en el manejo de dependencias largas.
LSTM con CNN:
Arquitectura avanzada que combina:
LSTM (Long Short-Term Memory): Para capturar dependencias temporales en el texto.
CNN (Redes Neuronales Convolucionales): Para extraer patrones relevantes en el texto.
Mostró un desempeño significativamente superior.
3. Resultados
Los resultados de los modelos se evaluaron mediante métricas estándar de clasificación:

Precisión (Accuracy): Porcentaje de predicciones correctas.
F1-Score: Equilibrio entre precisión y recall.
Recall: Capacidad del modelo para identificar correctamente ejemplos de cada clase.
Principales Resultados:
RNN:
Precisión: 29.09%
Macro F1-Score: 28.36%
Modelo limitado para manejar clases desbalanceadas.
LSTM con CNN:
Precisión: 90.91%
Macro F1-Score: 90.99%
Desempeño sobresaliente, con métricas equilibradas entre clases.
Requisitos
Tecnologías utilizadas:
Python 3.8 o superior
Bibliotecas principales:
TensorFlow / Keras: Para la implementación de los modelos de aprendizaje profundo.
Scikit-learn: Para la evaluación de métricas.
Matplotlib y Seaborn: Para la visualización de resultados.
Pandas y NumPy: Para manipulación y análisis de datos.
Instalación
Clona este repositorio e instala las dependencias utilizando el archivo requirements.txt:

bash
Copiar código
git clone <url-del-repositorio>
cd proyecto-clasificacion-noticias
pip install -r requirements.txt
Uso
1. Preprocesamiento de Datos
El preprocesamiento incluye:

Tokenización del texto.
Limpieza de datos (eliminación de stopwords, puntuación, etc.).
Conversión a embeddings (opcional: Word2Vec, GloVe o embeddings preentrenados).
Ejecuta el script de preprocesamiento:

bash
Copiar código
python preprocessing.py
2. Entrenamiento de Modelos
Para entrenar y evaluar los modelos, ejecuta el archivo principal:

bash
Copiar código
python train_models.py
3. Comparación de Resultados
Los resultados se guardarán en la carpeta results/ e incluirán:

Métricas de clasificación (accuracy, precision, recall, F1-score).
Gráficas comparativas de desempeño entre los modelos.
Aplicaciones Prácticas
Este proyecto tiene aplicaciones directas en:

Sistemas de Recomendación: Clasificación automática de noticias para personalizar contenido.
Análisis de Medios: Monitoreo y análisis de tendencias en noticias.
Moderación de Contenidos: Filtrado automático de noticias irrelevantes o inapropiadas.
Posibles Mejoras
Incorporar mecanismos de atención para mejorar el enfoque del modelo en palabras clave.
Implementar embeddings preentrenados como BERT o GloVe.
Experimentar con arquitecturas híbridas como LSTM + Transformer.
Ampliar el dataset para mejorar la generalización del modelo.