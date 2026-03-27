# Assignment 6 — Multilayer Neural Network with Hyperparameter Optimisation

## Descripción
Proyecto académico que desarrolla, optimiza y evalúa una red neuronal multicapa (MLP) para clasificar la aceptabilidad de vehículos usando el dataset "Car Evaluation" (UCI Machine Learning Repository). El objetivo es entrenar un clasificador robusto frente al desbalance de clases, optimizando hiperparámetros con Optuna y evaluando el equilibrio entre clases mediante la métrica G-Mean.

## Dataset
- Origen: UCI Machine Learning Repository — Car Evaluation (1997).  
- Registros: 1,728.  
- Features (6): `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety` (todas categóricas ordinales).  
- Variable objetivo (target): `class` con 4 categorías — `unacc`, `acc`, `good`, `vgood`.

## Tecnologías y librerías
- Python (recomendado >= 3.9)
- NumPy
- pandas
- scikit-learn
- TensorFlow / Keras
- imbalanced-learn
- Optuna
- Matplotlib
- Seaborn
- ucimlrepo (para descargar el dataset desde UCI)

## Instalación
1. Crear un entorno virtual (recomendado):

   python -m venv .venv
   .venv\Scripts\activate

2. Instalar dependencias:

   pip install -r requirements.txt

3. Abrir el cuaderno Jupyter:

   jupyter lab    # o `jupyter notebook`

Nota: Para entrenamientos extensos, se recomienda ejecutar en una máquina con GPU y la versión de TensorFlow compatible.

## Cómo ejecutar el notebook
1. Abrir `Assignment6_MultilayerNeuralNetwork_PCA.ipynb` en JupyterLab/Notebook.  
2. Ejecutar las celdas en orden (la ejecución secuencial reproduce el flujo: carga → preprocesamiento → entrenamiento → optimización → evaluación → guardado del modelo).  
3. Para reproducir exactamente los resultados, asegúrese de ejecutar todas las celdas desde el principio y tener instaladas las mismas versiones especificadas en `requirements.txt`.

## Flujo del proyecto (resumen)
- Carga del dataset usando `ucimlrepo.fetch_ucirepo`.  
- Mapear variables categóricas ordinales a valores enteros (mapeo manual).  
- Análisis exploratorio y visualizaciones (`seaborn`, `matplotlib`).  
- División `train/test` estratificada (`train_test_split`) y estandarización (`StandardScaler`).  
- Cálculo de `class_weight` para compensar el desbalance.  
- Definición de un MLP base con `tf.keras`, entrenamiento con `EarlyStopping`.  
- Evaluación con `confusion_matrix`, `classification_report` y `geometric_mean_score` (G-Mean).  
- Optimización de hiperparámetros con Optuna (espacio de búsqueda: learning_rate, batch_size, número/cantidad de neuronas por capa).  
- Reconstrucción del modelo con los mejores hiperparámetros, reentrenamiento y evaluación final.  
- Persistencia del modelo con `optimized_model.save('optimized_car_evaluation.keras')`.

## Resultados y métricas clave
- Accuracy (test, modelo optimizado): ~0.9682
- G-Mean (evaluación final en test, modelo optimizado): ~0.8591
- Observaciones: accuracy alto pero no suficiente por el desbalance; G-Mean reporta el equilibrio entre recalls de cada clase.

## Conclusiones
- El uso de `class_weight` y la optimización bayesiana (Optuna) permitió obtener modelos con buen balance entre clases minoritarias y mayoritarias.  
- El G-Mean es una métrica adecuada para problemas multiclase desbalanceados y guió correctamente la optimización.  
- El pipeline completo (preprocesamiento, ponderación, early stopping y optimización) demostró ser efectivo para este dataset.

## Mejoras futuras
- Realizar validación cruzada estratificada para estimar mejor la varianza de los resultados.  
- Probar técnicas de sobremuestreo/undersampling (SMOTE u otras variantes) combinadas con `class_weight`.  
- Experimentar arquitecturas más profundas o regularizaciones (Dropout, BatchNorm) y distintos optimizadores.  
- Automatizar pipelines con `scikit-learn` `Pipeline` o `tf.data` para producción.  
- Incluir seguimiento de experimentos (MLflow, Weights & Biases) para reproducibilidad y comparación sistemática.

## Artefactos generados
- Modelo guardado: `optimized_car_evaluation.keras`  
- Notebook principal: `Assignment6_MultilayerNeuralNetwork_PCA.ipynb`
