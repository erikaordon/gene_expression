# Importar librerías 
import pandas as pd 
# Importar SVM, en scikit-learn se llama Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con búsqueda aleatoria de hyperparámetros
from sklearn.model_selection import RandomizedSearchCV
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Importar objeto para escalamiento de datos
from sklearn.preprocessing import StandardScaler #normalización mediante puntuación z
# Importar metricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
# Importar objeto para imprimir matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importar objeto para imprimir reporte de clasificación
from sklearn.metrics import classification_report, matthews_corrcoef
# Plots
import matplotlib.pyplot as plt
# Import object to plot Precision-Recall curve
# from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import LabelEncoder, label_binarize


from scipy.special import softmax
import numpy as np

# Caragar los datos

def dowload_data(data_file, labels_file):
  df_data = pd.read_csv(data_file)

  df_labels = pd.read_csv(labels_file)

  return df_data, df_labels

# Separación de datos en conjuntos de entrenamiento y evaluación
def split_scale(df_data, df_labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
       df_data, 
       df_labels, 
       test_size=test_size,
       random_state=random_state, 
       stratify=df_labels)

    # Escalamiento de datos
    scaler = StandardScaler() # Instanciar
# StandardScaler - normalización mediante puntuación z
  # estandariza características elimminando la media y escalándollas a una varianza
  # fórmula: x_scaled = (X - μ) / σ
  # Efecto: media = 0 y ds = 1
    scaler.fit(X_train) # calcular media y desviación estándar
    X_train = scaler.transform(X_train) # aplicar transformación de estandarización
    X_test = scaler.transform(X_test)    

    return X_train, X_test, y_train, y_test, scaler

# Entrenamiento del SVM (support vector machine)

def training( search_parameters_rbf, search_parameters_linear, X_train, y_train):

    svm_model= SVC(probability=True)

    clf = RandomizedSearchCV(svm_model,
                            param_distributions=[search_parameters_linear, search_parameters_rbf],
                            random_state=0, cv=3,
                            scoring='f1_weighted',
                            verbose=1, n_jobs=-1, n_iter=30)

    classifier = clf.fit(X_train, y_train)

    return classifier

# Evaluación

def evaluation(classifier, X_test, y_test):
  # Evaluación
  # Predicción de los nuevos datos.
  y_pred = classifier.predict(X_test)
  # Antes de la preducción de los nuevos datos
  # Es necesaria para obtener los datos para elaborar la curva ROC.
  # El resultado son valores continuos
  y_score = classifier.decision_function(X_test)
  y_score_probability = softmax(y_score, axis=1)

  return y_pred, y_score_probability