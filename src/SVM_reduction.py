# Importar librerías
import pandas as pd
import numpy as np
# Importar SVM, en scikit-learn se llama Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con búsqueda aleatoria de hyperparámetros
from sklearn.model_selection import RandomizedSearchCV
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Obtener valore de una distribución uniforme
from scipy.stats import loguniform, expon
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

from sklearn.decomposition import PCA
from scipy.special import softmax
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold


def training_with_reduction(search_parameters_rbf, search_parameters_linear, X_train, y_train, k):
    vt = VarianceThreshold(threshold=0.0)
    X_train_var = vt.fit_transform(X_train)

    svm_model= SVC(probability=True)

    selector = SelectKBest(score_func=f_classif, k=min(k, X_train_var.shape[1]))
    X_train_sel = selector.fit_transform(X_train_var, y_train)

    clf = RandomizedSearchCV(svm_model,
                            param_distributions=[search_parameters_linear, search_parameters_rbf],
                            random_state=0, cv=3,
                            scoring='f1_weighted',
                            verbose=1, n_jobs=-1, n_iter=30)

    classifier = clf.fit(X_train_sel, y_train)

    return classifier, selector, vt

# Evaluación 

def evaluation(model, X_test_processed, y_test):
  # Evaluación
  # Predicción de los nuevos datos.
  y_pred = model.predict(X_test_processed)
  # Es necesaria para obtener los datos para elaborar la curva ROC.
  # El resultado son valores continuos
  y_score_probabilities = model.predict_proba(X_test_processed)

  return y_pred, y_score_probabilities