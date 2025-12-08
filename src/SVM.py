# Importar librerías

# Importar objeto para manipular data frames  
import pandas as pd 
# Importar SVM, en scikit-learn se llama Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con búsqueda aleatoria de hyperparámetros
from sklearn.model_selection import RandomizedSearchCV
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Importar objeto para escalamiento de datos
from sklearn.preprocessing import StandardScaler #normalización mediante puntuación z
# Importar objeto para convertir valores numéricos en probabilidad para la curva ROC
from scipy.special import softmax
# Importar objeto para graficar (plots)
import matplotlib.pyplot as plt
# Importar objeto para manejar matrices
import numpy as np
# Importar objeto para convertir categorías en números y en formato binario 
from sklearn.preprocessing import LabelEncoder, label_binarize
# Importar métricas de evaluación
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Caragar los datos

def download_data(data_file, labels_file):
  df_data = pd.read_csv(data_file)

  df_labels = pd.read_csv(labels_file)

  return df_data, df_labels

# Separación de datos en conjuntos de entrenamiento y evaluación para la representación del PCA
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
    scaler.fit(X_train) # calcular media y desviación estándar
    X_train_PCA = scaler.transform(X_train) # aplicar transformación de estandarización
    X_test_PCA = scaler.transform(X_test)    

    return X_train_PCA, X_test_PCA, y_train, y_test, scaler

# # Separación de datos en conjuntos de entrenamiento y evaluación para el resto del código

def split(df_data, df_labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
       df_data, 
       df_labels, 
       test_size=test_size,
       random_state=random_state, 
       stratify=df_labels)
    return X_train, X_test, y_train, y_test

# Entrenamiento del SVM (support vector machine)

def training(pipeline, search_parameters, X_train, y_train):

    clf = RandomizedSearchCV(pipeline,
                            param_distributions= search_parameters,
                            random_state=0, cv=3,
                            scoring='f1_weighted',
                            verbose=1, n_jobs=-1, n_iter=30)

    classifier = clf.fit(X_train, y_train)

    return classifier

# Evaluación

def evaluation(classifier, X_test, y_test):
  # Predicción de los nuevos datos.
  y_pred = classifier.predict(X_test)
  # Antes de la preducción de los nuevos datos
  # Es necesaria para obtener los datos para elaborar la curva ROC.
  # El resultado son valores continuos
  y_score = classifier.decision_function(X_test)
  y_score_probability = softmax(y_score, axis=1)

  return y_pred, y_score_probability

# ROC plots

def plot(model, X_test, y_test, class_names, file):

    classes = np.unique(y_test)
    n_classes = len(classes)

    y_test_bin = label_binarize(y_test, classes=classes)

    y_score = model.predict_proba(X_test)

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    auc_roc_weighted_ovr = roc_auc_score(
        y_test,
        y_score,
        multi_class="ovr",
        average="weighted"
    )

    print("AUC ROC (OVR, weighted):", auc_roc_weighted_ovr)

    plt.figure(figsize=(8, 8))

    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f'Clase {classes[i]} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title('Curvas ROC multiclase (one-vs-rest) – SVM')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file)
    plt.show()

# Curva ROC por clase 

def plot_per_class(model, X_test, y_test, class_names, file):

  classes = np.unique(y_test)
  y_test_bin = label_binarize(y_test, classes=classes)
  y_score_probabilities = model.predict_proba(X_test)

  for i, cls in enumerate(classes):

        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score_probabilities[:, i]) # Use probabilities
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, lw=2,
                 label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

        title = f"ROC – Clase {cls}"
        if class_names is not None:
            title = f"ROC – {class_names[cls]}"

        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(file)
        plt.show()