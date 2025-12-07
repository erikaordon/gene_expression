# Importar librerías 
import pandas as pd 
# Importar SVM, en scikit-learn se llama Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con búsqueda aleatoria de hyperparámetros
from sklearn.model_selection import RandomizedSearchCV
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Obtener valore de una distribución uniforme
from scipy.stats import uniform, expon
# Importar objeto para escalamiento de datos
from sklearn.preprocessing import StandardScaler #normalización mediante puntuación z
# Importar metricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
# Importar objeto para imprimir matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importar objeto para imprimir reporte de clasificación
from sklearn.metrics import classification_report
# Plots
import matplotlib.pyplot as plt
# Import object to plot Precision-Recall curve
# from sklearn.metrics import PrecisionRecallDisplay

# Caragar los datos
def dowload_data(data_file, labels_file):
    relative_path_data = '../Data/gene+expression+cancer+rna+seq/data.csv'
    df_data = pd.read_csv(relative_path_data)
    print(df_data.head(6))

    relative_path_labels = '../Data/gene+expression+cancer+rna+seq/labels.csv'
    df_labels = pd.read_csv(relative_path_labels)
    print(df_labels.head(6))

# Verificar categorías de clase 
print(df_labels.iloc[:,1].unique())

print(f"Dimensionalidad del data set: {df_data.shape}")
print(f"Dimensionalidad de los label: {df_labels.shape}")