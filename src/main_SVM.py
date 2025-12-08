# Importar funciones del archivo SVM.py como módulo
import SVM 
# Importar objeto para manejar matrices
import numpy as np
# Importar objeto para graficar (plots)
import matplotlib.pyplot as plt
# Importar objeto para reducción de dimensionalidad 
from sklearn.decomposition import PCA # Para visualización y reducción de dimensionalidad
# Importar objetos para obtener valores que se distribuyen uniformemente en una escala logarítmica (variables aleatorias continuas). 
from scipy.stats import loguniform # Para hiperparámtros 
# Importar métricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay, classification_report, matthews_corrcoef, auc
# Importar objeto para convertir categorías en números y en formato binario 
from sklearn.preprocessing import LabelEncoder, label_binarize
# Importar objeto para concatenar pasos de procesamiento y estimación 
from sklearn.pipeline import Pipeline # Simplificación en cross-validation e hiperparámetros

# Preparación de datos 
df_data, df_labels = SVM.download_data('../Data/gene+expression+cancer+rna+seq/data.csv', 
                                      '../Data/gene+expression+cancer+rna+seq/labels.csv')

# Verificar categorías de clase
print(f"Categorías de clase: {df_labels.iloc[:,1].unique()}")

# Verificar dimensionalidad de datos
print(f"Dimensionalidad del data set: {df_data.shape}")
print(f"Dimensionalidad de los label: {df_labels.shape}")

# Comprobación de la dimensionalidad de los datos
# Comprobar si existen las mismas muestras en df_data que en df_labels
samples_data = set(df_data['Unnamed: 0']) - set(df_labels['Unnamed: 0'])
# Comprobar si existen las mismas muestras en df_labels que en df_data
samples_labels = set(df_labels['Unnamed: 0']) - set(df_data['Unnamed: 0'])

if samples_data:
  print(f"Muestras que se encuentran en el dataframe de los datos pero no en los labels: {sorted(samples_data)}")
else:
  print("Todas las muestras se comparten entre los datos y los labels")

if samples_labels:
  print(f"Muestras que se encuentran en el dataframe de los labels pero no en el de los datos: {sorted(samples_labels)}")
else:
  print("Todas las muestras se comparten entre los labels y los datos")

# Verificar ejemplos por categoría, verificar si existe desbalance de datos
print(f"Frecuencia de las categorías: {df_labels.groupby('Class').size()}")

# Preparar datos
# X = sólo datos de características de ejemplos
# y = categorías de clase para cada ejemplo

# Quitar la primer columna, obtaculiza el proceso de aprendizaje 
df_data = df_data.drop(['Unnamed: 0'], axis=1)
# df_labels ahora tiene solamente la información correspondiente a las categorías
df_labels = df_labels.iloc[:, 1].values

# Mostrar los resultados 
print("Datos")
print(df_data.head(6))
print("Categorias de clase")
print(df_labels[:6])

# Codificar las clases
le = LabelEncoder()
df_labels = le.fit_transform(df_labels)
print(f"Muestra de la codificación. {df_labels[:6]}")
#==========================================================

# Mostrar desbalance de los datos

# Hacer un conteo de clases 
classes, counts = np.unique(df_labels, return_counts=True)
class_names = le.inverse_transform(classes)

# Vizualización 
plt.figure(figsize=(8,6))
plt.bar(classes, counts, 
        tick_label=class_names, 
        color=['cornflowerblue', 'royalblue', 'lightsteelblue', 'midnightblue','darkblue'])
plt.xlabel('Clase')
plt.ylabel('Número de ocurrencias')
plt.title("Distribución de clases (desbalance de los datos)")
# Modificiar los números que corresponden a cada categoría
for i, count in enumerate(counts):
  plt.text(i, count + 10, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('../results/SVM_results/Distribution.png')
plt.show()

#======================================================

# Separación de datos en conjuntos de entrenamiento y evaluación
# y Escalamiento de los datos 
X_train_PCA, X_test_PCA, y_train, y_test, scaler = SVM.split_scale(df_data, df_labels)

#====================================================

# Gráficas 
# PCA - visuaización para la desición del kernel 

# Reducción de dimensionalidad 
pca = PCA(n_components=2)
# Transformación de los datos
x_pca = pca.fit_transform(X_train_PCA)

# Vizualización
plt.figure(figsize=(8, 6))
scatter_plot = plt.scatter(x_pca[:, 0], x_pca[:, 1], c= y_train)
handles, labels = scatter_plot.legend_elements()
plt.legend(handles, class_names, title="Clases")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig('../results/SVM_results/PCA_for_visualization.png')
plt.show()

#============================================================

# Hiperparámetros para el entrenamiento de SVM 
# NOTA: la decisión de kernels se hizo con base en el plot anterior

# Se vuelve a llamar a la función para separar lo datos de entrenamiento y evaluación
X_train, X_test, y_train, y_test = SVM.split(df_data, df_labels)

# Definición de la función Pipeline
# Se define la función para los datos de validación 
pipeline = Pipeline([('scaler', SVM.StandardScaler()),
                     ('svm', SVM.SVC(probability=True))])

# Diccionario con los hiperparámetros para los kernels rbf y linear
                        # kernel rbf
search_parameters = [{'svm__C': loguniform(1e-3, 1e2),
                        'svm__gamma': loguniform(1e-4, 1),
                    'svm__kernel': ['rbf'],
                        'svm__class_weight':['balanced']},
                        
                        # kernel linear
                        {'svm__C': loguniform(1e-3, 1e2),
                            'svm__kernel': ['linear']}
                        ]

#=================================================================

# Verificamos resultados de entrenamiento y cross-validation
classifier = SVM.training(pipeline,
                          search_parameters,
                          X_train,
                          y_train )

# Imprimir mejores resultados 
print("Mejor F-1 weighted score")
print(classifier.best_score_)
print("Mejores hyperparámetros")
print(classifier.best_params_)

#============================================================

# Evaluación

y_pred, y_score_probability = SVM.evaluation(classifier, X_test, y_test)

# Imprimir los resultados de la evaluación 
print("Resultados de evaluación")
print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted')}")
print("Precision: {}".format(precision_score(y_test, y_pred, average='weighted')))
print("Recall: {}".format(recall_score(y_test, y_pred, average='weighted')))
print("MCC: {}". format(matthews_corrcoef(y_test, y_pred)))
print("AUROC weighted: {}".format(roc_auc_score(y_test, y_score_probability, multi_class='ovr', average='weighted')))

#=============================================================

# Matriz de confusión
svm_confusion_matrix=confusion_matrix(y_test,y_pred)
print(f"Matriz de confusión: {svm_confusion_matrix}")

# Visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix,
                               display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
# Guardar la matriz
plt.savefig('../results/SVM_results/confusion_matrix.png')
# Mostrar la matriz
plt.show()

# Reporte de clasificación
print(f"Reporte de clasificación: {classification_report(y_test,y_pred)}")

#=======================================================================

# Plots ROC

best_svm = classifier.best_estimator_
y_pred = best_svm.predict(X_test)

ROC_plt = SVM.plot(
        model = best_svm,
        X_test = X_test,
        y_test = y_test,
        class_names = le.classes_,
        file = '../results/SVM_results/ROC_curve.png'
    )

# Curva ROC por cada clase:

plot_per_class=SVM.plot_per_class(model=best_svm,
    X_test=X_test,
    y_test=y_test,
    class_names=le.classes_,
    file = '../results/SVM_results/ROC_curve_per_class.png')

# Guardar resultados 

results = SVM.pd.DataFrame(classifier.cv_results_)
results.to_csv('../results/results_svm.csv', index=True, encoding='utf-8')

#=============================================================================

# Comprobación con reducción de datos:

import SVM_reduction

ks = [500, 1000, 2000, 3000]
resultados = []

mejor_score = -np.inf
mejor_k = None
mejor_classifier = None
mejor_selector = None
mejor_vt = None

for k in ks:
  classifier, selector, vt_current = SVM_reduction.training_with_reduction(pipeline, search_parameters,
                            X_train,
                            y_train,
                            k=k)
  mean_cv = classifier.best_score_
  resultados.append((k, mean_cv))

  if mean_cv > mejor_score:
        mejor_score = mean_cv
        mejor_k = k
        mejor_classifier = classifier
        mejor_selector = selector
        mejor_vt = vt_current

print("Resultados por k:", resultados)
print("Mejor k:", mejor_k, "con score:", mejor_score)

X_test_var = mejor_vt.transform(X_test)
X_test_sel = mejor_selector.transform(X_test_var)

best_svm = mejor_classifier.best_estimator_
y_pred   = best_svm.predict(X_test_sel)

print("Mejor F-1 weighted score")
print(mejor_classifier.best_score_)
print("Mejores hyperparámetros")
print(mejor_classifier.best_params_)

y_pred, y_score_probabilities = SVM_reduction.evaluation(best_svm, X_test_sel, y_test)

print("Resultados de evaluación")
print("F-1 weighted: {}".format(f1_score(y_test, y_pred, average='weighted')))
print("Precision: {}".format(precision_score(y_test, y_pred, average='weighted')))
print("Recall: {}".format(recall_score(y_test, y_pred, average='weighted')))
print("MCC: {}". format(matthews_corrcoef(y_test, y_pred)))
print("AUROC weighted: {}".format(roc_auc_score(y_test, y_score_probabilities, multi_class='ovr', average='weighted')))

# Matriz de confusión
svm_confusion_matrix=confusion_matrix(y_test,y_pred)
print(f"Matriz de confusión con la reducción de dimensionalidad: {svm_confusion_matrix}")

disp = ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix,
                               display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.savefig('../results/SVM_results/confusion_matrix_with_reduction.png')
plt.show()

# Reporte de clasificación
print(f"Reporte de clasificación con reducción de dimensionalidad: {classification_report(y_test,y_pred)}")

# ROC curve plot 

best_svm = mejor_classifier.best_estimator_
y_pred = best_svm.predict(X_test_sel)

ROC_plot_with_reduction = SVM.plot(
        model = best_svm,
        X_test = X_test_sel,
        y_test = y_test,
        class_names = le.classes_,
        file = '../results/SVM_results/ROC_curve_with_reduction.png')

# Curva ROC por cada clase:

ROC_curve_per_class=SVM.plot_per_class(model=best_svm,
    X_test=X_test_sel,
    y_test=y_test,
    class_names=le.classes_, 
    file='../results/SVM_results/ROC_curve_per_class_with_reduction.png')