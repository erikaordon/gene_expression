# Modularización
import SVM 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Obtener valore de una distribución uniforme
from scipy.stats import loguniform
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer, matthews_corrcoef

# MAIN

# Preparación de datos 

df_data, df_labels = SVM.dowload_data('../Data/gene+expression+cancer+rna+seq/data.csv', 
                                      '../Data/gene+expression+cancer+rna+seq/labels.csv')

# Verificar categorías de clase
print(df_labels.iloc[:,1].unique())

# Verificar dimensionalidad de datos
print(f"Dimensionalidad del data set: {df_data.shape}")
print(f"Dimensionalidad de los label: {df_labels.shape}")

# Comprobación de la dimensionalidad de los datos

samples_data = set(df_data['Unnamed: 0']) - set(df_labels['Unnamed: 0'])
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
print(df_labels.groupby('Class').size())

# Preparar datos
# X = sólo datos de características de ejemplos
# y = categorías de clase para cada ejemplo

df_data = df_data.drop(['Unnamed: 0'], axis=1)
df_labels = df_labels.iloc[:, 1].values

print("Datos")
print(df_data.head(6))
print("Categorias de clase")
print(df_labels[3])

# Codificar las clases
le = SVM.LabelEncoder()
df_labels = le.fit_transform(df_labels)
print(df_labels)

#==========================================================

# Desbalance de los datos

classes, counts = np.unique(df_labels, return_counts=True)

plt.figure(figsize=(8,6))
plt.bar(classes, counts, tick_label=classes, color=['cornflowerblue', 'royalblue', 'lightsteelblue', 'midnightblue','darkblue'])

plt.xlabel('Clase')
plt.ylabel('Número de ocurrencias')
plt.title("Distribución de clases (desbalance de los datos)")

for i, count in enumerate(counts):
  plt.text(i, count + 10, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

#======================================================

# Separación de datos en conjuntos de entrenamiento y evaluación
# y Escalamiento de los datos 
X_train, X_test, y_train, y_test, scaler = SVM.split_scale(df_data, df_labels)

#====================================================

# PCA - visuaización para la desición del kernel 

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
scatter_plot = plt.scatter(x_pca[:, 0], x_pca[:, 1], c= y_train)
plt.legend(*scatter_plot.legend_elements(), title="Clases")
plt.grid(True)
plt.show()

#============================================================

# Hiperparámetros 

search_parameters_rbf = {'C': loguniform(1e-3, 1e2),
                        'gamma': loguniform(1e-4, 1),
                        'kernel': ['rbf'],
                        'class_weight':['balanced']}

search_parameters_linear = { 'C': loguniform(1e-3, 1e2),
                            'kernel': ['linear']}

#=================================================================

# Verificamos resultados de entrenamiento y cross-validation

classifier = SVM.training( search_parameters_rbf,
                          search_parameters_linear,
                          X_train,
                          y_train )

print("Mejor F-1 weighted score")
print(classifier.best_score_)
print("Mejores hyperparámetros")
print(classifier.best_params_)

#============================================================

# Evaluación

y_pred, y_score_probability = SVM.evaluation(classifier, X_test, y_test)

print("Resultados de evaluación")
print(f"F1 weighted: {f1_score(y_test, y_pred, average='weighted')}")
print("Precision: {}".format(precision_score(y_test, y_pred, average='weighted')))
print("Recall: {}".format(recall_score(y_test, y_pred, average='weighted')))
print("MCC: {}". format(matthews_corrcoef(y_test, y_pred)))
print("AUROC weighted: {}".format(roc_auc_score(y_test, y_score_probability, multi_class='ovr', average='weighted')))

#=============================================================

# Matriz de confusión
svm_confusion_matrix=SVM.confusion_matrix(y_test,y_pred)
print(svm_confusion_matrix)

disp = SVM.ConfusionMatrixDisplay(confusion_matrix=svm_confusion_matrix,
                               display_labels=classifier.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Reporte de clasificación
print(SVM.classification_report(y_test,y_pred))

#=======================================================================

# Plots ROC

import numpy as np
from sklearn.metrics import auc # Import auc explicitly

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score


def plot(model, X_test, y_test, class_names):

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
    plt.show()

best_svm = classifier.best_estimator_
y_pred = best_svm.predict(X_test)

plot(
        model = best_svm,
        X_test = X_test,
        y_test = y_test,
        class_names = le.classes_
    )

# Curva ROC por cada clase:

def plot_per_class(model, X_test, y_test, class_names):

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
        plt.show()


plot_per_class(model=best_svm,
    X_test=X_test,
    y_test=y_test,
    class_names=le.classes_)

# Guardar resultados 

results = SVM.pd.DataFrame(classifier.cv_results_)
results.to_csv('../results/results_svm.csv', index=True, encoding='utf-8')