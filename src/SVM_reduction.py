# Importar librerías

# Importar SVM, en scikit-learn se llama Support Vector Classifier (SVC)
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con búsqueda aleatoria de hyperparámetros
from sklearn.model_selection import RandomizedSearchCV
# Importar clase SelectKBest de scikit-learn para la selección de características 
# Importar función f_classif también para la selección de características en problemas de clasificación
# Importar clase VarianceThreshold de scikit-learn también para la selección de características 
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

# Entrenamiento con reducción de dimensionalidad 
def training_with_reduction(pipeline, search_parameters, X_train, y_train, k):
    vt = VarianceThreshold(threshold=0.0)
    X_train_var = vt.fit_transform(X_train)

    selector = SelectKBest(score_func=f_classif, k=min(k, X_train_var.shape[1]))
    X_train_sel = selector.fit_transform(X_train_var, y_train)

    clf = RandomizedSearchCV(pipeline,
                            param_distributions= search_parameters,
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