# Evaluación de modelos de aprendizaje para la clasificación de tipos de tumor

Colaboradores: 

Edna Karen Rivera Zagal

Karla Ximena González Platas

Erika Nathalia Ordoñez Guzmán 

![Clasificación de tipos tumorales](https://criscancer.org/wp-content/themes/cris_template/assets/images/tipos-de-cancer1.jpg)

El presente proyecto busca evaluar el desempeños de 3 diferentes tipos de aprendizaje 
auntomático; uno de aprendizaje y dos de aprendizaje clásico: un perceptrón multicapa, MLP, 
Máquinas 
de Vectores de Soporte, SVM y Bosque aleatorios, RF, respectivamente. La MLP, es muy flexible y 
potente para problemas complejos, mientras que el SVM, es eficaz en datos de alta dimensión y
predice de forma acertada cuando hay un margen de separación claro y el RF es robusto y versátil, 
que sobresale en la captura de relaciones no lineales. Se entrenaron los algoritmos 
en la tarea de clasificación de tipos de cáncer (cancer de mama, BRCA, carcinoma de células renales, 
KIRC, adenocarcinoma de colon, COAD, adnocarcinoma de pulmón, LUAD y adenocarcinoma de próstata, 
PRAD) con datos de RNA-seq. Para ello, se descargaron los datos de expresión génica de muestras de 
cáncer de https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq.

Este repositorio, posee:

* Una carpeta llamada "Data", posee los archivos "data.csv" y "labels.csv". Estos archivos son
necesarios para el entrenamiento de todos los modelos. El archivo "data.csv" los datos de
expresión y las muestras, por otro lado, el archivo "labels.csv" contiene las clases, es decir,
los tipos tumorales.

* También contiene una carpeta "results" que a su vez se divide en subcarpetas llamadas
"ML_results", "RF_results" y "SVM_results" que contienen archivos csv con las métricas y
gráficas correspondientes a cada algoritmo.

* La carpeta "src" contiene todos los códigos, en el caso del algoritmo SVM, tiene 3 archivos
relacionados.

Se utilizó el lenguaje de programación de Python 3.10 en dos entornos de trabajo: Jupyter 
Notebook para los modelos MLP y Random Forest, y scripts en Python para la implementación 
modular del SVM. El desarrollo se realizó con las librerías scikit-learn, NumPy y pandas para 
el procesamiento de datos, y Matplotlib para las visualizaciones.

