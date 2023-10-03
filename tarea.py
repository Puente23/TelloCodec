import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos desde el archivo CSV
df = pd.read_csv('C:/Users/puent/Downloads/Crimes_2022.csv')

# Seleccionar las características (atributos) que se utilizarán en la clasificación
X = df[['X Coordinate', 'Y Coordinate', 'Year']]

# Convertir la variable objetivo (Arrest) en una variable binaria (0 o 1)
df['Arrest'] = df['Arrest'].map({'false': 0, 'true': 1})

# Llenar valores faltantes en X con la media
X = X.fillna(X.mean())

# Definir la variable objetivo
y = df['Arrest']

# Llenar valores faltantes en y con la media
y = y.fillna(y.mean())

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo de Bayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = classifier.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

# Aplicar validación cruzada
scores = cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy')

print(f'Precisión del modelo: {accuracy:.2f}')
print(f'Precisión del modelo (validación cruzada): {scores.mean():.2f}')
