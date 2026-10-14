from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

# 2. Convertir a DataFrame con los nombres de las columnas
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

import seaborn as sns
sns.pairplot(df,  corner='True')

# 1. Configurar y entrenar K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(df.iloc[:, :4])

# 2. Mostrar centroides
print("Centroides de los clusters:")
print(pd.DataFrame(kmeans.cluster_centers_, columns=iris.feature_names))

# 3. Visualización rápida de los resultados (tomando las dos primeras variables)
plt.figure(figsize=(8, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x', label='Centroides')
plt.title('Visualización de Clusters K-Means (Iris Dataset)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.show()