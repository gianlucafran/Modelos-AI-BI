import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

clientes_telco=pd.read_csv('https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv')
#Se realiza un resumen estadistico e identificamos si hay valores nulos
descripciones_datasets={}
def info_dataset(df, nombre="Dataset"):
    global descripciones_datasets
    print(f"\n{'='*20} ANÁLISIS: {nombre} {'='*20}")

    # 1. Dimensiones y Tipos
    print(f"\n[DIMENSIONES]: {df.shape[0]} filas x {df.shape[1]} columnas")

    # 2. Información General y Nulos
    # Combinamos info de nulos, tipos y memoria
    info_df = pd.DataFrame({
        'Tipo': df.dtypes,
        'Nulos': df.isnull().sum(),
        '% Nulos': (df.isnull().sum() / len(df) * 100).round(2),
        'Valores Únicos': df.nunique()
    })
    print("\n[RESUMEN DE COLUMNAS]:")
    print(info_df)

    # 3. Estadísticas Numéricas
    if not df.select_dtypes(include=['number']).empty:
        print("\n[ESTADÍSTICAS NUMÉRICAS]:")
        print(df.describe().T) # .T (transponer) lo hace más fácil de leer

    descripcion_df=df.describe()
    descripciones_datasets[nombre] = descripcion_df
    print(f"\n{'='*50}\n")
    return descripcion_df

#utilizamos la función creada:
info_dataset(clientes_telco, 'Telco')

variables_categoricas=['gender','SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']

for variable in variables_categoricas:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=clientes_telco, x=variable, y='MonthlyCharges') # Cambia 'MonthlyCharges' por tu variable numérica de interés
    plt.title(f'Distribución de MonthlyCharges por {variable}')
    plt.show()

for variable in variables_categoricas:
    plt.figure(figsize=(10, 10))

    # Contamos cuántas veces aparece cada valor
    data = clientes_telco[variable].value_counts()

    # Generamos el gráfico de pastel
    data.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False)

    plt.title(f'Distribución de {variable}')
    plt.ylabel('') # Quitamos la etiqueta del eje Y que suele estorbar
    plt.show()


sns.pairplot(clientes_telco.select_dtypes(include=['number']),kind='kde',corner='True')
plt.title('Relaciones entre variables númericas')
plt.show()

clientes_telco.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.show()
variables_a_analizar = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Partner', 'Contract'] # Ajusta esta lista a tu gusto

etiquetas_partner = {'No': 0, 'Yes': 1}
etiquetas_multiple_lines = {'No': 0, 'Yes': 1, 'No phone service': 2}
etiquetas_internet_service = {'No': 0, 'Yes': 1, 'No internet service': 2, 'DSL': 3, 'Fiber optic': 4}
etiquetas_contract = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
etiquetas_paymentmethod = {
    'Mailed check': 0,
    'Bank transfer (automatic)': 1,
    'Credit card (automatic)': 2,
    'Electronic check': 3
}

# Diccionario maestro para el ciclo
mapeos = {
    'gender':{'Female': 0, 'Male': 1},
    'Partner': etiquetas_partner,
    'Dependents': etiquetas_partner,
    'PhoneService': etiquetas_partner,
    'PaperlessBilling': etiquetas_partner,
    'MultipleLines': etiquetas_multiple_lines,
    'InternetService': etiquetas_internet_service,
    'OnlineSecurity': etiquetas_internet_service,
    'OnlineBackup': etiquetas_internet_service,
    'DeviceProtection': etiquetas_internet_service,
    'TechSupport': etiquetas_internet_service,
    'StreamingTV': etiquetas_internet_service,
    'StreamingMovies': etiquetas_internet_service,
    'Contract': etiquetas_contract,
    'PaymentMethod': etiquetas_paymentmethod,
    'Churn': etiquetas_partner
}

variables_a_etiquetar=['gender','Partner', 'Dependents',

       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',

       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',

       'PaymentMethod', 'Churn']

#creamos una copia del df original para trabajar con uno nuevo
clientes_telco_etiquetado=clientes_telco.copy()
for variable in variables_a_etiquetar:
    if variable in mapeos:
        clientes_telco_etiquetado[variable] = clientes_telco_etiquetado[variable].map(mapeos[variable])
        print(f"Variable '{variable}' procesada.")
    else:
        print(f"Saltando '{variable}' (no requiere mapeo o es numérica).")
        
variables_a_analizar = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

for variable in variables_a_analizar:
    plt.figure(figsize=(8, 4))

    # Usamos kdeplot para el gráfico de densidad
    # Si la variable es numérica, el gráfico será suave.
    # Si es categórica (etiquetada como números), se verá como picos en esos valores.
    sns.kdeplot(data=clientes_telco_etiquetado, x=variable, fill=True, color="skyblue")

    plt.title(f'Distribución de densidad de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Densidad')
    plt.show()
    
    
    
matriz_covarianza=clientes_telco_etiquetado.select_dtypes(include=['number']).cov()
plt.figure(figsize=(15,10))
sns.heatmap(matriz_covarianza, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Covarianza')
plt.show()

sol = matriz_covarianza.where(np.triu(np.ones(matriz_covarianza.shape), k=1).astype(bool))
sol = sol.stack().reset_index()
sol.columns = ['Variable 1', 'Variable 2', 'Covarianza']

top_5_covarianza = sol.sort_values(by='Covarianza', ascending=False).head(5)
print("Top 5 relaciones con mayor covarianza:")
print(top_5_covarianza.to_string(index=False))

matriz_pearson=clientes_telco_etiquetado.select_dtypes(include=['number']).corr()
plt.figure(figsize=(15,15))
sns.heatmap(matriz_pearson, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Pearson')
plt.show()
upper_tri = matriz_pearson.where(np.triu(np.ones(matriz_pearson.shape), k=1).astype(bool))

pearson_stack = upper_tri.stack().reset_index()
pearson_stack.columns = ['Variable 1', 'Variable 2', 'Correlacion_Pearson']

top_5_pearson = pearson_stack.iloc[pearson_stack['Correlacion_Pearson'].abs().argsort()[::-1]].head(5)

print("Top 5 relaciones por Correlación de Pearson:")
print(top_5_pearson.to_string(index=False))

matriz_spearman=clientes_telco_etiquetado.select_dtypes(include=['number']).corr(method='spearman')
plt.figure(figsize=(15,15))
sns.heatmap(matriz_spearman, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matriz de Spearman')
plt.show()
upper_tri_spearman = matriz_spearman.where(np.triu(np.ones(matriz_spearman.shape), k=1).astype(bool))

spearman_stack = upper_tri_spearman.stack().reset_index()
spearman_stack.columns = ['Variable 1', 'Variable 2', 'Correlacion_Spearman']

top_5_spearman = spearman_stack.iloc[spearman_stack['Correlacion_Spearman'].abs().argsort()[::-1]].head(5)

print("Top 5 relaciones por Correlación de Spearman (No Lineal/Monótona):")
print(top_5_spearman.to_string(index=False))

#ingeniería de caracteristicas
#¿Qué tanto representa el cargo mensual sobre el total pagado?
clientes_telco_etiquetado['Ratio_Cargo_Total'] = (
    clientes_telco_etiquetado['MonthlyCharges'] / (clientes_telco_etiquetado['TotalCharges'] + 1)
)

# Crear ratio: ¿Qué tan fiel es el cliente según su tenure?
clientes_telco_etiquetado['Costo_Mensual_Promedio'] = (
    clientes_telco_etiquetado['TotalCharges'] / (clientes_telco_etiquetado['tenure'] + 1)
)
#categorías de fidelidad
bins = [0, 12, 24, 48, 72]
labels = ['Nuevo', 'Fiel_Corto', 'Fiel_Medio', 'Veterano']
clientes_telco_etiquetado['Fidelidad_Segmentada'] = pd.cut(
    clientes_telco_etiquetado['tenure'], bins=bins, labels=labels, include_lowest=True
)
clientes_telco_etiquetado = pd.get_dummies(clientes_telco_etiquetado, columns=['Fidelidad_Segmentada'])
clientes_telco_etiquetado.head()

#importamos librerías para normalizar los datos
from sklearn.preprocessing import StandardScaler

# 1. Definimos las variables a normalizar (Nota el doble corchete: [[]])
variables_a_normalizar = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 2. Hacemos copia del df para no perder los datos originales
clientes_telco_normalizado = clientes_telco_etiquetado.copy()

# 3. Aplicamos el StandardScaler
normalizador = StandardScaler()
clientes_telco_normalizado[variables_a_normalizar] = normalizador.fit_transform(clientes_telco_normalizado[variables_a_normalizar])

# 4. Verificación técnica
print("Variables normalizadas (Media ~0, Desviación ~1):")
print(clientes_telco_normalizado[variables_a_normalizar].describe().loc[['mean', 'std']])
clientes_telco_normalizado.head()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Normalización de variables
variables_a_normalizar = ['tenure', 'MonthlyCharges', 'TotalCharges']
clientes_telco_normalizado = clientes_telco_etiquetado.copy()

normalizador = StandardScaler()
clientes_telco_normalizado[variables_a_normalizar] = normalizador.fit_transform(
    clientes_telco_normalizado[variables_a_normalizar]
)

# 2. Separación de datos (X e y)
X = clientes_telco_normalizado.drop(['Churn', 'customerID'], axis=1, errors='ignore')
y = clientes_telco_normalizado['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Instanciación y entrenamiento
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# Predicción
y_pred_rf = modelo_rf.predict(X_test)

# Evaluación y Matriz de Confusión
print("--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_rf))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_rf))
