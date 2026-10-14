"""
Naive Bayes: Filtro de Spam (Clasificación de Texto)
Demostración con dataset '20 Newsgroups' de scikit-learn (Adaptado a Spam/Ham)

Conceptos:
- Teorema de Bayes: Calcula la probabilidad de que un correo sea Spam basándose en 
  las palabras que contiene (Probabilidad Condicional).
- Naive (Ingenuo): Asume ingenuamente que la presencia de una palabra es completamente 
  independiente de la presencia de otra (ej. "oferta" y "gratis" se calculan por separado).
- MultinomialNB: Variante de Naive Bayes diseñada específicamente para variables 
  discretas como el conteo de palabras o frecuencias (TF-IDF) en clasificación de texto.
- Vectorización (TF-IDF): Transforma el texto crudo en una matriz numérica de características, 
  dando más peso a palabras importantes y menos a conectores comunes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. CARGAR Y ADAPTAR EL DATASET (SIMULACIÓN SPAM VS HAM)
# ============================================================================
print("=" * 80)
print("PASO 1: CARGAR DATASET DE TEXTO (SCIKIT-LEARN 20 NEWSGROUPS)")
print("=" * 80)

# Scikit-learn no tiene un dataset de "Spam" por defecto, pero tiene el famoso 
# '20 Newsgroups' (foros de noticias). 
# Adaptación: Usaremos el foro de ventas 'misc.forsale' como nuestra clase SPAM (anuncios)
# y el foro de ciencia 'sci.space' como nuestra clase HAM (correos legítimos).
print("Descargando/Cargando dataset (Puede tardar unos segundos la primera vez)...")

categorias = ['sci.space', 'misc.forsale']
dataset = fetch_20newsgroups(
    subset='all', 
    categories=categorias, 
    remove=('headers', 'footers', 'quotes'), # Removemos metadatos para que el modelo aprenda del texto real
    random_state=42
)

# Mapeamos a 0 (Ham/Legítimo) y 1 (Spam/Ventas)
# sci.space (0) -> Ham, misc.forsale (1) -> Spam
X_texto = dataset.data
y_etiquetas = dataset.target
target_names = ['Legítimo (Ham)', 'Spam (Ventas/Anuncios)']

df = pd.DataFrame({'Texto': X_texto, 'Clase': y_etiquetas})
df['Etiqueta'] = df['Clase'].map({0: 'Legítimo (Ham)', 1: 'Spam'})

# Limpiar correos vacíos generados por el remove de headers/quotes
df = df[df['Texto'].str.strip().astype(bool)]

print(f"\nDataset: 20 Newsgroups (Adaptado a Spam/Ham)")
print(f"Total de correos procesados: {df.shape[0]}")
print(f"Distribución:")
print(df['Etiqueta'].value_counts())

print(f"\nEjemplo de correo Legítimo (Ham):")
print("-" * 50)
print(df[df['Clase'] == 0]['Texto'].iloc[2][:300] + "...")

print(f"\nEjemplo de correo Spam (Ventas):")
print("-" * 50)
print(df[df['Clase'] == 1]['Texto'].iloc[2][:300] + "...")

# ============================================================================
# 2. VECTORIZACIÓN (TF-IDF) Y DIVISIÓN DE DATOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: PREPROCESAMIENTO Y VECTORIZACIÓN (TF-IDF)")
print("=" * 80)

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df['Texto'], df['Clase'], test_size=0.2, random_state=42, stratify=df['Clase']
)

# Transformar texto a números usando TF-IDF (Term Frequency-Inverse Document Frequency)
# Ignoramos palabras que aparecen en el >90% de los correos y las que aparecen en <5 correos
vectorizador = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)

print("Entrenando Vectorizador TF-IDF...")
X_train_vec = vectorizador.fit_transform(X_train)
X_test_vec = vectorizador.transform(X_test)

vocabulario_size = len(vectorizador.get_feature_names_out())
print(f"✓ Texto convertido a matriz numérica.")
print(f"✓ Tamaño del vocabulario aprendido: {vocabulario_size} palabras únicas.")
print(f"  (Cada correo ahora es un vector de {vocabulario_size} dimensiones)")

# ============================================================================
# 3. ENTRENAMIENTO DEL MODELO NAIVE BAYES
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: ENTRENAR EL CLASIFICADOR MULTINOMIAL NAIVE BAYES")
print("=" * 80)

print("Inicializando modelo MultinomialNB...")
# alpha=1.0 es el Suavizado de Laplace (evita probabilidades de cero si una 
# palabra nueva aparece en el futuro).
modelo_nb = MultinomialNB(alpha=1.0)

print("Entrenando el modelo calculando probabilidades condicionales...")
modelo_nb.fit(X_train_vec, y_train)

print("✓ Modelo entrenado exitosamente.")

# ============================================================================
# 4. EVALUACIÓN Y MÉTRICAS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: PREDICCIÓN Y EVALUACIÓN DEL RENDIMIENTO")
print("=" * 80)

# Predicciones en el conjunto de prueba
y_pred = modelo_nb.predict(X_test_vec)

exactitud = accuracy_score(y_test, y_pred)
print(f"\nExactitud (Accuracy) del Filtro: {exactitud * 100:.2f}%")

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, 
                   index=[f'Real {target_names[0]}', f'Real {target_names[1]}'], 
                   columns=[f'Pred {target_names[0]}', f'Pred {target_names[1]}']))

print("\nFalsos Positivos (Correos buenos mandados a Spam por error):", cm[0, 1])
print("Falsos Negativos (Spam que se coló a la bandeja de entrada):", cm[1, 0])

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================================================
# 5. VISUALIZACIÓN: PALABRAS CLAVE Y MATRIZ DE CONFUSIÓN
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: VISUALIZACIÓN DE MATRIZ DE CONFUSIÓN Y PALABRAS CLAVE")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico 1: Matriz de Confusión Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names, 
            linewidths=1, linecolor='black', ax=axes[0])
axes[0].set_title('Matriz de Confusión del Filtro de Spam', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicción del Modelo')
axes[0].set_ylabel('Clase Real')

# Gráfico 2: Palabras más indicativas de Spam vs Ham
# Naive Bayes calcula log-probabilidades empíricas de características dado un clase
log_probs_ham = modelo_nb.feature_log_prob_[0]
log_probs_spam = modelo_nb.feature_log_prob_[1]

# Encontrar el Top 10 palabras por clase (las que tienen mayor probabilidad logarítmica)
vocabulario = np.array(vectorizador.get_feature_names_out())

top_indices_ham = np.argsort(log_probs_ham)[-10:]
top_indices_spam = np.argsort(log_probs_spam)[-10:]

# Diferencia de log-probabilidades (Palabras que gritan "SPAM" o "HAM" fuertemente)
diferencia = log_probs_spam - log_probs_ham
top_spam_words_idx = np.argsort(diferencia)[-10:]
top_ham_words_idx = np.argsort(diferencia)[:10]

palabras_clave = np.concatenate([vocabulario[top_ham_words_idx], vocabulario[top_spam_words_idx]])
pesos = np.concatenate([diferencia[top_ham_words_idx], diferencia[top_spam_words_idx]])
colores = ['forestgreen'] * 10 + ['crimson'] * 10

axes[1].barh(range(20), pesos, color=colores)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(palabras_clave)
axes[1].set_title('Top Palabras Clave: Legítimo (Verde) vs Spam (Rojo)\n(Según Log-Probabilidad de Bayes)', 
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Peso hacia Spam (Positivo) o Legítimo (Negativo)')

# Etiqueta explicativa
axes[1].text(0.05, 0.95, "Palabras como 'sale',\n'offer', 'condition'\ninclinan la balanza a Spam.", 
             transform=axes[1].transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
#plt.savefig('/home/usuario/01_naive_bayes_spam.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_naive_bayes_spam.png")
plt.close()

# ============================================================================
# 6. SISTEMA EN ACCIÓN: PRUEBA CON CORREOS NUEVOS
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: MOTOR DE FILTRADO EN TIEMPO REAL (PRUEBA)")
print("=" * 80)

def probar_filtro_spam(correos, modelo, vectorizador):
    """Función para probar correos crudos contra el filtro Naive Bayes entrenado."""
    print(f"{'='*60}")
    print(" BANDEJA DE ENTRADA (Clasificación en tiempo real)")
    print(f"{'='*60}")
    
    vecs = vectorizador.transform(correos)
    predicciones = modelo.predict(vecs)
    probabilidades = modelo.predict_proba(vecs)
    
    for i, correo in enumerate(correos):
        clase = "🚨 SPAM / ANUNCIO" if predicciones[i] == 1 else "✅ LEGÍTIMO (INBOX)"
        prob_spam = probabilidades[i][1] * 100
        
        # Recortar texto largo
        texto_mostrar = correo.replace('\n', ' ')[:80] + ("..." if len(correo) > 80 else "")
        
        print(f"\nAsunto/Cuerpo: \"{texto_mostrar}\"")
        print(f"Decisión: {clase} (Certeza de Spam: {prob_spam:.1f}%)")

# Textos inventados para probar el modelo
correos_prueba = [
    "Hello team, please remember that the launch of the new satellite is scheduled for tomorrow at NASA.", # Ham (sci.space)
    "FOR SALE: Excellent condition monitor! 50% discount. Please send me an email with your offer. Shipping included.", # Spam (misc.forsale)
    "I am doing a research on lunar orbit and gravity forces. Does anyone have a good physics book recommendation?", # Ham (sci.space)
    "BUY NOW CHEAP PRINTER! Great condition. $50 or best offer. Works perfectly." # Spam (misc.forsale)
]

probar_filtro_spam(correos_prueba, modelo_nb, vectorizador)

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   NAIVE BAYES PARA FILTRADO DE SPAM                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. FUNDAMENTO DEL ALGORITMO:
   • Naive Bayes calcula: P(Spam | Palabras) = P(Palabras | Spam) * P(Spam) / P(Palabras)
   • Aprende la frecuencia relativa de cada palabra en los correos Legítimos y 
     en el Spam. Al llegar un correo nuevo, multiplica estas probabilidades.

2. ¿POR QUÉ "INGENUO" (NAIVE)?:
   • El modelo asume que las palabras "Oferta" y "Especial" aparecen 
     independientemente la una de la otra. Aunque gramaticalmente esto es falso 
     (las palabras tienen contexto), matemáticamente este modelo "ingenuo" 
     funciona excepcionalmente bien para la clasificación de textos.

3. USO DEL DATASET DE SCIKIT-LEARN:
   • Se utilizó el módulo `fetch_20newsgroups` (restringido por la petición).
   • Adaptamos las clases 'misc.forsale' (anuncios de ventas/spam comercial) y 
     'sci.space' (discusiones científicas/legítimas) para emular el problema de Spam.

4. PREPROCESAMIENTO TF-IDF:
   • Vectorizar el texto es obligatorio. TF-IDF no solo cuenta cuántas veces 
     aparece una palabra, sino que castiga las palabras que aparecen en TODOS 
     los correos (como "the", "and") restándoles importancia predictiva.

5. RENDIMIENTO:
   • El modelo logró una Exactitud del {exactitud * 100:.2f}%.
   • Naive Bayes es muy rápido de entrenar, extremadamente eficiente en memoria 
     y suele ser la línea base (baseline) dorada para cualquier problema de NLP 
     antes de intentar arquitecturas complejas (como Transformers/BERT).

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

MultinomialNB(alpha=1.0):
  - alpha=1.0: "Suavizado de Laplace". Si en un correo nuevo entra la palabra 
    "Xilófono", la probabilidad condicional sería 0 (porque no estaba en el entreno), 
    lo que anularía toda la ecuación. Alpha suma '1' a todas las frecuencias 
    para evitar este error matemático.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_naive_bayes_spam.png - Matriz de Confusión del modelo y un análisis visual 
                          de las 10 palabras con mayor peso probabilístico hacia 
                          Spam y hacia Ham.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Entrenamiento del Filtro de Spam Naive Bayes completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")