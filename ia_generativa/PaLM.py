"""
Búsqueda Semántica y Resumen de Textos con Gemini API
=======================================================

Implementación moderna basada en Gemini API (sucesor de PaLM) para:
- Búsqueda semántica usando embeddings vectoriales
- Resumen automático de documentos
- Dataset 20 newsgroups de scikit-learn

Gemini Embedding vs PaLM:
✓ Gemini-Embedding-2: Multimodal, hasta 3072 dimensiones
✓ Mayor precisión en búsqueda semántica
✓ Soporte para 100+ idiomas
✓ Mejor rendimiento que PaLM legacy

Documentación: https://ai.google.dev/gemini-api/docs/embeddings
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from typing import List, Dict, Tuple
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================

print("=" * 80)
print("Búsqueda Semántica y Resumen con Gemini API")
print("=" * 80)

# Configurar API Key
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("\n⚠️ GEMINI_API_KEY no está configurada")
    print("   Pasos para obtenerla:")
    print("   1. Ve a: https://ai.google.dev")
    print("   2. Haz clic en 'Get API Key'")
    print("   3. Copia la clave y configura: export GEMINI_API_KEY='tu-clave'")
    print("\n   Continuando con demostración de arquitectura...")
    API_KEY = "demo-key-for-testing"
    DEMO_MODE = True
else:
    genai.configure(api_key=API_KEY)
    DEMO_MODE = False

print(f"\n📱 Modo de operación: {'DEMOSTRACIÓN' if DEMO_MODE else 'PRODUCCIÓN'}")

# Configuración
EMBEDDING_MODEL = "models/embedding-001"  # Gemini Embedding 1
SUMMARY_MODEL = "models/gemini-pro"       # Para resúmenes
BATCH_SIZE = 100
TOP_K = 5  # Resultados similares a retornar

print(f"\n⚙️ Configuración:")
print(f"   Modelo de embeddings: {EMBEDDING_MODEL}")
print(f"   Modelo de resumen: {SUMMARY_MODEL}")
print(f"   Top-K resultados: {TOP_K}")

# ============================================================================
# 2. CARGA DEL DATASET
# ============================================================================

print("\n" + "=" * 80)
print("📂 CARGANDO DATASET 20 NEWSGROUPS")
print("=" * 80)

# Seleccionar categorías
categories = [
    'sci.med',
    'sci.space',
    'comp.graphics',
    'talk.politics.guns'
]

print(f"\n⏳ Descargando dataset...")
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    random_state=42
)

# Preparar datos
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(newsgroups.target)

# Limitar a primeros N documentos para demostración
MAX_DOCS = 50  # Para no saturar la API
texts = newsgroups.data[:MAX_DOCS]
doc_labels = labels[:MAX_DOCS]
doc_categories = [label_encoder.classes_[l] for l in doc_labels]

print(f"\n✅ Dataset cargado:")
print(f"   Documentos: {len(texts)}")
print(f"   Categorías: {len(categories)}")

print(f"\n📰 Categorías:")
for cat, count in zip(label_encoder.classes_, 
                      [sum(doc_labels == i) for i in range(len(label_encoder.classes_))]):
    print(f"   • {cat}: {count} documentos")

# ============================================================================
# 3. FUNCIONES DE EMBEDDINGS (BÚSQUEDA SEMÁNTICA)
# ============================================================================

def get_embeddings(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """
    Obtiene embeddings usando Gemini Embedding API
    
    Args:
        texts: Lista de textos
        task: RETRIEVAL_DOCUMENT o RETRIEVAL_QUERY
    
    Returns:
        Array de embeddings (n_texts x embedding_dim)
    """
    if DEMO_MODE:
        # Simulación para demostración
        print(f"   [DEMO] Generando {len(texts)} embeddings (384 dimensiones)...")
        np.random.seed(42)
        embeddings = np.random.randn(len(texts), 384)
        # Normalizar
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    embeddings = []
    print(f"   Procesando {len(texts)} textos...")
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        
        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type=task
            )
            embeddings.extend([e for e in response['embedding']])
            print(f"   ✓ Procesados {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
        except Exception as e:
            print(f"   ❌ Error en lote {i//BATCH_SIZE}: {str(e)}")
            return None
    
    return np.array(embeddings)

# ============================================================================
# 4. BÚSQUEDA SEMÁNTICA
# ============================================================================

class BuscadorSemantico:
    """Motor de búsqueda semántica basado en embeddings"""
    
    def __init__(self, documentos: List[str], metadatos: Dict = None):
        """
        Inicializa el buscador
        
        Args:
            documentos: Lista de textos
            metadatos: Información adicional (categorías, etc)
        """
        self.documentos = documentos
        self.metadatos = metadatos or {}
        self.embeddings = None
        self._build_index()
    
    def _build_index(self):
        """Construye índice de embeddings"""
        print("\n🔨 Construyendo índice de búsqueda...")
        self.embeddings = get_embeddings(
            self.documentos,
            task="RETRIEVAL_DOCUMENT"
        )
        print("   ✅ Índice creado")
    
    def buscar(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        Busca documentos similares a la consulta
        
        Args:
            query: Texto de búsqueda
            top_k: Número de resultados
        
        Returns:
            Lista de documentos similares con puntuaciones
        """
        # Obtener embedding de la consulta
        query_embedding = get_embeddings(
            [query],
            task="RETRIEVAL_QUERY"
        )
        
        if query_embedding is None:
            return []
        
        # Calcular similitud coseno
        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]
        
        # Obtener top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        resultados = []
        for idx in top_indices:
            resultados.append({
                'indice': int(idx),
                'similitud': float(similarities[idx]),
                'documento': self.documentos[idx][:300] + "..." if len(self.documentos[idx]) > 300 else self.documentos[idx],
                'categoria': self.metadatos.get('categorias', [None])[idx] if 'categorias' in self.metadatos else None
            })
        
        return resultados

# ============================================================================
# 5. FUNCIONES DE RESUMEN
# ============================================================================

def resumir_texto(texto: str, max_sentences: int = 3) -> str:
    """
    Resume un texto usando Gemini
    
    Args:
        texto: Texto a resumir
        max_sentences: Máximo de oraciones en el resumen
    
    Returns:
        Resumen del texto
    """
    if DEMO_MODE:
        # Simulación de resumen
        oraciones = texto.split('.')[:max_sentences]
        return '.'.join(oraciones) + '.'
    
    try:
        prompt = f"""Proporciona un resumen en máximo {max_sentences} oraciones del siguiente texto:

{texto}

Resumen:"""
        
        response = genai.generate_text(prompt=prompt)
        return response.result if response.result else "No se pudo generar resumen"
    
    except Exception as e:
        return f"Error en resumen: {str(e)}"

# ============================================================================
# 6. EJECUCIÓN - BÚSQUEDA SEMÁNTICA
# ============================================================================

print("\n" + "=" * 80)
print("🔍 BÚSQUEDA SEMÁNTICA")
print("=" * 80)

# Crear buscador
buscador = BuscadorSemantico(
    documentos=texts,
    metadatos={'categorias': doc_categories}
)

# Ejemplos de búsqueda
queries = [
    "Investigación en medicina moderna y tratamientos",
    "Viajes espaciales y exploración del universo",
    "Gráficos por computadora y visión artificial"
]

for i, query in enumerate(queries, 1):
    print(f"\n{'='*80}")
    print(f"Búsqueda {i}: '{query}'")
    print(f"{'='*80}")
    
    resultados = buscador.buscar(query, top_k=3)
    
    for j, resultado in enumerate(resultados, 1):
        print(f"\n📄 Resultado {j}")
        print(f"   Similitud: {resultado['similitud']:.2%}")
        print(f"   Categoría: {resultado['categoria']}")
        print(f"   Fragmento: {resultado['documento'][:150]}...")

# ============================================================================
# 7. RESUMEN DE DOCUMENTOS
# ============================================================================

print("\n" + "=" * 80)
print("📝 RESUMEN DE DOCUMENTOS")
print("=" * 80)

for i, (texto, categoria) in enumerate(list(zip(texts, doc_categories))[:3], 1):
    print(f"\n{'='*80}")
    print(f"Documento {i} - Categoría: {categoria}")
    print(f"{'='*80}")
    
    # Mostrar original (truncado)
    print(f"\n📖 Texto original (primeros 300 caracteres):")
    print(f"   {texto[:300]}...")
    
    # Generar resumen
    print(f"\n⏳ Generando resumen...")
    resumen = resumir_texto(texto, max_sentences=2)
    print(f"\n✅ Resumen:")
    print(f"   {resumen}")

# ============================================================================
# 8. ANÁLISIS DE BÚSQUEDA SEMÁNTICA
# ============================================================================

print("\n" + "=" * 80)
print("📊 ANÁLISIS DE BÚSQUEDA SEMÁNTICA")
print("=" * 80)

if buscador.embeddings is not None:
    # Calcular similitudes entre documentos
    similaridades = cosine_similarity(buscador.embeddings)
    
    # Estadísticas
    print(f"\n📈 Estadísticas de embeddings:")
    print(f"   Dimensión: {buscador.embeddings.shape[1]}")
    print(f"   Documentos indexados: {buscador.embeddings.shape[0]}")
    
    # Documentos más similares (sin contar diagonal)
    print(f"\n   Similitud entre documentos:")
    for i in range(len(similaridades)):
        # Obtener índices más similares (excluyendo el documento mismo)
        top_similar = np.argsort(similaridades[i])[::-1][1:3]
        for j in top_similar:
            sim = similaridades[i][j]
            if sim > 0.5:  # Solo mostrar si similitud > 0.5
                print(f"   Doc {i} ↔ Doc {j}: {sim:.2%} | {doc_categories[i]} ↔ {doc_categories[j]}")

# ============================================================================
# 9. BÚSQUEDA CON FILTRADO POR CATEGORÍA
# ============================================================================

print("\n" + "=" * 80)
print("🔎 BÚSQUEDA FILTRADA POR CATEGORÍA")
print("=" * 80)

query = "tecnología médica avanzada"
print(f"\nConsulta: '{query}'")
print(f"Filtrando por: 'sci.med'")

resultados_all = buscador.buscar(query, top_k=10)
resultados_filtrados = [
    r for r in resultados_all 
    if r['categoria'] == 'sci.med'
][:3]

print(f"\n✅ Resultados en categoría 'sci.med':")
for i, resultado in enumerate(resultados_filtrados, 1):
    print(f"\n   {i}. Similitud: {resultado['similitud']:.2%}")
    print(f"      {resultado['documento'][:100]}...")

# ============================================================================
# 10. TABLA COMPARATIVA - BÚSQUEDA KEYWORD vs SEMÁNTICA
# ============================================================================

print("\n" + "=" * 80)
print("📊 COMPARACIÓN: BÚSQUEDA KEYWORD vs SEMÁNTICA")
print("=" * 80)

print(f"""
┌─────────────────────┬──────────────────────┬──────────────────────┐
│      Criterio       │   Búsqueda Keyword   │  Búsqueda Semántica  │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Coincidencia exacta │ ✓ Muy precisa        │ ✗ Basada en contexto │
│ Sinónimos          │ ✗ No encuentra       │ ✓ Entiende           │
│ Contexto           │ ✗ Ignora             │ ✓ Analiza            │
│ Typos              │ ✗ No maneja          │ ✓ Tolera             │
│ Idioma natural     │ ✗ Limitado           │ ✓ Completo           │
│ Velocidad          │ ✓ Muy rápida         │ ~ Moderada           │
│ Uso de embeddings  │ ✗ No                 │ ✓ Sí (Gemini)        │
└─────────────────────┴──────────────────────┴──────────────────────┘
""")

# ============================================================================
# 11. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("📋 RESUMEN")
print("=" * 80)

summary_text = f"""
✅ Implementación completada: Gemini API para Búsqueda Semántica y Resumen

📊 Resultados del análisis:
   • Dataset: 20 Newsgroups ({len(texts)} documentos)
   • Modelo: Gemini Embedding ({buscador.embeddings.shape[1] if buscador.embeddings is not None else 384} dimensiones)
   • Modo: {'DEMOSTRACIÓN' if DEMO_MODE else 'PRODUCCIÓN con API Real'}

🔑 Características implementadas:
   ✓ Búsqueda semántica con embeddings vectoriales
   ✓ Similitud coseno entre documentos
   ✓ Resumen automático de textos
   ✓ Filtrado por metadatos (categoría)
   ✓ Análisis de similitud entre documentos

💡 Ventajas de Gemini vs PaLM Legacy:
   • Gemini-Embedding-2: Multimodal (texto, imagen, video)
   • Mayor dimensionalidad: hasta 3072 vs 768 de PaLM
   • Soporte para 100+ idiomas (PaLM limitado)
   • API moderna con mejor documentación
   • Mejor rendimiento en búsqueda semántica

🔗 Casos de uso prácticos:
   • Motor de búsqueda para bases de datos documentales
   • Sistema de recomendaciones basado en contenido
   • Deduplicación inteligente de documentos
   • Resumen automático de artículos/noticias
   • Clasificación semántica de textos

📈 Próximos pasos:
   1. Configurar GEMINI_API_KEY para usar API real
   2. Aumentar MAX_DOCS para más documentos
   3. Implementar chunking para documentos muy largos
   4. Usar gemini-embedding-2 para multimodalidad
   5. Indexar en base de datos vectorial (Pinecone, Milvus, etc)

🔗 Documentación oficial:
   • Gemini API: https://ai.google.dev/gemini-api
   • Embeddings: https://ai.google.dev/gemini-api/docs/embeddings
   • Migración desde PaLM: https://ai.google.dev/migrate/gemini-api
"""

print(summary_text)

print("=" * 80)
print("¡Proceso completado exitosamente!")
print("=" * 80)