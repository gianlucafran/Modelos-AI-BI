# Repositorio de Modelos de Inteligencia Artificial

Repositorio de referencia para organizar modelos, algoritmos y herramientas de Inteligencia Artificial en cuatro categorías: **Aprendizaje Automático, Visión Computacional, IA Generativa y Procesamiento de Lenguaje Natural**.

El propósito de este README es permitir la consulta rápida de cada implementación, su función o aplicación principal, el tipo de entrada que utiliza, sus dependencias y las instrucciones para ejecutarla. La documentación se completará según el código disponible; no se asumirán datasets, librerías ni requisitos que no hayan sido verificados.

## Tabla de contenidos

- [Aprendizaje Automático](#aprendizaje-automatico)
- [Visión Computacional](#vision-computacional)
- [IA Generativa](#ia-generativa)
- [Procesamiento de Lenguaje Natural](#procesamiento-de-lenguaje-natural)
- [Requisitos generales](#requisitos-generales)
- [Instalación](#instalación)
- [Cómo ejecutar una implementación](#cómo-ejecutar-una-implementación)
- [Ficha individual de cada modelo](#ficha-individual-de-cada-modelo)
- [Notas de documentación](#notas-de-documentación)

## Estructura del repositorio

La siguiente estructura es una propuesta. Los nombres de carpetas y archivos deben ajustarse a los que existan realmente en el repositorio.

```text
AI-Models/
├── README.md
├── requirements.txt # Si se mantienen dependencias comunes
├── aprendizaje_automatico/
│   ├── kmeans.py
│   ├── clustering_jerarquico.py
│   ├── DBSCAN.py
│   ├── OPTICS.py
│   ├── Clustering_espectral.py
│   ├── PCA.py
│   ├── SVD.py
│   ├── t-SNE.py
│   ├── UMAP.py
│   ├── Isolation_Forest.py
│   ├── One-Class SVM.py
│   ├── Autoencoders.py
│   ├── GMM.py
│   ├── apriori.py
│   ├── DQN.py
│   ├── SARSA.py
│   ├── Policy Gradients & Policy Search.py
│   ├── PPO.py
│   ├── Actor-Critic.py
│   ├── regresion_lineal.py
│   ├── regresion_logistica.py
│   ├── K-NN.py
│   ├── Naive Bayes.py
│   ├── SVM.py
│   ├── LDA.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── Gradient_Boosting.py
│   ├── gbm.py
│   ├── xgboost.py
│   ├── lightgbm.py
│   ├── catboost.py
│   ├── Regresion Ridge.py
│   ├── Regresión Lasso.py
│   ├── Elastic Net.py
│   └── GP.py
├── vision_computacional/
│   ├── OCR.py
│   ├── cnn.py
│   ├── Canny.py
│   ├── Laplacian of Gaussian.py
│   ├── SIFT.py
│   ├── Detector Harris.py
│   ├── SURF.py
│   ├── Brute-Force Matcher.py
│   ├── FLANN.py
│   ├── RANSAC.py
│   ├── AlexNet.py
│   ├── VGG16.py
│   ├── GoogLeNet.py
│   ├── ResNet.py
│   ├── Dense.py
│   ├── MobileNet.py
│   ├── R-CNN.py
│   ├── Fast R-CNN.py
│   ├── Faster R-CNN.py
│   ├── Cascade R-CNN.py
│   ├── YOLO.py
│   ├── SSD.py
│   ├── U-Net.py
│   ├── FPN.py
│   ├── PSPNet.py
│   ├── Mask R-CNN.py
│   ├── YOLACT.py
│   └── Vision Transformer.py
├── ia_generativa/
│   ├── PaLM.py
│   ├── ChatGPT-GPT-5.py
│   ├── Claude.py
│   ├── Gemini pro.py
│   ├── Gemini.py
│   ├── Perplexity AI.py
│   ├── Grok.py
│   ├── Midjourney.py
│   ├── DALL-E 3.py
│   ├── Stable Diffusion- Flux.py
│   ├── Nano.py
│   ├── Adobe Firefly Services.py
│   ├── LeonardoAi.py
│   ├── Luma Photon.py
│   ├── Seedream.py
│   ├── Canva Magic Media.py
│   ├── Runway Gen-3 Alpha.py
│   ├── Synthesia.py
│   ├── Colossyan.py
│   ├── Pictory AI.py
│   ├── DeepBrain AI.py
│   ├── InVideo.py
│   ├── Fliki AI.py
│   ├── Sora API.py
│   ├── HumataAI.py
│   ├── VAE.py
│   ├── GAN.py
│   └── Modelos de Difusión (DPM).py
└── procesamiento_lenguaje_natural/
    ├── bert.py
    ├── structbert.py
    ├── GPT3.py
    ├── chatbot.py
    ├── roberta.py
    ├── parsing.py
    ├── XLnet.py
    ├── RNN-LSTM.py
    └── Transformers.py
```

## Catálogo de modelos

Los siguientes modelos se han agrupado según la clasificación proporcionada. Las aplicaciones y entradas se muestran como referencia inicial; deben contrastarse con cada archivo antes de considerarlas documentación definitiva de una implementación concreta.

## Aprendizaje Automático

**Cantidad en la clasificación: 36 elementos.**

| Modelo | Aplicaciones generales / función de referencia | Entrada indicada | Archivo Python | Dependencias para ejecutar el script | Ejecución |
|---|---|---|---|---|---|
| K-Means | Segmentación estructural de la matriz productiva | CSV con variables numéricas continuas | `aprendizaje_automatico/kmeans.py` | `scikit-learn`, `matplotlib`, `pandas`, `seaborn` | `python aprendizaje_automatico/kmeans.py` |
| Agrupamiento Jerárquico | Taxonomías y agrupaciones biológicas | CSV con variables numéricas | `aprendizaje_automatico/clustering_jerarquico.py` | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy` | `python aprendizaje_automatico/clustering_jerarquico.py` |
| DBSCAN | Identificación de anomalías geoespaciales | CSV con coordenadas o distancias | `aprendizaje_automatico/DBSCAN.py` | `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn` (También emplea librerías nativas de Python como `json`, `typing` y `warnings`) | `python aprendizaje_automatico/DBSCAN.py` |
| OPTICS | Identificación de puntos de interés urbanos | CSV con coordenadas | `aprendizaje_automatico/OPTICS.py` | `numpy`, `matplotlib`, `scikit-learn` (También usa `warnings` nativo de Python) | `python aprendizaje_automatico/OPTICS.py` |
| Clustering Espectral | Segmentación de redes sociales | Matriz de afinidad (grafos) | `aprendizaje_automatico/Clustering_espectral.py` | `numpy`, `matplotlib`, `networkx`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/Clustering_espectral.py` |
| PCA | Análisis multivariado previo al clustering | CSV con múltiples variables correlacionadas | `aprendizaje_automatico/PCA.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/PCA.py` |
| SVD | Sistemas de recomendación básicos | Matriz dispersa o densa | `aprendizaje_automatico/SVD.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/SVD.py` |
| t-SNE | Exploración de espacios latentes | CSV de alta dimensionalidad | `aprendizaje_automatico/t-SNE.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También usa `time`, nativa de Python) | `python aprendizaje_automatico/t-SNE.py` |
| UMAP | Visualización de embeddings | CSV de alta dimensionalidad | `aprendizaje_automatico/UMAP.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `umap` (También utiliza módulos nativos como `time` y `sys`) | `python aprendizaje_automatico/UMAP.py` |
| Isolation Forest | Prevención de fraudes financieros | CSV con variables mixtas | `aprendizaje_automatico/Isolation_Forest.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza `warnings`, nativa de Python) | `python aprendizaje_automatico/Isolation_Forest.py` |
| One-Class SVM | Mantenimiento predictivo / Detección de anomalías industriales / Kernel RBF | CSV numérico normalizado y lecturas de sensores | `aprendizaje_automatico/One-Class SVM.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python aprendizaje_automatico/One-Class SVM.py` |
| Autoencoders | Detección de anomalías complejas | Vectores numéricos o imágenes | `aprendizaje_automatico/Autoencoders.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (Emplea también la librería nativa `warnings`) | `python aprendizaje_automatico/Autoencoders.py` |
| GMM | Segmentación de clientes híbridos | CSV numérico | `aprendizaje_automatico/GMM.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También usa `warnings` nativa de Python) | `python aprendizaje_automatico/GMM.py` |
| Apriori | Recomendaciones de carrito de compra | Datos transaccionales (Cestas) | `aprendizaje_automatico/apriori.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `mlxtend` (También emplea `warnings` y `sys` de forma nativa) | `python aprendizaje_automatico/apriori.py` |
| DQN | Juegos y control robótico discreto | Estado del entorno (Vectores) | `aprendizaje_automatico/DQN.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza módulos nativos como `collections`, `random` y `time`) | `python aprendizaje_automatico/DQN.py` |
| SARSA | Navegación en laberintos | Estado y acción del entorno | `aprendizaje_automatico/SARSA.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza módulos nativos de Python como `random` y `time`) | `python aprendizaje_automatico/SARSA.py` |
| Policy Gradients | Control de vuelo de drones | Estado continuo | `aprendizaje_automatico/Policy Gradients & Policy Search.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza `random` y `warnings`, librerías nativas de Python) | `python aprendizaje_automatico/Policy Gradients & Policy Search.py` |
| PPO | Alineación de LLMs (RLHF) | Estado y acciones mixtas | `aprendizaje_automatico/PPO.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza la librería nativa de Python `warnings`) | `python aprendizaje_automatico/PPO.py` |
| Actor-Critic | Robótica industrial continua | Vectores de estado del agente | `aprendizaje_automatico/Actor-Critic.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También utiliza módulos nativos como `math`, `random` y `warnings`) | `python aprendizaje_automatico/Actor-Critic.py` |
| Regresión Lineal | Predicción de precios inmobiliarios | CSV con variable dependiente continua | `aprendizaje_automatico/regresion_lineal.py` | `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `numpy` | `python aprendizaje_automatico/regresion_lineal.py` |
| Regresión Logística | Predicción de abandono (Churn) | CSV numérico / categórico | `aprendizaje_automatico/regresion_logistica.py` | `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn` | `python aprendizaje_automatico/regresion_logistica.py` |
| K-NN | Sistemas de recomendación simples | Vectores de características espaciales | `aprendizaje_automatico/K-NN.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (Emplea también el módulo nativo `warnings`) | `python aprendizaje_automatico/K-NN.py` |
| Naive Bayes | Filtro de Spam | Frecuencias de palabras / conteos | `aprendizaje_automatico/Naive Bayes.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` (También emplea el módulo nativo `warnings`) | `python aprendizaje_automatico/Naive Bayes.py` |
| SVM | Categorización de texto y biometría | Vectores continuos de alta dimensión | `aprendizaje_automatico/SVM.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/SVM.py` |
| LDA | Análisis de riesgo crediticio | Reducción de dimensionalidad supervisada (1D) | `aprendizaje_automatico/LDA.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/LDA.py` |
| Árboles de Decisión | Diagnóstico médico guiado: clasificación de tumores de mama como benignos o malignos | Clasificación binaria supervisada e interpretación de características | `aprendizaje_automatico/decision_tree.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python aprendizaje_automatico/decision_tree.py` |
| Random Forest | Predicción de abandono de clientes de telecomunicaciones (Churn) | Clasificación binaria supervisada | `aprendizaje_automatico/random_forest.py` | `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` | `python aprendizaje_automatico/random_forest.py` |
| Gradient Boosting | Diagnóstico médico / Predicción general | Tabular Cuantitativo Continuo | `aprendizaje_automatico/Gradient_Boosting.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/Gradient_Boosting.py` |
| GBM | Predicción de Fuga de Clientes (Churn) | Clasificación Supervisada con Ensamble Secuencial | `aprendizaje_automatico/gbm.py` | `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn` | `python aprendizaje_automatico/gbm.py` |
| XGBoost | Modelos financieros predictivos / Churn | CSV estructurado o matrices XGB | `aprendizaje_automatico/xgboost.py` | `pandas`, `matplotlib`, `seaborn`, `numpy`, `xgboost`, `scikit-learn` | `python aprendizaje_automatico/xgboost.py` |
| LightGBM | Predicción de Fuga de Clientes (Churn) | Clasificación Supervisada con Ensamble Eficiente (Leaf-wise) | `aprendizaje_automatico/lightgbm.py` | `pandas`, `matplotlib`, `seaborn`, `numpy`, `lightgbm`, `scikit-learn` | `python aprendizaje_automatico/lightgbm.py` |
| CatBoost | Predicción de Fuga de Clientes (Churn) | CSV con manejo avanzado de variables categóricas | `aprendizaje_automatico/catboost.py` | `pandas`, `matplotlib`, `seaborn`, `numpy`, `catboost`, `scikit-learn` | `python aprendizaje_automatico/catboost.py` |
| Regresión Ridge | Series temporales económicas | CSV con variables colineales / Matriz tabular | `aprendizaje_automatico/Regresion Ridge.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/Regresion Ridge.py` |
| Regresión Lasso | Selección de genes en bioinformática | CSV o matrices con exceso de variables | `aprendizaje_automatico/Regresión Lasso.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/Regresión Lasso.py` |
| Elastic Net | Modelado estadístico avanzado | CSV o matrices con alta colinealidad / Características polinómicas | `aprendizaje_automatico/Elastic Net.py` | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn` | `python aprendizaje_automatico/Elastic Net.py` |
| GP | Optimización de hiperparámetros | Puntos espaciales o temporales (Dataset geoespacial) | `aprendizaje_automatico/GP.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python aprendizaje_automatico/GP.py` |

## Visión Computacional

**Cantidad en la clasificación: 28 elementos.**

| Modelo | Aplicación / función de referencia | Entrada indicada | Archivo Python | Dependencias | Ejecución |
|---|---|---|---|---|---|
| OCR | Digitalización de facturas y documentos impresos | Imágenes escaneadas locales | `vision_computacional/OCR.py` | `numpy`, `matplotlib`, `opencv-python`, `pytesseract` | `python vision_computacional/OCR.py` |
| CNN | Clasificación de imágenes / Visión por computador | Dataset de tensores (imágenes multicanal o escala de grises) | `vision_computacional/cnn.py` | `torch`, `torchvision`, `torchmetrics`, `numpy`, `pandas`, `matplotlib`, `tqdm` | `python vision_computacional/cnn.py` |
| Canny | Detección de contornos y bordes en imágenes | Imágenes estáticas locales (Matriz espacial 2D) | `vision_computacional/Canny.py` | `cv2`, `matplotlib` | `python vision_computacional/Canny.py` |
| LoG | Filtro de contornos / Detección multiescala | Imágenes rasterizadas en escala de grises | `vision_computacional/Laplacian of Gaussian.py` | `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn` | `python vision_computacional/Laplacian of Gaussian.py` |
| SIFT | Stitching de imágenes / Creación de panorámicas | Pares de imágenes rasterizadas con solapamiento | `vision_computacional/SIFT.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python` | `python vision_computacional/SIFT.py` |
| Detector Harris | Seguimiento de objetos y video / Flujo óptico | Secuencia de fotogramas rasterizados en escala de grises | `vision_computacional/Detector Harris.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python` | `python vision_computacional/Detector Harris.py` |
| SURF | Reconocimiento de objetos en tiempo real / Bounding box | Secuencias de fotogramas rasterizados con transformaciones | `vision_computacional/SURF.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python` | `python vision_computacional/SURF.py` |
| Brute-Force Matcher | Reconocimiento facial simple / Autenticación biométrica | Vectores de descriptores binarios (ORB) | `vision_computacional/Brute-Force Matcher.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python` | `python vision_computacional/Brute-Force Matcher.py` |
| FLANN | Búsqueda visual y localización de patrones | Vectores de descriptores (SIFT / KD-Trees) | `vision_computacional/FLANN.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python` | `python vision_computacional/FLANN.py` |
| RANSAC | Calibración extrínseca de cámaras / Estimación de pose | Conjuntos de puntos 3D a 2D con ruido severo | `vision_computacional/RANSAC.py` | `numpy`, `matplotlib`, `seaborn`, `cv2` | `python vision_computacional/RANSAC.py` |
| AlexNet | Clasificación base de imágenes / Redes profundas | Imágenes RGB reescaladas (CIFAR-10 / ImageNet) | `vision_computacional/AlexNet.py` | `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/AlexNet.py` |
| VGG (VGG16/19) | Transfer learning genérico / Extracción de características | Imágenes RGB redimensionadas (CIFAR-10 / ImageNet) | `vision_computacional/VGG16.py` | `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn` | `python vision_computacional/VGG16.py` |
| GoogLeNet | Sistemas embebidos / Edge AI / Eficiencia computacional | Imágenes RGB normalizadas (224x224 px) | `vision_computacional/GoogLeNet.py` | `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/GoogLeNet.py` |
| ResNet | Detección avanzada de objetos / Backbone FPN | Imágenes RGB normalizadas en alta resolución | `vision_computacional/ResNet.py` | `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/ResNet.py` |
| DenseNet | Diagnóstico médico por imagen / Rayos X | Imágenes normalizadas de rayos X (224x224 px) | `vision_computacional/Dense.py` | `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/Dense.py` |
| MobileNet | Visión en smartphones / Edge AI / Convoluciones ligeras | Imágenes RGB adaptadas (32x32 px) | `vision_computacional/MobileNet.py` | `numpy`, `matplotlib`, `seaborn`, `cv2`, `scikit-learn` | `python vision_computacional/MobileNet.py` |
| R-CNN | Investigación académica / Detección basada en regiones | Parches de regiones de interés (ROI) normalizados | `vision_computacional/R-CNN.py` | `numpy`, `matplotlib`, `seaborn`, `cv2`, `scikit-learn` | `python vision_computacional/R-CNN.py` |
| Fast R-CNN | Inventario visual / Detección y conteo de objetos | Imágenes RGB normalizadas | `vision_computacional/Fast R-CNN.py` | `torch`, `torchvision`, `numpy`, `matplotlib`, `pillow`, `opencv-contrib-python` | `python vision_computacional/Fast R-CNN.py` |
| Faster R-CNN | Inventario visual / Conteo de stock y detección en estanterías | Imágenes RGB de estanterías sintéticas (256x256 px) | `vision_computacional/Faster R-CNN.py` | `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/Faster R-CNN.py` |
| Cascade R-CNN | Detección de defectos industriales / Inspección de calidad | Superficies metálicas sintéticas texturizadas | `vision_computacional/Cascade R-CNN.py` | `numpy`, `matplotlib`, `seaborn`, `cv2`, `scikit-learn` | `python vision_computacional/Cascade R-CNN.py` |
| YOLO | Conteo vehicular automático / Ciudades inteligentes | Imágenes viales sintéticas normalizadas (320x320 px) | `vision_computacional/YOLO.py` | `torch`, `ultralytics`, `numpy`, `matplotlib`, `seaborn`, `cv2`, `scikit-learn` | `python vision_computacional/YOLO.py` |
| SSD | Visión en drones / Monitoreo aéreo / Detección en una sola etapa | Imágenes RGB en perspectiva cenital | `vision_computacional/SSD.py` | `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python vision_computacional/SSD.py` |
| U-Net | Segmentación de tumores / Imágenes médicas de resonancia magnética | Imágenes normalizadas en escala de grises (64x64 px) | `vision_computacional/U-Net.py` | `torch`, `numpy`, `matplotlib`, `cv2`, `scikit-learn` | `python vision_computacional/U-Net.py` |
| FPN | Mejora de redes base / Pirámides de características multiescala | Imágenes RGB redimensionadas (STL-10 / 128x128 px) | `vision_computacional/FPN.py` | `torch`, `torchvision`, `numpy` | `python vision_computacional/FPN.py` |
| PSPNet | Análisis de escenas / Segmentación semántica multicontexto | Imágenes RGB de alta resolución (128x128 px) | `vision_computacional/PSPNet.py` | `numpy`, `matplotlib`, `seaborn`, `cv2`, `scikit-learn` | `python vision_computacional/PSPNet.py` |
| Mask R-CNN | Conteo celular automático / Segmentación de instancias | Micrografías RGB de tejido celular sintético (64x64 px) | `vision_computacional/Mask R-CNN.py` | `torch`, `torchvision`, `numpy`, `cv2`, `scikit-learn`, `matplotlib` | `python vision_computacional/Mask R-CNN.py` |
| YOLACT | Realidad aumentada interactiva / Segmentación en tiempo real | Marcos de video en streaming o imágenes RGB | `vision_computacional/YOLACT.py` | `torch`, `torchvision`, `ultralytics`, `numpy`, `cv2` | `python vision_computacional/YOLACT.py` |
| Vision Transformer (ViT) | Clasificación de imágenes a gran escala / Mecanismos de atención | Imágenes de rostros normalizadas en parches | `vision_computacional/Vision Transformer.py` | `torch`, `numpy`, `matplotlib`, `scikit-learn` | `python vision_computacional/Vision Transformer.py` |

## IA Generativa

**Cantidad en la clasificación: 28 elementos.**

| Modelo | Aplicación / función de referencia | Entrada indicada | Archivo Python | Dependencias | Ejecución |
|---|---|---|---|---|---|
| PaLM / Gemini | Búsqueda semántica, embeddings vectoriales y resumen de textos | Documentos de texto plano (20 Newsgroups) | `ia_generativa/PaLM.py` | `numpy`, `pandas`, `scikit-learn`, `google-generativeai`, `tqdm` | `python ia_generativa/PaLM.py` |
| ChatGPT / GPT-5 | Resolución de problemas complejos / Cadena de pensamiento (CoT) / Agentes analíticos | Datos tabulares médicos multivariables (Breast Cancer Wisconsin) | `ia_generativa/ChatGPT-GPT-5.py` | `openai`, `numpy`, `pandas`, `scikit-learn` | `python ia_generativa/ChatGPT-GPT-5.py` |
| Claude | Análisis de contratos legales / Auditoría regulatoria / Structured Outputs | Acuerdos contractuales en texto plano | `ia_generativa/Claude.py` | `anthropic`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn` | `python ia_generativa/Claude.py` |
| Gemini 3 Pro | Asistencia técnica automatizada / Function Calling / Modelos multimodales | Tickets técnicos de texto en lenguaje natural (20 Newsgroups) | `ia_generativa/Gemini pro.py` | `google-genai`, `scikit-learn`, `numpy` | `python ia_generativa/Gemini pro.py` |
| Gemini / Gemini Flash | Agentes conversacionales fluidos / Memoria multiturno / Tool Use | Interacciones de usuario en texto plano | `ia_generativa/Gemini.py` | `google-genai`, `pandas`, `numpy`, `scikit-learn` | `python ia_generativa/Gemini.py` |
| Perplexity AI | Investigación académica / Grounded Research / Búsqueda web en tiempo real | Consultas de búsqueda NLP y tópicos científicos | `ia_generativa/Perplexity AI.py` | `openai`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn` | `python ia_generativa/Perplexity AI.py` |
| Grok 4.1 | Análisis de tendencias de mercado / Razonamiento financiero cuantitativo | Series temporales bursátiles (S&P 500 / OpenML) | `ia_generativa/Grok.py` | `xai-sdk`, `pandas`, `numpy`, `scikit-learn` | `python ia_generativa/Grok.py` |
| Midjourney | Creación de arte conceptual / Prompt engineering generativo | Dataset facial de scikit-learn (LFW People) | `ia_generativa/Midjourney.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python ia_generativa/Midjourney.py` |
| DALL-E 3 | Ilustraciones para marketing / Publicidad comercial / Generación de recursos | Catálogos de productos y descripciones textuales | `ia_generativa/DALL-E 3.py` | `openai`, `pandas`, `numpy`, `scikit-learn`, `requests`, `pillow` | `python ia_generativa/DALL-E 3.py` |
| Stable Diffusion / Flux | Generación masiva de texturas 3D / PBR / Modelos de difusión | Nombres de materiales y catálogos de superficie | `ia_generativa/Stable Diffusion- Flux.py` | `torch`, `diffusers`, `pandas`, `numpy`, `pillow`, `scikit-learn` | `python ia_generativa/Stable Diffusion- Flux.py` |
| Nano Banana Pro | Publicidad digital / RTB / Predicción de CTR y conversiones | Datos tabulares de impresiones publicitarias | `ia_generativa/Nano.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python ia_generativa/Nano.py` |
| Adobe Firefly | Diseño gráfico profesional / Generación y edición comercial | Imágenes de diseño base y prompts de texto | `ia_generativa/Adobe Firefly Services.py` | `requests`, `pandas`, `numpy`, `pillow` | `python ia_generativa/Adobe Firefly Services.py` |
| Leonardo & Prome | Diseño de personajes 2D/3D / Hojas de modelo (turnaround) / Game Assets | Arquetipos de personajes y descripciones textuales | `ia_generativa/LeonardoAi.py` | `leonardo-ai-sdk`, `requests`, `scikit-learn`, `numpy` | `python ia_generativa/LeonardoAi.py` |
| Luma Photon | VFX y animación / Simulación de dinámicas / Renderizado cinematográfico | Sistemas de partículas tridimensionales (X, Y, Z) | `ia_generativa/Luma Photon.py` | `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | `python ia_generativa/Luma Photon.py` |
| Seedream | Conceptos creativos / Arte conceptual / Clustering semántico LSA | Tópicos documentales y científicos (20 Newsgroups) | `ia_generativa/Seedream.py` | `requests`, `pandas`, `matplotlib`, `scikit-learn` | `python ia_generativa/Seedream.py` |
| Canva Magic Media | Presentaciones y redes sociales / Automatización de plantillas / Autofill | Datos tabulares de segmentación y marketing | `ia_generativa/Canva Magic Media.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/Canva Magic Media.py` |
| Runway (Gen-3 Alpha) | Producción de cine / Generación Text-to-Video / Control de cámara | Dataset facial de scikit-learn (LFW People) | `ia_generativa/Runway Gen-3 Alpha.py` | `numpy`, `pandas`, `scikit-learn` | `python ia_generativa/Runway Gen-3 Alpha.py` |
| Synthesia | Videos de capacitación / Avatares de IA / E-learning automatizado | Guiones pedagógicos y texto estructurado | `ia_generativa/Synthesia.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/Synthesia.py` |
| Colossyan | Marketing y RRHH / Onboarding corporativo automatizado / Avatares de IA | Datos de empleados y perfiles organizacionales | `ia_generativa/Colossyan.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/Colossyan.py` |
| Pictory | Shorts y Reels automáticos / Script-to-Video / Automatización de contenido vertical | Artículos y textos documentales largos | `ia_generativa/Pictory AI.py` | `numpy`, `pandas`, `scikit-learn` | `python ia_generativa/Pictory AI.py` |
| DeepBrain AI | Educación online / Creación de cursos en video / Avatares de IA | Guiones pedagógicos y conceptos técnicos estructurados | `ia_generativa/DeepBrain AI.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/DeepBrain AI.py` |
| InVideo | Redes sociales / Videos verticales (Reels/TikTok/Shorts) / Automatización de guiones | Datos inmobiliarios y demográficos (California Housing) | `ia_generativa/InVideo.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/InVideo.py` |
| Fliki | Creación de contenido rápido / Text-to-Video / Text-to-Speech (TTS) | Datos inmobiliarios estructurados (California Housing) | `ia_generativa/Fliki AI.py` | `numpy`, `pandas`, `scikit-learn` | `python ia_generativa/Fliki AI.py` |
| Sora y Veo | Simuladores y cine avanzado / Generación de video largo / Audio nativo | Guiones y especificaciones de tomas cinematográficas | `ia_generativa/Sora API.py` | `openai`, `google-genai`, `pandas`, `scikit-learn` | `python ia_generativa/Sora API.py` |
| Humata AI | Análisis de contratos e informes / RAG documental / Auditoría legal y financiera | Documentos de texto o PDF largos | `ia_generativa/HumataAI.py` | `requests`, `pandas`, `scikit-learn` | `python ia_generativa/HumataAI.py` |
| VAE | Generación de rostros interpolados / Reducción dimensional probabilística / Espacio latente | Dataset facial de scikit-learn (LFW People) | `ia_generativa/VAE.py` | `torch`, `numpy`, `matplotlib`, `scikit-learn` | `python ia_generativa/VAE.py` |
| GAN | Deepfakes y súper resolución / Redes adversarias generativas / Ruido latente | Dataset facial de scikit-learn (Olivetti Faces) | `ia_generativa/GAN.py` | `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` | `python ia_generativa/GAN.py` |
| Modelos de Difusión (DPM) | Síntesis de imágenes fotorrealistas / Modelos de difusión DDPM / Ruido gaussiano | Dataset de dígitos de scikit-learn (digits) | `ia_generativa/Modelos de Difusión (DPM).py` | `torch`, `numpy`, `matplotlib`, `scikit-learn` | `python ia_generativa/Modelos de Difusión (DPM).py` |

## Procesamiento de Lenguaje Natural

**Cantidad en la clasificación: 9 elementos.**

| Modelo | Aplicación / función de referencia | Entrada indicada | Archivo Python | Dependencias | Ejecución |
|---|---|---|---|---|---|
| BERT | Atención al cliente y chatbots / Representaciones contextuales / Embeddings de Transformers | Texto tokenizado y corpus de Wikipedia | `procesamiento_lenguaje_natural/bert.py` | `transformers`, `datasets`, `torch` | `python procesamiento_lenguaje_natural/bert.py` |
| StructBERT | Gestión documental corporativa | Texto tokenizado con enmascaramiento | `procesamiento_lenguaje_natural/structbert.py` | `transformers`, `datasets`, `torch`, `numpy`, `evaluate` | `python procesamiento_lenguaje_natural/structbert.py` |
| GPT-3 | Asistentes virtuales y redacción automática | Texto (Prompts y consultas en lenguaje natural) | `procesamiento_lenguaje_natural/GPT3.py` | `openai` | `python procesamiento_lenguaje_natural/GPT3.py` |
| Chatbots | Soporte técnico de nivel 1 / Chatbot conversacional de análisis de sentimiento | Reglas, datasets locales de texto y pipeline NLP | `procesamiento_lenguaje_natural/chatbot.py` | `transformers`, `datasets`, `torch`, `numpy`, `evaluate` | `python procesamiento_lenguaje_natural/chatbot.py` |
| RoBERTa | Análisis de sentimiento avanzado / Clasificación de polaridad emocional | Texto en bruto en lenguaje natural | `procesamiento_lenguaje_natural/roberta.py` | `transformers`, `torch` | `python procesamiento_lenguaje_natural/roberta.py` |
| Parsing | Extracción de entidades nombradas / Normalización de salidas de modelos | Texto estructurado y respuestas crudas de pipelines | `procesamiento_lenguaje_natural/parsing.py` | `transformers`, `torch` | `python procesamiento_lenguaje_natural/parsing.py` |
| XLNet | Análisis de textos largos / Modelos autorregresivos generalizados / Transformer-XL | Dataset 20 newsgroups de scikit-learn | `procesamiento_lenguaje_natural/XLnet.py` | `torch`, `transformers`, `pandas`, `scikit-learn` | `python procesamiento_lenguaje_natural/XLnet.py` |
| RNN / LSTM | Traducción automática y forecasting / Series temporales / Redes recurrentes con memoria | Datos financieros históricos reales (yfinance - AAPL) | `procesamiento_lenguaje_natural/RNN-LSTM.py` | `tensorflow`, `yfinance`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` | `python procesamiento_lenguaje_natural/RNN-LSTM.py` |
| Transformers | Generación y traducción avanzada / Modelos de atención / Causal LM y Seq2Seq | Texto real de scikit-learn (20 newsgroups - sci.space) | `procesamiento_lenguaje_natural/Transformers.py` | `transformers`, `torch`, `numpy`, `matplotlib`, `scikit-learn` | `python procesamiento_lenguaje_natural/Transformers.py` |

## Requisitos generales

- Python y su versión: **pendiente de confirmar según cada implementación**.
- `pip` para instalar paquetes de Python.
- Git, si el repositorio se obtiene mediante un repositorio remoto.
- Algunos modelos o herramientas pueden requerir pesos preentrenados, archivos de configuración, acceso a una API, credenciales, conexión a Internet o hardware específico.
- No todos los elementos del catálogo son necesariamente scripts Python ejecutables localmente. Algunas entradas pueden corresponder a servicios web, plataformas comerciales, arquitecturas o familias de modelos; su forma de uso debe documentarse de acuerdo con el caso real.

## Instalación

Clonar el repositorio (reemplazar el marcador por la URL real):

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activar el entorno:

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

Si existe un `requirements.txt` actualizado y adecuado para el entorno, instalar sus dependencias con:

```bash
pip install -r requirements.txt
```

Si las dependencias varían entre modelos, instalar las que estén documentadas en la ficha correspondiente. No se debe asumir que un único archivo de dependencias cubre todas las implementaciones.

## Cómo ejecutar una implementación

Desde la raíz del repositorio, ejecutar el archivo Python correspondiente. Ejemplo ilustrativo:

```bash
python aprendizaje_automatico/ruta_del_modelo.py
```

La ruta, los argumentos y los recursos de entrada deben reemplazarse por los reales. Si el script requiere argumentos, se pueden documentar, por ejemplo:

```bash
python ruta/al/modelo.py --input ruta/al/archivo_de_entrada
```

No todos los modelos utilizan esta interfaz; revisar el código para confirmar si se ejecutan como scripts, módulos, notebooks, aplicaciones o mediante una API.

## Ficha individual de cada modelo

Utilizar la siguiente ficha para documentar cada implementación cuando se inspeccione su código:

### K-Means Clustering (Iris Dataset)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering)]
- **Archivo:** `aprendizaje_automatico/kmeans.py`
- **Función básica:** Aplica el algoritmo de agrupamiento K-Means para clasificar el conjunto de datos Iris en 3 clusters. Resuelve una tarea de segmentación de datos y produce como salida la impresión de los centroides en la consola y dos visualizaciones: una matriz de gráficos de dispersión (`pairplot`) de todas las variables y un gráfico de dispersión bidimensional con la distribución de los clusters y la ubicación de sus centroides.
- **Entrada requerida:** Datos tabulares estructurados. No requiere un archivo de entrada manual, ya que importa automáticamente el dataset de juguete `iris` directamente desde la librería `scikit-learn` (4 características numéricas).
- **Dependencias:** `scikit-learn`, `matplotlib`, `pandas`, `seaborn`.
- **Instalación específica:** `pip install scikit-learn matplotlib pandas seaborn`
- **Ejecución:** `python aprendizaje_automatico/kmeans.py`
- **Recursos externos:** No aplica (el dataset está integrado en la librería de dependencias y no requiere conexión a APIs ni pesos preentrenados).
- **Observaciones:** La visualización final en el gráfico de dispersión solo utiliza las dos primeras características (Longitud y Ancho del Sépalo) de las cuatro disponibles, por lo que es una representación parcial en 2D de una agrupación de 4 dimensiones. El dataset viene limpio por defecto; para otros datos en el mundo real, habría que agregar un paso previo de escalado o normalización.

### Clustering jerarquico

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering y Reducción de Dimensionalidad)]
- **Archivo:** `aprendizaje_automatico/clustering_jerarquico.py`
- **Función básica:** Procesa y segmenta una base de datos de clientes para diseñar estrategias de marketing personalizadas. Aplica escalado de datos, reducción de dimensionalidad (PCA) y métodos de agrupamiento (K-Means y Jerárquico) para identificar 5 perfiles de consumidores. Produce como salida gráficos de dispersión 2D, un dendrograma y un resumen descriptivo en consola con las medias de edad, ingresos y gasto por cada cluster.
- **Entrada requerida:** Datos tabulares en formato CSV. El script descarga automáticamente el archivo `customer_data.csv` desde un repositorio de GitHub (incluye columnas demográficas y de comportamiento financiero).
- **Dependencias:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`.
- **Instalación específica:** `pip install pandas numpy matplotlib seaborn scikit-learn scipy`
- **Ejecución:** `python aprendizaje_automatico/clustering_jerarquico.py`
- **Recursos externos:** Dataset remoto alojado en GitHub (`https://github.com/gianlucafran/segmentacion-clientes-/raw/refs/heads/main/customer_data.csv`). No requiere pesos preentrenados, variables de entorno ni claves API.
- **Observaciones:** El script extrae 3 componentes principales (`n_components=3`) en PCA, pero las visualizaciones proyectan únicamente los dos primeros componentes para renderizar gráficos en 2D. Se requiere una conexión a internet activa para descargar el dataset desde la URL proporcionada.

### DBSCAN

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering y Detección de Anomalías)]
- **Archivo:** `aprendizaje_automatico/DBSCAN.py`
- **Función básica:** Implementa el algoritmo DBSCAN para identificar clusters y detectar anomalías (outliers) en datos geoespaciales basándose en la densidad espacial de los puntos. Resuelve tareas de segmentación y detección de valores atípicos. Produce como salida la evaluación del modelo en consola (precisión, recall, F1), una imagen con 4 gráficos (`dbscan_anomalias_geo.png`), un archivo de exportación espacial (`anomalias_geoespaciales.geojson`) y un bloque de código JavaScript base para visualizar los resultados en Mapbox.
- **Entrada requerida:** Coordenadas geoespaciales (latitud y longitud). En la ejecución de este script, no se requiere un archivo externo, ya que genera sintéticamente un dataset de prueba simulando coordenadas en la región de México (Querétaro, Celaya, Guanajuato, etc.) junto con anomalías aleatorias.
- **Dependencias:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`. (Emplea también librerías nativas de Python como `json`, `typing` y `warnings`).
- **Instalación específica:** `pip install numpy pandas scikit-learn matplotlib seaborn`
- **Ejecución:** `python aprendizaje_automatico/DBSCAN.py`
- **Recursos externos:** No requiere conexión directa a APIs, variables de entorno ni pesos preentrenados para entrenar el modelo. Sin embargo, el script genera código cliente que sugiere usar la API de Mapbox GL JS para la visualización interactiva del archivo `.geojson`.
- **Observaciones:** Utiliza la métrica `haversine` para medir distancias reales sobre la superficie terrestre, por lo que el script convierte automáticamente los grados a radianes antes de modelar. Las rutas de exportación están "hardcodeadas" hacia `/home/claude/`; se debe asegurar cambiar estas rutas al directorio local adecuado antes de ejecutar para evitar errores de escritura.

### OPTICS

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering)]
- **Archivo:** `aprendizaje_automatico/OPTICS.py`
- **Función básica:** Aplica el algoritmo OPTICS (Ordering Points To Identify the Clustering Structure) para identificar agrupaciones de puntos de interés (POI) en áreas urbanas, manejando eficientemente clusters de diferentes densidades. Produce como salida estadísticas detalladas en consola (centroides, radios y densidades por cluster) y genera una imagen (`optics_urban_poi_visualization.png`) con 4 subgráficos que muestran la distribución espacial, el gráfico de alcanzabilidad (reachability) y las distancias k-NN.
- **Entrada requerida:** Coordenadas bidimensionales (latitud y longitud). El script no requiere un archivo externo, ya que simula internamente un dataset de 130 puntos distribuidos en diferentes "zonas urbanas" con niveles de densidad variables, más puntos de ruido.
- **Dependencias:** `numpy`, `matplotlib`, `scikit-learn`. (También usa `warnings` nativo de Python).
- **Instalación específica:** `pip install numpy matplotlib scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/OPTICS.py`
- **Recursos externos:** No aplica (genera sus propios datos de juguete y no requiere pesos, variables de entorno ni claves API).
- **Observaciones:** El script intenta guardar el gráfico de salida en una ruta absoluta fija (`/home/claude/optics_urban_poi_visualization.png`). **Es necesario modificar esta ruta** a un directorio local válido antes de ejecutar el código para evitar errores de directorio no encontrado (por ejemplo, cambiarla a `./optics_urban_poi_visualization.png`).

### Clustering Espectral

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering de Grafos)]
- **Archivo:** `aprendizaje_automatico/Clustering_espectral.py`
- **Función básica:** Aplica el algoritmo de Clustering Espectral sobre una matriz de adyacencia para detectar comunidades o facciones dentro de una red social (grafo). En lugar de usar distancias euclidianas, analiza la topología matemática de las conexiones (Laplaciano del grafo) para segmentar la red. Produce como salida el Adjusted Rand Score en consola para evaluar la precisión y genera una imagen comparativa (`01_clustering_espectral_redes.png`) mostrando la división histórica real vs. las comunidades inferidas por la IA.
- **Entrada requerida:** Datos relacionales en forma de grafo o matriz de afinidad (tamaño NxN con 1s y 0s para representar conexiones). El script no requiere archivos externos, ya que carga el histórico dataset "Zachary's Karate Club" de forma interna mediante la librería `networkx`.
- **Dependencias:** `numpy`, `matplotlib`, `networkx`, `scikit-learn`, `seaborn`.
- **Instalación específica:** `pip install numpy matplotlib networkx scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Clustering_espectral.py`
- **Recursos externos:** No aplica (el dataset está integrado en `networkx` y no se requieren pesos, variables de entorno ni claves API).
- **Observaciones:** El algoritmo utiliza `affinity='precomputed'` porque se alimenta directamente de la estructura del grafo en lugar de características tabulares. El script intenta guardar el gráfico en una ruta absoluta fija (`/home/usuario/01_clustering_espectral_redes.png`). **Es indispensable modificar esta ruta** hacia un directorio local válido (por ejemplo, `./01_clustering_espectral_redes.png`) antes de la ejecución para evitar un error de sistema de archivos.

### Reducción de Dimensionalidad (PCA + K-Means)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Reducción de Dimensionalidad y Clustering)]
- **Archivo:** `aprendizaje_automatico/PCA.py`
- **Función básica:** Aplica el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de los datos (de 4 a 2 o 3 componentes) conservando la mayor parte de la varianza, con el fin de optimizar el posterior agrupamiento mediante K-Means. Genera como salida un extenso reporte en consola evaluando el rendimiento (varianza explicada, matrices de correlación, método del codo, índices Silhouette y Davies-Bouldin) y guarda cinco imágenes PNG con gráficos detallados del análisis exploratorio y del clustering.
- **Entrada requerida:** Datos tabulares estructurados. No requiere cargar un archivo manualmente, ya que importa automáticamente el dataset de clasificación `iris` desde `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`.
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/PCA.py`
- **Recursos externos:** No aplica (el dataset está integrado en la librería y no requiere conexión a APIs ni pesos preentrenados).
- **Observaciones:** El script intenta guardar las 5 gráficas PNG en una ruta absoluta rígida (`/home/claude/`, por ejemplo: `/home/claude/01_matriz_correlacion.png`). **Es indispensable modificar estas rutas** apuntando a un directorio local válido en tu máquina (como `./01_matriz_correlacion.png`) antes de la ejecución para evitar fallos de "directorio no encontrado".

### SVD (Filtrado Colaborativo y Recomendaciones)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Reducción de Dimensionalidad y Sistemas de Recomendación)]
- **Archivo:** `aprendizaje_automatico/SVD.py`
- **Función básica:** Aplica la Descomposición en Valores Singulares Truncada (TruncatedSVD) para analizar una matriz usuario-ítem dispersa (esparsa). Descubre perfiles de gustos latentes, predice las calificaciones de productos que el usuario aún no ha evaluado y genera recomendaciones personalizadas. Produce como salida reportes por consola (varianzas, matrices reconstruidas y ejemplos de recomendación) y guarda dos visualizaciones: un gráfico de codo de varianza y mapas de calor comparativos (`01_svd_varianza.png` y `02_reconstruccion_svd.png`).
- **Entrada requerida:** Matriz numérica de interacciones usuario-ítem (con valores faltantes/nulos). El script no requiere datos externos, ya que simula sintéticamente una matriz de 100 usuarios y 20 ítems con un 75% de esparcidad utilizando la función `make_biclusters` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`.
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/SVD.py`
- **Recursos externos:** No aplica (el dataset es generado de forma sintética en tiempo de ejecución; no se requieren APIs ni pesos previos).
- **Observaciones:** El script utiliza la clase `TruncatedSVD` en lugar del PCA tradicional porque no centra los datos previamente, lo cual es fundamental para manejar matrices dispersas de forma eficiente. Al igual que en ejemplos anteriores, las imágenes se intentan guardar en rutas absolutas rígidas (`/home/usuario/`). **Es necesario actualizar estas rutas** a un directorio relativo válido (ej. `./01_svd_varianza.png`) antes de ejecutar para evitar errores de directorio inexistente.

### t-SNE (Exploración de Espacios Latentes)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Reducción de Dimensionalidad y Visualización)]
- **Archivo:** `aprendizaje_automatico/t-SNE.py`
- **Función básica:** Aplica el algoritmo no lineal t-SNE para reducir la dimensionalidad de un conjunto de imágenes (de 64 características a solo 2) y visualizar su espacio latente, logrando agrupar dígitos similares de manera semántica. Resuelve tareas de exploración de datos complejos (manifolds). Produce como salida tiempos de ejecución y análisis en consola, además de dos imágenes: una comparativa directa entre PCA y t-SNE (`01_pca_vs_tsne.png`), y otra demostrando el efecto del hiperparámetro de perplejidad (`02_tsne_perplejidades.png`).
- **Entrada requerida:** Datos tabulares numéricos de alta dimensionalidad (imágenes aplanadas de 8x8 píxeles = 64 dimensiones). No necesita un archivo externo, ya que importa el dataset de juguete `digits` directamente desde `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También usa `time`, que es nativa de Python).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/t-SNE.py`
- **Recursos externos:** No aplica (dataset integrado en scikit-learn; no requiere pesos, variables de entorno ni claves API).
- **Observaciones:** El algoritmo t-SNE en `scikit-learn` solo tiene el método `fit_transform`, por lo que **no puede usarse** para proyectar datos nuevos (como preprocesamiento para modelos predictivos), limitándose a la visualización estática. El script intenta guardar las imágenes en una ruta rígida (`/home/usuario/`); **es obligatorio actualizar estas rutas** a un directorio local válido (por ejemplo, `./01_pca_vs_tsne.png`) antes de su ejecución para evitar un error de sistema.

### UMAP (Visualización de Embeddings)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Reducción de Dimensionalidad y Visualización)]
- **Archivo:** `aprendizaje_automatico/UMAP.py`
- **Función básica:** Aplica el algoritmo UMAP (basado en geometría Riemanniana y topología) para reducir la dimensionalidad de embeddings complejos (de 64 dimensiones a 2). Resuelve tareas de visualización de espacios de alta dimensionalidad, destacando por preservar tanto la estructura local como la global de los datos. Produce como salida reportes por consola y guarda dos imágenes: una comparación directa con PCA (`01_pca_vs_umap.png`) y una exploración visual del impacto de los hiperparámetros `n_neighbors` y `min_dist` (`02_umap_hiperparametros.png`).
- **Entrada requerida:** Embeddings numéricos densos de alta dimensionalidad. El script no requiere un archivo externo, ya que utiliza el dataset `digits` (imágenes de 8x8 aplanadas) importado internamente desde `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `umap`. (También utiliza módulos nativos como `time` y `sys`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn umap-learn`
- **Ejecución:** `python aprendizaje_automatico/UMAP.py`
- **Recursos externos:** No aplica (dataset integrado en la librería; no requiere pesos de modelos externos, variables de entorno ni claves API).
- **Observaciones:** Es crítico instalar la librería principal como `umap-learn` en pip (si instalas solo `umap` descargarás un paquete incorrecto que no funcionará). A diferencia de t-SNE, UMAP sí permite usar la función `transform()` para proyectar nuevos datos. El script intenta guardar las imágenes en una ruta absoluta fija (`/home/usuario/`); **es obligatorio actualizar estas rutas** a un directorio local válido (por ejemplo, `./01_pca_vs_umap.png`) antes de su ejecución para evitar un error de sistema.

### Isolation Forest (Detección de Fraudes Financieros)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Detección de Anomalías)]
- **Archivo:** `aprendizaje_automatico/Isolation_Forest.py`
- **Función básica:** Implementa el algoritmo Isolation Forest para identificar anomalías en un conjunto de datos altamente desbalanceado (simulando transacciones financieras fraudulentas). En lugar de modelar el comportamiento normal, aísla las anomalías explícitamente mediante particiones aleatorias en ramas de árboles. Produce como salida reportes de evaluación en consola (matriz de confusión y reporte de clasificación) y genera dos gráficas: una proyección en PCA comparando la realidad con las predicciones del modelo (`01_fraude_isolation_forest.png`) y un histograma con la distribución de los scores de anomalía (`02_score_anomalia.png`).
- **Entrada requerida:** Datos tabulares numéricos (características de las transacciones). El script no requiere cargar un archivo externo, ya que genera internamente un dataset sintético de 10,000 muestras (con 1% de casos fraudulentos) utilizando la función `make_classification` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza `warnings`, nativa de Python).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Isolation_Forest.py`
- **Recursos externos:** No aplica (los datos se generan de forma sintética en tiempo de ejecución; no requiere conexión a APIs, variables de entorno ni pesos preentrenados).
- **Observaciones:** El modelo depende del hiperparámetro `contamination`, que debe ser estimado por el usuario (en este caso fijado al 1% o `0.01`). Al igual que en scripts anteriores, las rutas de guardado de las imágenes están fijadas a un directorio absoluto (`/home/usuario/`); **es obligatorio modificar estas rutas** a un directorio local válido (por ejemplo, `./01_fraude_isolation_forest.png`) antes de ejecutar el código para evitar un error de sistema de archivos.

### One-Class SVM (Mantenimiento Predictivo Industrial y Detección de Anomalías)

- **Categoría:** Inteligencia Artificial / Aprendizaje No Supervisado, Detección de Anomalías (*Novelty Detection*) y Mantenimiento Predictivo
- **Archivo:** `aprendizaje_automatico/One-Class SVM.py`
- **Función básica:** Implementa un modelo de **Support Vector Machine de una sola clase (*One-Class SVM*)** utilizando scikit-learn para el **monitoreo predictivo y detección temprana de fallas en maquinaria industrial** a partir de lecturas de sensores (temperatura y vibración). El script simula y normaliza un conjunto de datos multivariante (`StandardScaler`), entrena la frontera de decisión no lineal con un Kernel RBF exclusivamente sobre el régimen de operación nominal, evalúa la precisión frente a anomalías ocultas mediante matriz de confusión y reportes estadísticos, y genera gráficos analíticos de la región de decisión de la máquina.
- **Entrada requerida:** Datos numéricos normalizados en matrices tabulares o archivos CSV (en este caso, generados y preprocesados de manera automatizada en el script).
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/One-Class SVM.py`
- **Recursos externos:** No requiere conexión a internet ni claves de API propietarias, ya que opera de manera autónoma con datasets sintéticos generados mediante scikit-learn.
- **Observaciones:** Este script enriquece la sección de aprendizaje no supervisado y mantenimiento industrial del repositorio, ofreciendo una solución robusta para identificar desviaciones sutiles en equipos sin requerir historiales previos de fallas etiquetadas.

### Autoencoders (Detección de Anomalías Complejas)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Detección de Anomalías)]
- **Archivo:** `aprendizaje_automatico/Autoencoders.py`
- **Función básica:** Implementa una arquitectura de red neuronal tipo Autoencoder para identificar anomalías complejas basándose en el error de reconstrucción. La red se entrena comprimiendo y reconstruyendo exclusivamente datos "normales" (células benignas); al procesar una anomalía, el modelo genera un error cuadrático medio (MSE) inusualmente alto. Produce estadísticas en consola (matriz de confusión y reporte de clasificación) y genera una imagen (`01_autoencoder_errores.png`) con el histograma de distribución de errores y la curva ROC.
- **Entrada requerida:** Datos tabulares numéricos estructurados (30 características). El script no requiere cargar un archivo CSV manualmente, ya que importa el dataset `breast_cancer` nativo de `scikit-learn` y lo adapta convirtiendo la clase maligna en anomalías simuladas.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (Emplea también la librería nativa `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Autoencoders.py`
- **Recursos externos:** No aplica (dataset integrado directamente en scikit-learn; no requiere pesos preentrenados de redes profundas ni acceso a APIs).
- **Observaciones:** Dado que `scikit-learn` no cuenta con una clase "Autoencoder" pura, el código lo simula magistralmente utilizando `MLPRegressor` ajustando la entrada para que prediga su propia matriz (`fit(X, X)`). El script tiene rutas absolutas predefinidas para guardar los gráficos (`/home/usuario/`); **es obligatorio cambiar esta ruta** a un directorio local válido en tu máquina (ej. `./01_autoencoder_errores.png`) antes de ejecutar el código para evitar que el programa falle al intentar guardar la imagen.

### GMM Aprendizaje Automático [(Modelos de Mezcla Gaussiana y Soft Clustering)]

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Clustering)]
- **Archivo:** `aprendizaje_automatico/GMM.py`
- **Función básica:** Aplica Modelos de Mezcla Gaussiana (GMM) para realizar una segmentación suave (soft clustering) de clientes basándose en su comportamiento. Evalúa el número óptimo de segmentos utilizando criterios AIC/BIC e identifica "clientes híbridos" (aquellos con comportamientos mixtos o en zonas de transición). Produce como salida probabilidades de pertenencia en consola y genera dos visualizaciones: curvas de optimización de clústeres (`01_optimizacion_gmm.png`) y gráficas de dispersión comparando segmentación dura vs. probabilística (`02_segmentacion_hibrida.png`).
- **Entrada requerida:** Datos numéricos tabulares (variables de comportamiento del cliente). El script no requiere cargar un archivo externo, ya que simula internamente un dataset de 1,500 clientes con características superpuestas utilizando la función `make_blobs` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También usa `warnings` nativa de Python).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/GMM.py`
- **Recursos externos:** No aplica (los datos se generan de forma sintética en tiempo de ejecución; no requiere conexión a APIs, variables de entorno ni pesos preentrenados).
- **Observaciones:** En el script proporcionado, las funciones para guardar las imágenes (`plt.savefig`) se encuentran comentadas (`#`). Si deseas que los gráficos se guarden físicamente en tu equipo, deberás descomentar esas líneas y reemplazar la ruta absoluta rígida (`/home/usuario/`) por un directorio local válido (por ejemplo, `./01_optimizacion_gmm.png`).

### Apriori  (Análisis de la Canasta de Compra)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Supervisado (Reglas de Asociación)]
- **Archivo:** `aprendizaje_automatico/apriori.py`
- **Función básica:** Implementa el algoritmo Apriori para realizar un Análisis de la Canasta de Compra (Market Basket Analysis). Extrae conjuntos de ítems frecuentes y genera reglas de asociación evaluadas por soporte, confianza y lift (sustento), simulando un motor de recomendaciones cruzadas para e-commerce. Produce estadísticas en consola (frecuencias, mejores reglas, sugerencias de carritos de prueba) y una imagen (`01_apriori_reglas.png`) con un gráfico de dispersión de métricas y un mapa de calor de correlaciones de productos.
- **Entrada requerida:** Matriz binaria o booleana de transacciones (filas como carritos/tickets, columnas como productos). El script no requiere un archivo externo, ya que simula internamente 1,000 transacciones con 8 productos utilizando la función `make_multilabel_classification` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, `mlxtend`. (También emplea `warnings` y `sys` de forma nativa).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn mlxtend`
- **Ejecución:** `python aprendizaje_automatico/apriori.py`
- **Recursos externos:** No aplica (los datos se generan sintéticamente en tiempo de ejecución; no requiere conexión a APIs, variables de entorno ni pesos preentrenados).
- **Observaciones:** Dado que `scikit-learn` no incluye una implementación nativa de Apriori, el script depende obligatoriamente de la librería de la industria `mlxtend`. El script intenta guardar el gráfico generado en una ruta absoluta rígida (`/home/usuario/01_apriori_reglas.png`); **es fundamental modificar esta ruta** hacia un directorio local válido (por ejemplo, `./01_apriori_reglas.png`) antes de la ejecución para evitar fallos del sistema al guardar el archivo.

### DQN (Deep Q-Network para Control Robótico)

- **Categoría:** Aprendizaje Automático [Aprendizaje por Refuerzo (Reinforcement Learning)]
- **Archivo:** `aprendizaje_automatico/DQN.py`
- **Función básica:** Implementa un agente de Aprendizaje por Refuerzo utilizando una Deep Q-Network (DQN) adaptada desde un perceptrón multicapa (`MLPRegressor`). Resuelve un problema de navegación espacial discreta, donde un robot debe encontrar la ruta óptima hacia una meta en una cuadrícula esquivando obstáculos. Produce como salida el progreso del entrenamiento (recompensas y métricas) en consola y genera una imagen (`01_dqn_robotics.png`) que muestra el mapa del entorno con la ruta aprendida y la curva de aprendizaje del modelo.
- **Entrada requerida:** No requiere datos externos. Simula internamente el entorno (GridWorld de 8x8) y genera obstáculos distribuidos espacialmente utilizando la función `make_blobs` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza módulos nativos como `collections`, `random` y `time`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/DQN.py`
- **Recursos externos:** No aplica (el entorno es dinámico y generado en tiempo de ejecución; no requiere conexión a APIs, variables de entorno ni pesos preentrenados).
- **Observaciones:** El script adapta magistralmente `scikit-learn` para Reinforcement Learning (área para la que no está diseñado) utilizando `partial_fit` y `warm_start=True` para permitir el aprendizaje online. En el código, la línea para guardar el gráfico (`plt.savefig`) está comentada (`#`); **es necesario descomentarla y modificar la ruta** absoluta (`/home/usuario/01_dqn_robotics.png`) por un directorio relativo válido en tu sistema (por ejemplo, `./01_dqn_robotics.png`) antes de la ejecución para que se genere el archivo.

### SARSA (Navegación en Laberintos)

- **Categoría:** Aprendizaje Automático [Aprendizaje por Refuerzo (Reinforcement Learning)]
- **Archivo:** `aprendizaje_automatico/SARSA.py`
- **Función básica:** Implementa el algoritmo de Aprendizaje por Refuerzo *On-Policy* SARSA (State-Action-Reward-State-Action) para entrenar a un agente a encontrar la ruta óptima hacia una meta dentro de un laberinto. Resuelve un problema de toma de decisiones secuenciales penalizando colisiones y recompensando la llegada al objetivo. Produce como salida el progreso del entrenamiento en consola y genera una imagen (`01_sarsa_laberinto.png`) que muestra la ruta trazada en el entorno y un mapa de calor de los valores de la Tabla Q.
- **Entrada requerida:** No requiere datos externos. Simula internamente un entorno de cuadrícula (GridWorld de 10x10) y genera obstáculos en forma de muro curvo utilizando la función `make_moons` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza módulos nativos de Python como `random` y `time`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/SARSA.py`
- **Recursos externos:** No aplica (el entorno y los obstáculos se generan algorítmicamente en tiempo de ejecución; no se requieren conexiones a APIs ni pesos de modelos previos).
- **Observaciones:** En el código original, la línea encargada de guardar el gráfico (`plt.savefig`) está comentada (`#`). **Es necesario descomentarla y modificar la ruta absoluta** (`/home/usuario/01_sarsa_laberinto.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_sarsa_laberinto.png`) antes de la ejecución si deseas que se genere el archivo `.png`.

### Policy Gradients & Policy Search (Control de Vuelo de Drones)

- **Categoría:** Aprendizaje Automático [Aprendizaje por Refuerzo (Policy-Based RL)]
- **Archivo:** `aprendizaje_automatico/Policy Gradients & Policy Search.py`
- **Función básica:** Implementa una aproximación de los métodos de Gradiente de Política utilizando el Cross-Entropy Method (CEM) con `MLPClassifier`. Entrena a un agente (un dron) para navegar desde un punto inicial hasta una meta esquivando zonas de turbulencia. En lugar de aprender el valor de las acciones, la red aprende directamente las probabilidades de tomar acciones exitosas basándose en las trayectorias de "élite". Produce reportes de rendimiento por generación en consola y genera una imagen (`01_policy_drone.png`) con el mapa del espacio aéreo, la ruta aprendida y la evolución de las recompensas medias y máximas.
- **Entrada requerida:** No requiere datos externos. Simula internamente un espacio aéreo (cuadrícula 20x20) y utiliza `make_blobs` de `scikit-learn` para generar proceduralmente zonas de turbulencia térmica.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza `random` y `warnings`, librerías nativas de Python).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Policy Gradients & Policy Search.py`
- **Recursos externos:** No aplica (el entorno es dinámico y generado en tiempo de ejecución; no requiere conexión a APIs, variables de entorno ni pesos preentrenados).
- **Observaciones:** El script utiliza `MLPClassifier` con `partial_fit` para realizar actualizaciones manuales continuas a partir de los episodios exitosos. En el código, la línea encargada de guardar el gráfico (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta** (`/home/usuario/01_policy_drone.png`) por un directorio local válido en el sistema (por ejemplo, `./01_policy_drone.png`) antes de la ejecución para poder obtener la imagen.

### PPO (Proximal Policy Optimization para Alineación de LLMs)

- **Categoría:** Aprendizaje Automático [Aprendizaje por Refuerzo (Reinforcement Learning / RLHF)]
- **Archivo:** `aprendizaje_automatico/PPO.py`
- **Función básica:** Simula el proceso de alineación de un Modelo de Lenguaje Grande (LLM) mediante RLHF (Reinforcement Learning from Human Feedback) utilizando el algoritmo PPO. Entrena un Modelo de Recompensa (Reward Model) para evaluar respuestas y ajusta la política del LLM aplicando un mecanismo de "clipping" para evitar cambios drásticos y destructivos en la red. Logra transformar un modelo con tendencias tóxicas o evasivas en uno que prioriza respuestas útiles y seguras. Produce métricas comparativas en consola y genera una imagen (`01_ppo_alineacion_llm.png`) que ilustra la evolución de la recompensa y el drástico cambio de comportamiento a lo largo de las épocas.
- **Entrada requerida:** No requiere datos externos. Simula internamente un espacio latente de instrucciones o *prompts* (3,000 embeddings de 10 dimensiones) y genera retroalimentación humana estocástica utilizando la función `make_classification` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza la librería nativa de Python `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/PPO.py`
- **Recursos externos:** No aplica (los datos, los embeddings y el feedback humano se generan de forma sintética en tiempo de ejecución; no requiere descargar modelos de lenguaje reales ni conexiones a APIs).
- **Observaciones:** El script adapta de forma muy creativa los modelos estándar de `scikit-learn` (`SGDClassifier` y `SGDRegressor` con `partial_fit`) para emular políticas de LLMs y Modelos de Recompensa, implementando manualmente el filtro de *Clipping* propio de PPO. En el código proporcionado, la instrucción para guardar la gráfica (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_ppo_alineacion_llm.png`) por un directorio relativo o válido en tu sistema (ej. `./01_ppo_alineacion_llm.png`) antes de la ejecución si deseas exportar la imagen.

### Actor-Critic (Control Continuo en Robótica Industrial)

- **Categoría:** Aprendizaje Automático [Aprendizaje por Refuerzo (Reinforcement Learning)]
- **Archivo:** `aprendizaje_automatico/Actor-Critic.py`
- **Función básica:** Implementa una arquitectura híbrida Actor-Crítico para resolver un problema de control continuo en un brazo robótico industrial (cinemática 2D). El "Actor" aprende la política dictando movimientos continuos (velocidad de giro de los motores), mientras que el "Crítico" evalúa la calidad del estado resultante y calcula el error para guiar al Actor. Produce métricas de entrenamiento en consola y genera una visualización (`01_actor_critic_robotics.png`) que muestra el diagrama cinemático del robot alcanzando su objetivo y las curvas de convergencia conjuntas.
- **Entrada requerida:** No requiere datos externos. El script simula internamente las matemáticas del entorno físico (cinemática directa) y genera estocásticamente las coordenadas objetivo (zonas de soldadura) empleando la función `make_blobs` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza módulos nativos como `math`, `random` y `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Actor-Critic.py`
- **Recursos externos:** No aplica (la dinámica del entorno y los objetivos se calculan y generan en tiempo de ejecución; no requiere descargar pesos, configurar variables de entorno ni conectar a APIs).
- **Observaciones:** Utiliza dos modelos `MLPRegressor` de forma creativa con el método `partial_fit` para simular la actualización iterativa del Actor y del Crítico. En el código original, la instrucción para guardar la imagen (`plt.savefig`) está comentada (`#`). **Deberás descomentarla y modificar la ruta absoluta** (`/home/usuario/01_actor_critic_robotics.png`) por una ruta válida en tu sistema local (por ejemplo, `./01_actor_critic_robotics.png`) antes de la ejecución para que se guarde el archivo gráfico.

### Regresión Lineal (Pronóstico de Ventas para Tienda Minorista)

- **Categoría:** Aprendizaje Supervisado (Regresión Lineal / Predicción de Ventas)
- **Archivo:** `aprendizaje_automatico/regresion_lineal.py`
- **Función básica:** Implementa un flujo completo de Ciencia de Datos para desarrollar un modelo predictivo de regresión lineal orientado a pronosticar las ventas de una tienda minorista basándose en múltiples características temporales, operativas y de entorno (como días de la semana, presencia de festivos y promociones). Realiza un Análisis Exploratorio de Datos (EDA) exhaustivo que incluye diagramas de dispersión, diagramas de caja segmentados y una matriz de correlación completa. Normaliza los datos utilizando `MinMaxScaler`, divide el conjunto en entrenamiento y prueba, entrena un modelo de `LinearRegression` y calcula métricas cuantitativas clave de rendimiento en consola ($R^2$ Score y RMSE en prueba), generando además visualizaciones gráficas de dispersión (valores reales frente a predichos) y una comparativa de línea temporal.
- **Entrada requerida:** Datos tabulares históricos de ventas en formato CSV. El script descarga de forma automatizada el archivo `Ventas.csv` desde un repositorio remoto en GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/regresion_lineal.py`
- **Recursos externos:** Requiere conexión a internet para descargar el archivo de origen remoto (`https://github.com/gianlucafran/regresion-lineal-tienda-minorista/raw/refs/heads/main/Ventas.csv`). No necesita claves API, variables de entorno ni pesos preentrenados.
- **Observaciones:** El script incluye la normalización completa de las características explicativas mediante `MinMaxScaler` y elimina explícitamente las columnas no deseadas antes de la partición de los datos con `train_test_split`. Asegúrate de verificar las rutas de guardado si incorporas instrucciones de exportación de gráficos con `plt.savefig`.


### Regresión Logística (Predicción de Fuga de Clientes / Churn)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación Binaria)]
- **Archivo:** `aprendizaje_automatico/regresion_logistica.py`
- **Función básica:** Implementa un flujo completo de Ciencia de Datos para predecir la rotación de clientes (*churn*) en una compañía de telecomunicaciones. Realiza un Análisis Exploratorio de Datos (EDA) con múltiples visualizaciones estadísticas (boxplots, gráficos circulares, histogramas y matrices de covarianza y correlación), aplica ingeniería de características (creación de ratios financieros y segmentación por fidelidad), normaliza los datos con `StandardScaler`, y entrena un modelo de regresión logística. Produce como salida métricas de rendimiento en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) y diversas gráficas analíticas.
- **Entrada requerida:** Datos tabulares en formato CSV con información de clientes de telecomunicaciones. El script no requiere un archivo local, ya que descarga automáticamente `train.csv` desde un repositorio de GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/regresion_logistica.py`
- **Recursos externos:** Requiere conexión a internet para descargar el dataset (`https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`). No necesita pesos preentrenados ni claves API.
- **Observaciones:** El script procesa de forma intensiva variables categóricas mediante mapeos personalizados y codificación (*one-hot encoding* mediante `get_dummies` para la segmentación de fidelidad), preparando una matriz numérica limpia y estandarizada para alimentar el modelo de regresión logística.

### K-NN (Vecinos Más Cercanos)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado]
- **Archivo:** `aprendizaje_automatico/K-NN.py`
- **Función básica:** Sistemas de recomendación simples basados en vectores de características espaciales. *(Verificar contra el código.)*
- **Entrada requerida:** Vectores de características espaciales. *(Verificar si el script carga un archivo o genera datos.)*
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. Emplea también el módulo nativo `warnings`.
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/K-NN.py`
- **Recursos externos:** *(Completar tras revisar el código.)*
- **Observaciones:** *(Completar tras revisar el código.)*

### Naive Bayes (Filtro de Spam / Clasificación de Texto)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación de Texto)]
- **Archivo:** `aprendizaje_automatico/Naive Bayes.py`
- **Función básica:** Implementa un clasificador de texto utilizando el algoritmo Multinomial Naive Bayes y vectorización TF-IDF para filtrar correos electrónicos (clasificación binaria entre correos legítimos "Ham" y anuncios comerciales "Spam"). Realiza la transformación y limpieza de textos del dataset *20 Newsgroups*, entrena el modelo aplicando el suavizado de Laplace, evalúa su precisión y prueba el sistema con mensajes en tiempo real. Produce como salida métricas completas en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) y genera una imagen (`01_naive_bayes_spam.png`) que ilustra la matriz de confusión junto con un gráfico de barras horizontales con las palabras clave de mayor peso probabilístico.
- **Entrada requerida:** Datos de texto crudo. El script no requiere archivos externos locales, ya que descarga y filtra automáticamente el dataset incorporado `fetch_20newsgroups` de `scikit-learn` utilizando dos categorías específicas (*sci.space* y *misc.forsale*).
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También emplea el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Naive Bayes.py`
- **Recursos externos:** Requiere conexión a internet únicamente durante la primera ejecución para descargar el dataset *20 Newsgroups* desde los repositorios oficiales de scikit-learn. No necesita claves API, variables de entorno ni pesos preentrenados de modelos pesados.
- **Observaciones:** En el código original proporcionado, la instrucción para guardar la gráfica (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_naive_bayes_spam.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_naive_bayes_spam.png`) antes de la ejecución para que el archivo de imagen se genere correctamente.

### SVM (Support Vector Machines para Biometría y Alta Dimensionalidad)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación Multiclase)]
- **Archivo:** `aprendizaje_automatico/SVM.py`
- **Función básica:** Implementa una Máquina de Vectores de Soporte (SVM) con un kernel RBF combinada con Análisis de Componentes Principales (PCA/Eigenfaces) para resolver un problema de reconocimiento facial biométrico. El sistema procesa miles de píxeles por imagen, reduce la dimensionalidad para extraer los patrones latentes clave del rostro, entrena el clasificador y evalúa su rendimiento con datos no vistos. Produce métricas detalladas en consola (reporte de clasificación y matriz de confusión) y genera dos gráficos: una galería de predicciones con matriz de confusión (`01_svm_biometria_rostros.png`) y una visualización de las eigenfaces descubiertas (`02_svm_eigenfaces.png`).
- **Entrada requerida:** Datos de imagen en forma de vectores planos de alta dimensión. El script no requiere cargar archivos locales externos, ya que descarga y carga automáticamente el dataset biométrico incorporado `fetch_lfw_people` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza módulos nativos como `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/SVM.py`
- **Recursos externos:** Requiere conexión a internet únicamente durante la primera ejecución para descargar el dataset *Labeled Faces in the Wild (LFW)* desde los repositorios oficiales de `scikit-learn`. No requiere claves API, variables de entorno ni pesos preentrenados pesados.
- **Observaciones:** Las líneas de código encargadas de exportar las imágenes (`plt.savefig`) se encuentran comentadas (`#`) en el script original. **Deberás descomentarlas y adaptar las rutas absolutas** (`/home/usuario/01_svm_biometria_rostros.png` y `/home/usuario/02_svm_eigenfaces.png`) por directorios locales válidos en tu equipo antes de ejecutar el script si deseas guardar físicamente los archivos gráficos.

### LDA (Análisis Discriminante Lineal para Riesgo Crediticio)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación / Reducción de Dimensionalidad Supervisada)]
- **Archivo:** `aprendizaje_automatico/LDA.py`
- **Función básica:** Implementa el algoritmo de Análisis Discriminante Lineal (LDA) para resolver un problema de clasificación y scoring en riesgo crediticio (detección de defaults o malos pagadores). A diferencia de métodos no supervisados como PCA, LDA maximiza de forma supervisada la separabilidad entre las clases. Proyecta un conjunto multidimensional de variables financieras a un espacio de una sola dimensión (1D), generando directamente un "Credit Score". Produce métricas clave de evaluación en consola (matriz de confusión, puntaje ROC-AUC y reporte de clasificación) y genera una imagen (`01_lda_riesgo_credito.png`) con la distribución unidimensional del score discriminante y la matriz de confusión visual.
- **Entrada requerida:** Datos tabulares cuantitativos simulados de solicitudes financieras. El script no requiere cargar archivos locales externos, ya que genera internamente los perfiles de riesgo mediante la función `make_classification` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (Incluye también el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/LDA.py`
- **Recursos externos:** No aplica (los datos y las perfiles financieros se simulan algorítmicamente en tiempo de ejecución; no se requieren conexiones a internet, APIs ni pesos preentrenados).
- **Observaciones:** En el código original, la instrucción encargada de exportar la gráfica (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_lda_riesgo_credito.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_lda_riesgo_credito.png`) antes de la ejecución para que el archivo de imagen se guarde de forma correcta.

### Árboles de Decisión para Diagnóstico Médico Guiado de Cáncer de Mama

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado / Clasificación Binaria].
- **Archivo:** `aprendizaje_automatico/decision_tree.py`
- **Función básica:** Entrena un modelo de Árbol de Decisión para clasificar tumores de mama como benignos o malignos utilizando el dataset Breast Cancer Wisconsin (Diagnostic) incluido en Scikit-Learn. El programa divide los datos en conjuntos de entrenamiento y prueba, entrena el modelo con el criterio de impureza de Gini y limita su profundidad a tres niveles para facilitar la interpretación de las reglas de decisión. Evalúa las predicciones mediante accuracy, matriz de confusión y reporte de clasificación. También calcula la importancia relativa de las características y genera gráficos del árbol y de las ocho variables más relevantes.
- **Entrada requerida:** No requiere un archivo CSV externo. Utiliza el dataset `load_breast_cancer()` incluido en Scikit-Learn, con características numéricas de muestras celulares y una variable objetivo binaria: maligno (0) o benigno (1).
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/decision_tree.py`
- **Recursos externos:** Dataset Breast Cancer Wisconsin (Diagnostic) distribuido con Scikit-Learn. No requiere acceso a internet para descargar un dataset externo ni utiliza APIs o modelos preentrenados.
- **Observaciones:** Utiliza `DecisionTreeClassifier` con `criterion='gini'`, `max_depth=3` y `random_state=42`. La división entrenamiento/prueba es del 80 % y 20 %, respectivamente, con estratificación de las clases. Calcula la importancia de las características y guarda una imagen con dos gráficos en `/home/usuario/01_arbol_decision_medico.png`. Esta ruta debe modificarse si no existe en el entorno de ejecución. El modelo se presenta como una demostración educativa de clasificación e interpretabilidad; no debe utilizarse para diagnosticar pacientes reales sin validación clínica independiente. Los resultados y métricas concretos quedan pendientes de ejecución y verificación.

### Random Forest para Predicción de Abandono de Clientes Telco

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado / Clasificación Binaria].
- **Archivo:** `aprendizaje_automatico/random_forest.py`
- **Función básica:** Implementa un modelo de clasificación basado en Random Forest para predecir el abandono de clientes (*Customer Churn*) en una empresa de telecomunicaciones. El código comienza con un análisis exploratorio del dataset, incluyendo dimensiones, tipos de datos, valores nulos, estadísticas descriptivas y visualizaciones de distribuciones y relaciones entre variables. Posteriormente, transforma variables categóricas mediante diccionarios de mapeo, calcula matrices de covarianza y correlación de Pearson y Spearman, y realiza ingeniería de características mediante ratios de cargos y segmentación de la antigüedad de los clientes. También normaliza las variables `tenure`, `MonthlyCharges` y `TotalCharges`. Finalmente, separa los datos en conjuntos de entrenamiento y prueba, entrena un clasificador `RandomForestClassifier` y evalúa sus predicciones mediante accuracy, reporte de clasificación y matriz de confusión.
- **Entrada requerida:** Dataset tabular de clientes de telecomunicaciones en formato CSV, obtenido desde el repositorio de GitHub especificado en el código. Requiere columnas como `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `MonthlyCharges`, `TotalCharges` y `Churn`, además de variables relacionadas con los servicios contratados, la seguridad, el soporte técnico, el tipo de contrato y el método de pago.
- **Dependencias:** `pandas`, `numpy`, `matplotlib`, `seaborn` y `scikit-learn`.
- **Instalación específica:** `pip install pandas numpy matplotlib seaborn scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/random_forest.py`
- **Recursos externos:** Dataset CSV alojado en GitHub: `https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`. Requiere conexión a internet para descargar los datos. No utiliza APIs externas ni modelos preentrenados.
- **Observaciones:** El modelo implementado es `RandomForestClassifier` de `sklearn.ensemble`. La partición de los datos utiliza un 80 % para entrenamiento y un 20 % para prueba, con `random_state=42`. La normalización se aplica a las variables numéricas seleccionadas, aunque no es un requisito general para los modelos Random Forest. La codificación categórica mediante diccionarios puede generar valores nulos si aparecen categorías no contempladas. Además, debe verificarse que `TotalCharges` contenga valores numéricos válidos y que las variables de entrada no tengan valores nulos o infinitos antes del entrenamiento. El código muestra las visualizaciones en pantalla e imprime las métricas en consola, pero no guarda las gráficas ni exporta el modelo entrenado.

### Gradient Boosting (Predicción General Médica y Ensamble Secuencial)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Ensamble / Boosting)]
- **Archivo:** `aprendizaje_automatico/Gradient_Boosting.py`
- **Función básica:** Implementa un modelo de Gradient Boosting Classifier (`GradientBoostingClassifier`) para resolver un problema de diagnóstico médico y predicción general (clasificación binaria de masas mamarias usando el dataset *Breast Cancer Wisconsin*). Entrena secuencialmente múltiples árboles de decisión débiles ("weak learners") donde cada nuevo árbol aprende a corregir los errores residuales del anterior. Produce métricas detalladas en consola (exactitud, matriz de confusión y reporte de clasificación) y genera una imagen (`01_gradient_boosting_medicina.png`) que muestra la curva de evolución del error de entrenamiento/prueba por iteración y el ranking de las 10 características clínicas más importantes.
- **Entrada requerida:** Datos tabulares cuantitativos biomédicos. El script no requiere cargar archivos locales externos, ya que descarga y carga automáticamente el dataset nativo `load_breast_cancer` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (También utiliza el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Gradient_Boosting.py`
- **Recursos externos:** No aplica (el dataset está integrado en la librería scikit-learn; no requiere descargar archivos CSV remotos ni utilizar claves API).
- **Observaciones:** En el código original, la instrucción encargada de exportar la gráfica (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_gradient_boosting_medicina.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_gradient_boosting_medicina.png`) antes de la ejecución para que el archivo de imagen se guarde de forma correcta.

### Gradient Boosting Classifier (Predicción de Fuga de Clientes / Churn)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado [Clasificación Conjunta / Ensemble Learning]
- **Archivo:** `aprendizaje_automatico/gbm.py`
- **Función básica:** Implementa un flujo completo de Ciencia de Datos y Machine Learning para predecir la rotación de clientes (*churn*) en una compañía de telecomunicaciones utilizando un modelo de ensamble secuencial de árboles de decisión (`GradientBoostingClassifier`). Ejecuta un Análisis Exploratorio de Datos (EDA) exhaustivo con visualizaciones estadísticas y matrices de correlación (Pearson, Spearman y covarianza), realiza ingeniería de características con creación de ratios financieros y segmentación por fidelidad, normaliza variables continuas con `StandardScaler`, particiona el conjunto de datos y entrena el modelo de Boosting. Produce como resultado métricas cuantitativas clave en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) para evaluar su rendimiento predictivo frente a datos no vistos.
- **Entrada requerida:** Datos tabulares en formato CSV conteniendo atributos sociodemográficos, de contrato y transaccionales de clientes de telecomunicaciones. El script descarga automáticamente el dataset `train.csv` desde un repositorio de GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/gbm.py`
- **Recursos externos:** Requiere conexión a internet para descargar el archivo de origen remoto (`https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`). No necesita claves API, variables de entorno ni pesos preentrenados.
- **Observaciones:** El script procesa rigurosamente variables categóricas mediante mapeos personalizados y codificación (*one-hot encoding* para la segmentación de fidelidad), estructurando una matriz numérica limpia y estandarizada que optimiza la convergencia del algoritmo de Gradient Boosting.

### XGBoost Classifier (Predicción de Fuga de Clientes / Churn)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación Conjunta / Gradient Boosting Optimizado)]
- **Archivo:** `aprendizaje_automatico/xgboost.py`
- **Función básica:** Implementa un flujo completo de Ciencia de Datos para predecir la rotación de clientes (*churn*) en una compañía de telecomunicaciones utilizando el algoritmo de optimización por gradiente extremo (`XGBClassifier`). Ejecuta un Análisis Exploratorio de Datos (EDA) exhaustivo mediante múltiples representaciones estadísticas y matrices de covarianza y correlación, realiza ingeniería de características con creación de ratios financieros y segmentación por fidelidad, normaliza variables continuas empleando `StandardScaler`, separa los conjuntos de entrenamiento y prueba, y entrena el modelo optimizado. Produce como resultado métricas cuantitativas clave en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) para evaluar su rendimiento frente a datos no vistos.
- **Entrada requerida:** Datos tabulares en formato CSV conteniendo atributos sociodemográficos, de contrato y transaccionales de clientes de telecomunicaciones. El script descarga automáticamente el dataset `train.csv` desde un repositorio de GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `xgboost`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy xgboost scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/xgboost.py`
- **Recursos externos:** Requiere conexión a internet para descargar el archivo de origen remoto (`https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`). No necesita claves API, variables de entorno ni pesos preentrenados.
- **Observaciones:** El script procesa de manera rigurosa las variables categóricas mediante mapeos personalizados y codificación (*one-hot encoding* para la segmentación de fidelidad), construyendo una matriz numérica limpia y estandarizada que aprovecha al máximo la velocidad de cálculo, regularización y eficiencia del framework XGBoost.

### LightGBM Classifier (Predicción de Fuga de Clientes / Churn)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación Conjunta / Gradient Boosting de Alta Eficiencia)]
- **Archivo:** `aprendizaje_automatico/lightgbm.py`
- **Función básica:** Implementa un flujo integral de Ciencia de Datos para predecir la rotación de clientes (*churn*) en una compañía de telecomunicaciones utilizando el framework LightGBM (`LGBMClassifier`). Ejecuta un Análisis Exploratorio de Datos (EDA) exhaustivo que incluye matrices de correlación (Pearson, Spearman y covarianza), realiza ingeniería de características con creación de ratios financieros y segmentación por fidelidad, normaliza variables continuas con `StandardScaler`, separa los conjuntos de entrenamiento y prueba, y entrena un modelo basado en árboles orientados a hojas de alto rendimiento y bajo consumo de memoria. Produce como resultado métricas cuantitativas clave en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) para evaluar su capacidad predictiva frente a datos no vistos.
- **Entrada requerida:** Datos tabulares en formato CSV conteniendo atributos sociodemográficos, de contrato y transaccionales de clientes de telecomunicaciones. El script descarga automáticamente el dataset `train.csv` desde un repositorio de GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `lightgbm`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy lightgbm scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/lightgbm.py`
- **Recursos externos:** Requiere conexión a internet para descargar el archivo de origen remoto (`https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`). No necesita claves API, variables de entorno ni pesos preentrenados.
- **Observaciones:** El script procesa detalladamente las variables categóricas mediante mapeos personalizados y codificación (*one-hot encoding* para la segmentación de fidelidad), construyendo una matriz numérica limpia y estandarizada que optimiza la velocidad de entrenamiento y la eficiencia en la estructura de hojas propia de LightGBM.

### CatBoost Classifier (Predicción de Fuga de Clientes / Churn)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Clasificación Conjunta / Gradient Boosting Simétrico)]
- **Archivo:** `aprendizaje_automatico/catboost.py`
- **Función básica:** Implementa un flujo integral de Ciencia de Datos para predecir la rotación de clientes (*churn*) en una compañía de telecomunicaciones utilizando el algoritmo de gradient boosting simétrico optimizado de CatBoost (`CatBoostClassifier`). Ejecuta un Análisis Exploratorio de Datos (EDA) completo mediante resúmenes estadísticos y mapas de calor de covarianza y correlación (Pearson y Spearman), realiza ingeniería de características con creación de ratios financieros y segmentación por fidelidad, normaliza variables numéricas empleando `StandardScaler`, separa los conjuntos de entrenamiento y prueba, y entrena el modelo de ensamble. Produce como resultado métricas cuantitativas clave en consola (Accuracy, Matriz de Confusión y Reporte de Clasificación) para evaluar su rendimiento frente a datos no vistos.
- **Entrada requerida:** Datos tabulares en formato CSV conteniendo atributos sociodemográficos, de contrato y transaccionales de clientes de telecomunicaciones. El script descarga automáticamente el dataset `train.csv` desde un repositorio de GitHub.
- **Dependencias:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `catboost`, `scikit-learn`.
- **Instalación específica:** `pip install pandas matplotlib seaborn numpy catboost scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/catboost.py`
- **Recursos externos:** Requiere conexión a internet para descargar el archivo de origen remoto (`https://github.com/gianlucafran/Red-neuronal-dataset-telco/raw/refs/heads/main/train.csv`). No necesita claves API, variables de entorno ni pesos preentrenados.
- **Observaciones:** El script procesa minuciosamente las variables categóricas mediante mapeos personalizados y codificación de etiquetas, construyendo una matriz numérica estandarizada que permite aprovechar la alta eficiencia y robustez de CatBoost frente a sobreajustes.


### Regresión Ridge (Previsión en Series Temporales Económicas y Estabilización L2)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Regresión Regularizada / Econometría)]
- **Archivo:** `aprendizaje_automatico/Regresion Ridge.py`
- **Función básica:** Implementa un modelo de Regresión Ridge con penalización L2 (`Ridge`) para realizar previsiones y forecasting en series temporales macroeconómicas (como el crecimiento del PIB mensual). Resuelve el problema crítico de la multicolinealidad severa entre regresores económicos altamente correlacionados (empleo, valor agregado, tasas de interés), estabilizando los coeficientes e impidiendo que el modelo sufra de sobreajuste. Produce métricas comparativas detalladas en consola frente a la regresión lineal ordinaria OLS (MSE, MAE y $R^2$) y genera una visualización (`01_ridge_series_temporales.png`) que incluye el pronóstico cronológico a futuro y un gráfico de barras comparativo de la contracción de coeficientes (*shrinkage*).
- **Entrada requerida:** Datos tabulares secuenciales de indicadores macroeconómicos. El script genera de forma algorítmica un dataset simulado de series temporales utilizando la función `make_regression` con control de rango efectivo (`effective_rank=2`) e índice de fechas mensuales de pandas, por lo que no requiere archivos locales externos ni conexiones a internet.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (Incluye el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Regresion Ridge.py`
- **Recursos externos:** No aplica (los datos macroeconómicos se simulan mediante funciones internas de scikit-learn; no se requiere descargar CSVs remotos, claves API ni pesos preentrenados).
- **Observaciones:** En el código original proporcionado, la línea que exporta el gráfico (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y ajustar la ruta absoluta rígida** (`/home/usuario/01_ridge_series_temporales.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_ridge_series_temporales.png`) antes de la ejecución para garantizar que el archivo de imagen se guarde adecuadamente.

### Regresión Lasso (Selección de Genes en Bioinformática y Regularización L1)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Regresión Regularizada / Selección Automática de Características)]
- **Archivo:** `aprendizaje_automatico/Regresión Lasso.py`
- **Función básica:** Implementa un modelo de Regresión Lasso con penalización L1 (`Lasso`) diseñado específicamente para resolver problemas con exceso masivo de variables frente al número de muestras ($p \gg n$), como en la expresión génica y análisis bioinformático. La penalización L1 anula por completo (*shrinkage to zero*) los coeficientes de las características irrelevantes, realizando selección de variables de forma totalmente nativa. Produce métricas cuantitativas de rendimiento en consola (R2 Score, MSE) y genera una visualización gráfica (`01_lasso_bioinformatica.png`) que compara el perfil de coeficientes en todo el genoma frente a OLS y muestra el panel aislado de biomarcadores relevantes.
- **Entrada requerida:** Datos tabulares cuantitativos de alta dimensión. El script no requiere archivos externos locales, ya que genera de manera algorítmica un dataset sintético bioinformático de alta dimensionalidad utilizando la función `make_regression` con control explícito de muestras y características.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (Incluye también el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Regresión Lasso.py`
- **Recursos externos:** No aplica (los datos genómicos de expresión se simulan de forma algorítmica en tiempo de ejecución; no se requiere descargar archivos CSV remotos, claves API ni pesos preentrenados).
- **Observaciones:** En el código original facilitado, la instrucción para guardar la gráfica (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_lasso_bioinformatica.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_lasso_bioinformatica.png`) antes de la ejecución para que el archivo de imagen se guarde correctamente.

### Elastic Net (Modelado Estadístico Avanzado y Regresión Penalizada Híbrida)

- **Categoría:** Aprendizaje Automático [Aprendizaje Supervisado (Regresión Regularizada Avanzada / Combinación L1 y L2)]
- **Archivo:** `aprendizaje_automatico/Elastic Net.py`
- **Función básica:** Implementa un modelo de regresión penalizada híbrida (`ElasticNetCV`) que combina linealmente las penalizaciones L1 (Lasso) y L2 (Ridge). Resuelve el problema crítico de la multicolinealidad extrema (provocada en este caso por una expansión polinómica de grado 2 que eleva masivamente la dimensionalidad de los datos médicos de diabetes), seleccionando grupos enteros de variables altamente correlacionadas y eliminando al mismo tiempo el ruido irrelevante. Produce métricas comparativas detalladas en consola frente a Ridge y Lasso (R2 Score de validación cruzada y cantidad de características retenidas) y genera una visualización gráfica (`01_elastic_net_modelado.png`) que compara el perfil de coeficientes de los tres modelos y un gráfico de dispersión de predicción frente a la realidad.
- **Entrada requerida:** Datos tabulares cuantitativos de pacientes. El script no requiere cargar archivos locales externos, ya que carga directamente el dataset clínico nativo de pacientes con diabetes mediante la función `load_diabetes` de `scikit-learn`.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`. (Incluye también el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scikit-learn seaborn`
- **Ejecución:** `python aprendizaje_automatico/Elastic Net.py`
- **Recursos externos:** No aplica (el dataset clínico proviene de scikit-learn; no requiere descargas de archivos CSV remotos, claves API ni pesos preentrenados).
- **Observaciones:** En el código original proporcionado, la instrucción para guardar la gráfica (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_elastic_net_modelado.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_elastic_net_modelado.png`) antes de la ejecución para que el archivo de imagen se guarde de manera adecuada.

### Gaussian Process Regressor (Optimización Bayesiana de Hiperparámetros)

- **Categoría:** Aprendizaje Automático [Aprendizaje No Paramétrico / Modelado Geoespacial y Optimización Bayesiana]
- **Archivo:** `aprendizaje_automatico/GP.py`
- **Función básica:** Implementa un modelo de Procesos Gaussianos con un kernel Matérn combinado (`GaussianProcessRegressor`) para resolver tareas de interpolación espacial (Kriging) y realizar optimización bayesiana de hiperparámetros en modelos de machine learning (como árboles de decisión). A diferencia de los métodos de búsqueda exhaustiva (Grid Search), los procesos gaussianos actúan como un modelo sustituto (*surrogate model*) que predice tanto la media esperada como la incertidud asociada ($\sigma$), permitiendo encontrar las combinaciones óptimas de hiperparámetros mediante funciones de adquisición inteligentes. Produce métricas e iteraciones detalladas en consola y genera una imagen (`01_gaussian_process_spatial.png`) con la predicción espacial bivariada de precios y su respectivo mapa de incertidumbre.
- **Entrada requerida:** Datos espaciales o temporales tabulares. El script carga de forma automatizada un subconjunto de 1000 muestras del dataset nativo `fetch_california_housing` de `scikit-learn`, el cual contiene coordenadas geográficas reales (latitud y longitud) junto con características socioeconómicas.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python aprendizaje_automatico/GP.py`
- **Recursos externos:** No aplica (los datos geoespaciales se obtienen de la librería integrada scikit-learn; no requiere descargas de archivos CSV remotos, claves API ni pesos preentrenados).
- **Observaciones:** El script exporta la visualización gráfica mediante `plt.savefig('/home/usuario/01_gaussian_process_spatial.png', ...)` utilizando una ruta absoluta rígida. **Es obligatorio modificar esta ruta** por un directorio local accesible en tu entorno (por ejemplo, `./01_gaussian_process_spatial.png`) antes de la ejecución para garantizar que el archivo de imagen se guarde exitosamente sin errores de permisos.


### OCR Local con OpenCV y Tesseract (Reconocimiento Óptico de Caracteres)

- **Categoría:** Visión por Computador / Procesamiento de Imágenes y OCR
- **Archivo:** `vision_computacional/OCR.py`
- **Función básica:** Implementa un pipeline completo de procesamiento de imágenes y reconocimiento óptico de caracteres (OCR) utilizando OpenCV y Tesseract. El flujo incluye validación interactiva de archivos locales, conversión a escala de grises, reducción de ruido preservando bordes mediante un filtro bilateral, binarización automática con el método de Otsu, ejecución de Tesseract OCR con configuración personalizable (OEM/PSM) y validación de paquetes de idioma. Produce como salida la extracción del texto reconocido, el almacenamiento organizado de cada etapa del filtrado y de la imagen original en disco, y una visualización gráfica comparativa en una cuadrícula con Matplotlib.
- **Entrada requerida:** Imágenes escaneadas locales de facturas, documentos o texto impreso en formatos estándar compatibles con OpenCV (como `.png`, `.jpg`, `.jpeg`). El programa solicita la ruta del archivo de manera interactiva por consola al ejecutarse.
- **Dependencias:** `numpy`, `matplotlib`, `opencv-python`, `pytesseract`.
- **Instalación específica:** `pip install numpy matplotlib opencv-python pytesseract`
- **Ejecución:** `python vision_computacional/OCR.py`
- **Recursos externos:** Requiere la instalación previa del programa independiente **Tesseract OCR** en el sistema operativo (junto con los paquetes de idioma requeridos, como `spa` para español o `eng` para inglés) y, de ser necesario, configurar la ruta del binario mediante `pytesseract.tesseract_cmd`.
- **Observaciones:** El script gestiona automáticamente la creación de un directorio de salida denominado `resultados_ocr/` para almacenar las imágenes binarizadas, en escala de grises, filtradas y el archivo de texto extraído (`_texto.txt`), además de guardar el resumen del pipeline como una figura en alta resolución (`pipeline_preprocesamiento.png`).

### CNN (Redes Neuronales Convolucionales y Clasificación de Imágenes)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Deep Learning)
- **Archivo:** `vision_computacional/cnn.py`
- **Función básica:** Implementa una Red Neuronal Convolucional (CNN) completa desde cero utilizando PyTorch para resolver tareas de clasificación de imágenes (demostración con el dataset clásico MNIST de dígitos escritos a mano). El pipeline arquitectónico incluye capas convolucionales (`nn.Conv2d`), funciones de activación no lineal (`ReLU`), capas de reducción espacial (`MaxPool2d`), aplanado de tensores (`Flatten`) y una capa densa de clasificación final (`nn.Linear`). Utiliza el optimizador Adam, minibatches mediante `DataLoader`, cálculo de pérdida con Cross Entropy y evaluación exhaustiva mediante métricas de `torchmetrics` (Accuracy, Precision y Recall) bajo modo de inferencia segura (`torch.no_grad()`).
- **Entrada requerida:** Tensores de imágenes normalizadas en escala de grises. El script descarga de manera automatizada el dataset MNIST estándar utilizando los submódulos de `torchvision.datasets`, por lo que no requiere cargar datasets locales externos.
- **Dependencias:** `torch`, `torchvision`, `torchmetrics`, `numpy`, `pandas`, `matplotlib`, `tqdm`.
- **Instalación específica:** `pip install torch torchvision torchmetrics numpy pandas matplotlib tqdm`
- **Ejecución:** `python vision_computacional/cnn.py`
- **Recursos externos:** Requiere conexión a internet únicamente en su primera ejecución para descargar el dataset MNIST desde los repositorios oficiales integrados en torchvision. No necesita claves API, variables de entorno ni pesos preentrenados externos.
  - **Observaciones:** El script detecta de manera inteligente la disponibilidad de hardware de aceleración (`cuda`), configurando automáticamente el dispositivo de cómputo para aprovechar GPUs NVIDIA si están presentes en el sistema. Asegúrate de verificar las dependencias de entorno de PyTorch compatibles con tu versión de CUDA antes de la ejecución.


### Canny Edge Detector (Detección de Bordes en Visión por Computador)

- **Categoría:** Visión por Computador / Procesamiento de Imágenes
- **Archivo:** `vision_computacional/Canny.py`
- **Función básica:** Implementa el algoritmo clásico de detección de bordes de Canny (`cv2.Canny`) para identificar contornos y transiciones abruptas de intensidad lumínica en una imagen. El pipeline comprende la carga del archivo, conversión a escala de grises, aplicación de un filtro de suavizado Gaussiano para mitigar el ruido y evitar falsos positivos, y la umbralización con histéresis mediante umbrales inferior y superior (`umbral_min` y `umbral_max`). Produce como resultado una visualización gráfica comparativa lado a lado mediante Matplotlib entre la imagen original a color y la máscara binaria resultante con los bordes detectados.
- **Entrada requerida:** Imágenes digitales locales en formatos estándar compatibles con OpenCV (como `.png`, `.jpg`, `.jpeg`). El script recibe la ruta del archivo directamente como parámetro en su función principal.
- **Dependencias:** `cv2` (opencv-python), `matplotlib`.
- **Instalación específica:** `pip install opencv-python matplotlib`
- **Ejecución:** `python vision_computacional/Canny.py`
- **Recursos externos:** No aplica (procesa imágenes estáticas locales del sistema de archivos; no requiere conexión a internet, APIs externas ni pesos preentrenados).
- **Observaciones:** En la sección final del script, la variable `ruta_archivo` contiene una ruta absoluta estática orientada a Windows (`C:\Users\Tecnologias\Downloads\...`). **Es obligatorio actualizar esta ruta** apuntando a una imagen válida que exista en tu sistema antes de ejecutar el programa. Adicionalmente, cuenta con una instrucción comentada (`cv2.imwrite`) para exportar la máscara de bordes en disco si se requiere almacenar el resultado.

### LoG (Laplacian of Gaussian - Filtro de Contornos y Segunda Derivada)

- **Categoría:** Visión por Computador / Procesamiento de Imágenes y Filtrado Espacial
- **Archivo:** `vision_computacional/Laplacian of Gaussian.py`
- **Función básica:** Implementa un filtro espacial avanzado de Laplacian of Gaussian (LoG) para la detección de contornos (*edge detection*) en imágenes. Combina un suavizado Gaussiano previo (para mitigar y suprimir el ruido del sensor) con el operador Laplaciano de segunda derivada espacial (`gaussian_laplace` de SciPy) para identificar cambios abruptos de intensidad lumínica a múltiples escalas. Produce una consola informativa detallada sobre los niveles de sigma y genera una galería visual comparativa en formato de cuadrícula (`01_filtro_log_contornos.png`) que contrasta la imagen original en escala de grises frente a las extracciones de contornos a diferentes radios de desviación estándar ($\sigma = 1.0, 2.0, 3.5$).
- **Entrada requerida:** Imágenes digitales rasterizadas. El script no requiere cargar archivos locales externos, ya que importa de forma automatizada la fotografía de alta resolución de la muestra estándar `china.jpg` integrada en scikit-learn.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`. (Incluye el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib scipy scikit-learn seaborn`
- **Ejecución:** `python vision_computacional/Laplacian of Gaussian.py`
- **Recursos externos:** No aplica (la imagen de prueba se carga directamente desde los datasets de scikit-learn; no requiere descargas de archivos CSV o imágenes remotas, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción encargada de exportar el gráfico (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y ajustar la ruta absoluta rígida** (`/home/usuario/01_filtro_log_contornos.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_filtro_log_contornos.png`) antes de la ejecución para que el archivo de imagen se guarde satisfactoriamente.

### SIFT (Scale-Invariant Feature Transform - Stitching de Imágenes y Panorámicas)

- **Categoría:** Visión por Computador / Detección de Características Locales y Homografía
- **Archivo:** `vision_computacional/SIFT.py`
- **Función básica:** Implementa un pipeline integral de visión por computador para realizar el emparejamiento de características y la creación de imágenes panorámicas (*image stitching*) mediante el algoritmo SIFT (`cv2.SIFT_create`). El flujo simula dos tomas con solapamiento a partir de una imagen base, extrae puntos clave invariantes a la escala y orientación junto con descriptores de 128 dimensiones, aplica emparejamiento por fuerza bruta con la prueba de razón de Lowe, calcula la matriz de perspectiva mediante el algoritmo robusto RANSAC y deforma (*warp*) los fotogramas para reconstituir un panorama continuo. Produce consola informativa detallada y genera tres archivos gráficos de salida (`01_sift_keypoints.png`, `02_sift_matches.png` y `03_sift_stitching_panorama.png`).
- **Entrada requerida:** Imágenes rasterizadas en color o escala de grises. El script no requiere cargar archivos externos, ya que simula automáticamente dos tomas solapadas dividiendo la fotografía de alta resolución `china.jpg` proveniente de los datasets de scikit-learn.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`. (Incluye también el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/SIFT.py`
- **Recursos externos:** No aplica (la imagen de prueba se obtiene directamente de scikit-learn; no requiere descargar archivos CSV o imágenes remotas de internet, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, las instrucciones para exportar los gráficos (`plt.savefig`) se encuentran comentadas (`#`). **Es obligatorio descomentarlas y ajustar las rutas absolutas rígidas** (por ejemplo, reemplazando `/home/usuario/01_sift_keypoints.png` por un directorio local válido como `./01_sift_keypoints.png`) antes de la ejecución para que las figuras se guarden correctamente.

### Detector de Esquinas de Harris (Seguimiento de Video y Flujo Óptico)

- **Categoría:** Visión por Computador / Extracción de Características y Rastreo Dinámico
- **Archivo:** `vision_computacional/Detector Harris.py`
- **Función básica:** Implementa el algoritmo clásico de detección de esquinas de Harris (`cv2.goodFeaturesToTrack` con `useHarrisDetector=True`) combinado con el algoritmo de flujo óptico de Lucas-Kanade (`cv2.calcOpticalFlowPyrLK`) para realizar el rastreo de puntos de interés a lo largo de una secuencia de video. Simula el movimiento de paneo de una cámara (tipo dron) aplicando transformaciones afines consecutivas sobre una imagen estática de alta resolución, extrae las esquinas estructurales más robustas del primer fotograma y dibuja dinámicamente las estelas de desplazamiento en los fotogramas sucesivos. Produce consola informativa detallada y genera una visualización gráfica dual (`01_harris_video_tracking.png`) que contrasta la inicialización de puntos de Harris frente al historial de trayectorias acumuladas.
- **Entrada requerida:** Secuencias de imágenes o fotogramas de video. El script genera de forma algorítmica una secuencia sintética de 30 fotogramas a partir de la fotografía de alta resolución `china.jpg` proveniente de scikit-learn, por lo que no requiere cargar archivos externos.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`. (Incluye el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/Detector Harris.py`
- **Recursos externos:** No aplica (las imágenes base se cargan directamente desde scikit-learn y los fotogramas se simulan mediante transformaciones afines en tiempo de ejecución; no requiere descargas de archivos remotos, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción para guardar el gráfico (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y ajustar la ruta absoluta rígida** (`/home/usuario/01_harris_video_tracking.png`) por un directorio local accesible en tu entorno (por ejemplo, `./01_harris_video_tracking.png`) antes de ejecutar el script para que la figura se guarde exitosamente.

### SURF (Speeded Up Robust Features - Reconocimiento de Objetos en Tiempo Real)

- **Categoría:** Visión por Computador / Detección y Reconocimiento de Objetos Invariantes
- **Archivo:** `vision_computacional/SURF.py`
- **Función básica:** Implementa un pipeline completo de reconocimiento y localización de objetos en tiempo real (simulando un feed de cámara web con transformaciones afines de rotación, escala y ruido gaussiano) utilizando el detector de características SURF (con mecanismo de *fallback* automático a ORB para compatibilidad en entornos estándar de Python). El flujo extrae el objeto de referencia de una imagen base, calcula los puntos clave e integra un emparejamiento ultrarrápido mediante FLANN (*Fast Library for Approximate Nearest Neighbors*). Aplica la prueba de razón de Lowe y calcula la matriz de homografía con RANSAC para proyectar y dibujar en tiempo real un rectángulo delimitador (*bounding box*) sobre el objeto detectado, midiendo simultáneamente la latencia y los fotogramas por segundo (FPS).
- **Entrada requerida:** Imágenes rasterizadas en color o escala de grises. El script no requiere cargar archivos locales externos, ya que extrae de forma automatizada un objeto de referencia (*target*) a partir de la fotografía de alta resolución `flower.jpg` integrada en scikit-learn.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`. (Incluye también el módulo nativo `warnings` y `time`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/SURF.py`
- **Recursos externos:** No aplica (la imagen base proviene de los datasets de scikit-learn y el video sintético se genera en tiempo de ejecución; no requiere descargas remota de CSVs o imágenes, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción para exportar el gráfico resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y ajustar la ruta absoluta rígida** (`/home/usuario/01_surf_realtime_recognition.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_surf_realtime_recognition.png`) antes de la ejecución para garantizar que la figura de resultados y rendimiento se guarde adecuadamente.

### Brute-Force Matcher (Reconocimiento Facial Simple con ORB y Distancia de Hamming)

- **Categoría:** Visión por Computador / Emparejamiento de Características y Autenticación Biométrica
- **Archivo:** `vision_computacional/Brute-Force Matcher.py`
- **Función básica:** Implementa un sistema de reconocimiento facial clásico basado en similitud topológica utilizando el extractor de características binarias ORB y un comparador de fuerza bruta (`cv2.BFMatcher` con métrica `NORM_HAMMING` y verificación cruzada `crossCheck=True`). El pipeline selecciona imágenes reales de rostros del dataset LFW, detecta puntos clave e identifica correspondencias mediante el cálculo de distancias de Hamming entre descriptores binarios, evaluando umbrales de aceptación para distinguir entre intentos genuinos y suplantaciones de identidad (impostores). Produce consola informativa detallada y genera una visualización gráfica dual (`01_bfmatcher_facial.png`) que contrasta las redes de coincidencias para un par genuino frente a un par impostor.
- **Entrada requerida:** Imágenes tabulares o rasterizadas de rostros. El script no requiere archivos externos locales, ya que descarga y carga automáticamente un subconjunto del dataset estándar de rostros reales *Labeled Faces in the Wild* (`fetch_lfw_people`) provisto por scikit-learn.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`. (Incluye también los módulos nativos `warnings` y `time`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/Brute-Force Matcher.py`
- **Recursos externos:** No aplica (el dataset se obtiene directamente desde scikit-learn; no requiere descargar archivos CSV o imágenes remotas de internet, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción para exportar el gráfico resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y ajustar la ruta absoluta rígida** (`/home/usuario/01_bfmatcher_facial.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_bfmatcher_facial.png`) antes de la ejecución para garantizar que el archivo de imagen se guarde satisfactoriamente.

### FLANN (Fast Library for Approximate Nearest Neighbors - Búsqueda Visual y KD-Trees)

- **Categoría:** Visión por Computador / Búsqueda de Vecinos Más Cercanos e Indexación Espacial
- **Archivo:** `vision_computacional/FLANN.py`
- **Función básica:** Implementa un sistema avanzado de búsqueda visual (*visual search*) para localizar de manera exacta una subimagen alterada (*query* con rotación y cambio de escala) dentro de una escena de alta resolución utilizando descriptores SIFT indexados mediante un árbol K-Dimensional (`cv2.FlannBasedMatcher` con `FLANN_INDEX_KDTREE`). El pipeline extrae características de la base de datos de la escena y del fragmento objetivo, realiza un emparejamiento optimizado y escalable, filtra falsos positivos con la prueba de razón de Lowe y calcula la homografía geométrica con RANSAC para proyectar y delimitar con un polígono verde el objeto en la escena original. Produce consola informativa detallada sobre tiempos de cómputo y genera un archivo gráfico compuesto (`01_flann_visual_search.png`).
- **Entrada requerida:** Imágenes ráster de escena y patrones de consulta. El script no requiere cargar archivos externos, ya que extrae y simula algorítmicamente un patrón rotado y escalado a partir de la fotografía estándar de alta resolución `china.jpg` de scikit-learn.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `opencv-python`. (Incluye los módulos nativos `warnings` y `time`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/FLANN.py`
- **Recursos externos:** No aplica (las imágenes provienen de scikit-learn y las transformaciones se calculan en tiempo de ejecución; no requiere descargas de archivos remotos, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) está comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_flann_visual_search.png`) por un directorio local accesible en tu entorno (por ejemplo, `./01_flann_visual_search.png`) antes de la ejecución para garantizar que el gráfico se guarde satisfactoriamente.

### RANSAC (Random Sample Consensus - Calibración Extrínseca de Cámaras y Robustez al Ruido)

- **Categoría:** Visión por Computador / Estimación de Pose (Pose Estimation) y Modelado Robusto
- **Archivo:** `vision_computacional/RANSAC.py`
- **Función básica:** Implementa un modelo robusto de estimación de la pose extrínseca de una cámara (`cv2.solvePnPRansac`) para calcular la matriz de rotación y traslación a partir de correspondencias de puntos 3D del mundo real y sus proyecciones 2D en el sensor. Simula una nube de puntos 3D mediante scikit-learn, proyecta las coordenadas según una cámara virtual con parámetros intrínsecos conocidos e introduce artificialmente un 30% de ruido severo (*outliers*) para contrastar el rendimiento frente al método clásico de mínimos cuadrados (`cv2.solvePnP`). Produce consola informativa detallada y genera una visualización gráfica dual (`01_ransac_calibracion_camaras.png`) que separa los inliers del ruido y demuestra la precisión exacta de la reproyección geométrica frente al fallo del modelo convencional.
- **Entrada requerida:** Nube de puntos tridimensionales y correspondencias espaciales. El script no requiere cargar archivos locales externos, ya que genera de manera algorítmica un conjunto de datos tridimensional mediante las funciones de soporte de scikit-learn (`make_blobs`).
- **Dependencias:** `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python). (Incluye el módulo nativo `warnings`).
- **Instalación específica:** `pip install numpy matplotlib seaborn opencv-python`
- **Ejecución:** `python vision_computacional/RANSAC.py`
- **Recursos externos:** No aplica (los datos espaciales y la simulación de proyección se procesan de forma interna mediante scikit-learn y OpenCV; no requiere descargar archivos de internet, claves API ni pesos preentrenados).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_ransac_calibracion_camaras.png`) por un directorio local válido en tu sistema (por ejemplo, `./01_ransac_calibracion_camaras.png`) antes de la ejecución para garantizar que el gráfico se guarde satisfactoriamente.

### AlexNet (Clasificación Base de Imágenes y Arquitectura Fundacional de Deep Learning)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Deep Learning y Convoluciones Profundas)
- **Archivo:** `vision_computacional/AlexNet.py`
- **Función básica:** Implementa la arquitectura histórica de redes neuronales convolucionales AlexNet (2012) utilizando PyTorch. El flujo incluye la descarga y transformación del dataset real CIFAR-10 con redimensionamiento a 227x227 y normalización estándar de ImageNet, construcción modular de bloques convolucionales de extracción de características con activaciones ReLU y capas de max pooling, seguidas de un bloque clasificador denso con regularización estricta por Dropout ($p=0.5$). Simula el proceso de convergencia del descenso de gradiente estocástico (SGD con momentum) a lo largo de 20 épocas y genera una visualización gráfica dual (`01_alexnet_cifar10.png`) con las curvas de aprendizaje y la matriz de confusión multiclase.
- **Entrada requerida:** Imágenes rasterizadas a color. El script descarga de manera automatizada el dataset CIFAR-10 mediante los submódulos de `torchvision.datasets`, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también el módulo nativo `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/AlexNet.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset CIFAR-10 desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni pesos preentrenados externos.
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_alexnet_cifar10.png`) por un directorio local accesible en tu entorno (por ejemplo, `./01_alexnet_cifar10.png`) antes de la ejecución para garantizar que el archivo de imagen se guarde satisfactoriamente.

### VGG16 (Transfer Learning Genérico y Clasificación con Modelos Preentrenados)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Transfer Learning y Redes Profundas)
- **Archivo:** `vision_computacional/VGG16.py`
- **Función básica:** Implementa una estrategia de transferencia de conocimiento (*transfer learning*) y ajuste fino (*fine-tuning*) utilizando la arquitectura clásica VGG16 con pesos preentrenados en ImageNet (`torchvision.models.vgg16`). El pipeline redimensiona y normaliza el dataset CIFAR-10 a 224x224 píxeles, congela el extractor de características convolucionales (`requires_grad = False`) para evitar la recomputación de pesos universales (bordes, texturas) y reemplaza la capa clasificadora final (`classifier[6]`) para adaptarla a 10 clases específicas. Produce consola informativa detallada de entrenamiento acelerado y genera una visualización gráfica dual (`01_vgg16_transfer_learning.png`) comparando la convergencia frente a entrenar desde cero junto con el diagrama de congelamiento de gradientes.
- **Entrada requerida:** Imágenes rasterizadas a color. El script descarga automáticamente el dataset CIFAR-10 y los pesos de ImageNet mediante los submódulos de `torchvision`, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy matplotlib seaborn`
- **Ejecución:** `python vision_computacional/VGG16.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset CIFAR-10 y los pesos preentrenados de VGG16 desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_vgg16_transfer_learning.png`) por un directorio local accesible en tu entorno (por ejemplo, `./01_vgg16_transfer_learning.png`) antes de la ejecución para garantizar que el archivo de imagen se guarde satisfactoriamente.

### GoogLeNet / Inception v1 (Sistemas Embebidos y Eficiencia Computacional)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Edge AI y Redes Eficientes)
- **Archivo:** `vision_computacional/GoogLeNet.py`
- **Función básica:** Implementa la arquitectura GoogLeNet (Inception v1) utilizando PyTorch (`torchvision.models.googlenet`) para demostrar optimización en sistemas embebidos y Edge AI. El pipeline carga un subconjunto del dataset real LFW (*Labeled Faces in the Wild*), adapta la última capa totalmente conectada (`fc`) al número de clases para un caso de uso de reconocimiento facial local (Smart Lock), audita comparativamente el consumo masivo de memoria frente a VGG16 y simula ciclos de inferencia en tiempo real midiendo latencia y fotogramas por segundo (FPS). Produce consola informativa detallada y genera un gráfico comparativo dual (`01_googlenet_embedded.png`) que contrasta el conteo de parámetros y la eficiencia arquitectónica.
- **Entrada requerida:** Imágenes rasterizadas a color. El script descarga de forma automatizada los datos de rostros desde scikit-learn y los pesos preentrenados en ImageNet desde torchvision, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/GoogLeNet.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset LFW desde scikit-learn y los pesos preentrenados de GoogLeNet desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_googlenet_embedded.png`) por un directorio local válido en tu entorno (por ejemplo, `./01_googlenet_embedded.png`) antes de la ejecución para asegurar que el archivo de imagen se guarde satisfactoriamente.

### ResNet (Residual Networks - Detección Avanzada de Objetos y Faster R-CNN)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección de Objetos y Backbones Resilientes)
- **Archivo:** `vision_computacional/ResNet.py`
- **Función básica:** Implementa un modelo avanzado de detección de objetos en imágenes complejas utilizando **Faster R-CNN con un backbone ResNet-50 y Feature Pyramid Networks (FPN)** (`torchvision.models.detection.fasterrcnn_resnet50_fpn`). El flujo carga una fotografía natural de alta resolución de scikit-learn (`china.jpg`), procesa la inferencia en PyTorch para extraer cajas delimitadoras (*bounding boxes*), etiquetas de clase del dataset COCO y puntuaciones de confianza, y filtra las detecciones mediante un umbral riguroso. Produce consola informativa detallada y genera una visualización gráfica dual (`01_resnet_deteccion_avanzada.png`) que muestra la detección espacial de objetos junto con la curva teórica del flujo de gradiente residual.
- **Entrada requerida:** Imágenes rasterizadas a color. El script carga de forma automatizada las imágenes de muestra desde scikit-learn y descarga los pesos preentrenados del detector desde torchvision, por lo que no requiere archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/ResNet.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar los pesos preentrenados de Faster R-CNN con ResNet-50 desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`/home/usuario/01_resnet_deteccion_avanzada.png`) por un directorio local válido en tu entorno (por ejemplo, `./01_resnet_deteccion_avanzada.png`) antes de la ejecución para asegurar que el archivo de imagen se guarde satisfactoriamente.

### DenseNet (Densely Connected Convolutional Networks - Diagnóstico de Rayos X e Imágenes Médicas)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Imágenes Médicas y Conectividad Densa)
- **Archivo:** `vision_computacional/Dense.py`
- **Función básica:** Implementa una red neuronal convolucional basada en DenseNet-121 (`torchvision.models.densenet121`) adaptada para la detección y diagnóstico automatizado de patologías médicas en radiografías. El pipeline transforma el dataset Olivetti Faces de scikit-learn para simular radiografías craneales e inyecta anomalías sintéticas (masas tumorales radiopacas con distribución gaussiana), realiza *transfer learning* congelando los bloques convolucionales base y redefiniendo el clasificador denso (`nn.Sequential`), y ejecuta un entrenamiento real con validación cruzada. Produce consola informativa detallada y genera una visualización gráfica integral en cuadrícula (`01_densenet_rayosx.png`) que incluye las curvas de convergencia, el esquema teórico de un Bloque Denso y las inferencias clínicas en pacientes sanos y con patología.
- **Entrada requerida:** Imágenes médicas rasterizadas en escala de grises. El script descarga de manera automatizada las imágenes base desde scikit-learn y los pesos preentrenados de DenseNet-121 desde torchvision, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/Dense.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset Olivetti Faces desde scikit-learn y los pesos preentrenados de DenseNet-121 desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** En el código original provisto, la instrucción de exportación (`plt.savefig('01_densenet_rayosx.png', ...)`) y visualización (`plt.show()`) se encuentran activas. **Se recomienda ajustar la ruta de guardado** si prefieres almacenar la figura en un directorio específico en lugar de la carpeta de trabajo actual.

### MobileNet (Depthwise Separable Convolutions - Visión Artificial en Smartphones y Edge Devices)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Redes Ligeras y Edge AI Móvil)
- **Archivo:** `vision_computacional/MobileNet.py`
- **Función básica:** Implementa los fundamentos conceptuales y matemáticos de la arquitectura MobileNet para la ejecución eficiente de modelos de visión artificial en dispositivos móviles y sistemas embebidos. El flujo procesa el dataset Digits de scikit-learn, transformándolo sintéticamente en un flujo de imágenes RGB de 32x32 píxeles con simulación de ruido de sensor móvil para OCR. Calcula comparativamente el ahorro drástico de operaciones de punto flotante (FLOPs) mediante Convoluciones Separables en Profundidad (*Depthwise Separable Convolutions*), simula un benchmark de latencia en milisegundos y genera una visualización gráfica integral (`02_mobilenet_vision_mvil.png`) que detalla el diagrama arquitectónico y muestras de inferencia móvil.
- **Entrada requerida:** Imágenes rasterizadas a color optimizadas para baja resolución. El script genera y adapta de manera automatizada los datos a partir del dataset incorporado en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install numpy matplotlib seaborn opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/MobileNet.py`
- **Recursos externos:** No aplica (los datos provienen de scikit-learn y las transformaciones se simulan mediante OpenCV; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
  - **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta absoluta rígida** (`02_mobilenet_vision_mvil.png`) por un directorio local accesible en tu entorno antes de la ejecución para asegurar que el archivo de imagen se guarde satisfactoriamente.

### R-CNN (Region-based Convolutional Neural Networks - Investigación Académica)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección de Objetos y Propuesta de Regiones)
- **Archivo:** `vision_computacional/R-CNN.py`
- **Función básica:** Implementa la arquitectura histórica de detección de objetos **R-CNN (Girshick et al., 2014)** adaptada a la investigación médica para la localización de lesiones sintéticas en radiografías craneales. El pipeline procesa el dataset Olivetti Faces de scikit-learn inyectando nódulos hiperdensos sintéticos con *Ground Truth* conocido, utiliza **Búsqueda Selectiva (*Selective Search*)** mediante OpenCV para extraer candidatas a regiones de interés (*region proposals*), calcula la métrica **IoU (Intersection over Union)** para etiquetar las regiones positivas, extrae vectores de características a partir de parches normalizados y entrena un clasificador basado en Regresión Logística, aplicando finalmente *Non-Max Suppression (NMS)* para eliminar detecciones redundantes y visualizar las cajas delimitadoras (*bounding boxes*).
- **Entrada requerida:** Imágenes rasterizadas en escala de grises. El script genera y adapta de manera automatizada las muestras médicas sintéticas a partir de los datos incorporados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install numpy matplotlib seaborn opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/R-CNN.py`
- **Recursos externos:** No aplica (los datos provienen de scikit-learn y las propuestas de región se calculan internamente mediante segmentación con OpenCV; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`), mientras que la ventana de visualización interactiva (`plt.show()`) está activa. **Se recomienda ajustar la ruta de guardado** si prefieres almacenar la figura con la lesión detectada en lugar de mostrarla directamente en pantalla.

### Fast R-CNN (Region-Based Convolutional Networks for Object Detection)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección y Conteo de Objetos)
- **Archivo:** `vision_computacional/Fast R-CNN.py`
- **Función básica:** Implementa la arquitectura **Fast R-CNN (Ross Girshick, 2015)**, la cual mejora drásticamente los cuellos de botella computacionales de la versión original (R-CNN). En lugar de pasar cada propuesta de región de manera independiente por la red convolucional, Fast R-CNN procesa **toda la imagen de entrada una sola vez** a través del backbone (ResNet-50) para generar un mapa de características global compartido. A partir de este mapa, la capa de **RoI Pooling** extrae vectores de características de longitud fija para cada propuesta generada externamente (vía Selective Search), alimentando una cabeza multitarea que realiza simultáneamente la clasificación de objetos (Softmax) y la regresión de cajas delimitadoras (*bounding box regression*).
- **Entrada requerida:** Imágenes rasterizadas a color (RGB). El script utiliza propuestas de regiones basadas en OpenCV (`opencv-contrib-python`) y requiere dependencias de PyTorch y torchvision.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `matplotlib`, `pillow`, `opencv-contrib-python`. (Incluye los módulos nativos `pathlib` y `time`).
- **Instalación específica:** `pip install numpy matplotlib pillow torch torchvision opencv-contrib-python`
- **Ejecución:** `python vision_computacional/Fast R-CNN.py`
- **Recursos externos:** No requiere descargas de bases de datos remotas en línea ni claves API externas; sin embargo, las cabezas de predicción requieren entrenamiento previo con un dataset de detección (como Pascal VOC o COCO) para realizar inferencias útiles, ya que los pesos preentrenados por defecto en torchvision corresponden a la variante más avanzada *Faster R-CNN*.
- **Observaciones:** El script presentado actúa como un bloque de construcción didáctico y estructural para comprender el flujo de mapeo de coordenadas y la operación de RoI Pooling sobre mapas de características profundos.

### Faster R-CNN (Region-Based Convolutional Neural Networks with RPN)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección de Objetos en Dos Etapas y Conteo Automático)
- **Archivo:** `vision_computacional/Faster R-CNN.py`
- **Función básica:** Implementa un sistema completo de **Inventario Visual y Conteo Automatizado de Stock** utilizando la arquitectura avanzada **Faster R-CNN** con un backbone ResNet-50 y Feature Pyramid Networks (FPN) a través de la API moderna de PyTorch y torchvision (`fasterrcnn_resnet50_fpn_v2`). El script genera sintéticamente estanterías corporativas inyectando productos recortados del dataset LFW People, configura el cabezal del predictor de cajas (*FastRCNNPredictor*) para múltiples clases personalizadas (Fondo, Producto Tipo A y Producto Tipo B), entrena la red optimizando conjuntamente las pérdidas de la Red de Propuesta de Regiones (RPN) y del cabezal de clasificación/regresión, y realiza la evaluación espacial para calcular automáticamente el stock disponible.
- **Entrada requerida:** Imágenes rasterizadas de estanterías y productos a color (RGB). El script genera de forma algorítmica las muestras de inventario a partir de los datos integrados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/Faster R-CNN.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar los pesos preentrenados del modelo desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script incluye la instrucción activa para exportar el gráfico resumen de convergencia y las cajas delimitadoras de conteo de inventario bajo el archivo `02_fast_rcnn_inventario.png`. Asegúrate de contar con un entorno con soporte de hardware acelerado (GPU CUDA) si deseas acelerar el tiempo de entrenamiento de las épocas convolucionales.

### Cascade R-CNN (Cai & Vasconcelos, 2018 - Detección de Defectos e Inspección Industrial)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección de Objetos en Múltiples Etapas y Refinamiento Geométrico)
- **Archivo:** `vision_computacional/Cascade R-CNN.py`
- **Función básica:** Implementa la arquitectura avanzada **Cascade R-CNN** para la localización precisa y clasificación de microdefectos industriales (grietas y porosidades) en superficies metálicas sintéticas. El pipeline genera un dataset proxy basado en texturas procesadas de scikit-learn, extrae propuestas de región iniciales mediante Búsqueda Selectiva (*Selective Search*) y entrena una secuencia de **tres cabezas de detección encadenadas con umbrales de IoU crecientes (0.50 -> 0.65 -> 0.75)**. Cada etapa filtra progresivamente los falsos positivos y realiza una regresión progresiva de las cajas delimitadoras (*bounding boxes*), elevando de manera sostenida la calidad geométrica de la predicción final.
- **Entrada requerida:** Imágenes industriales rasterizadas en escala de grises. El script construye y adapta de manera automatizada las piezas y defectos sintéticos a partir de los datos integrados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install numpy matplotlib seaborn opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/Cascade R-CNN.py`
- **Recursos externos:** No aplica (los datos y texturas se generan internamente mediante scikit-learn y OpenCV; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** El script ejecuta la visualización secuencial interactiva (`plt.show()`) mostrando la evolución del refinamiento geométrico a través de las 3 etapas de la cascada comparadas frente al *Ground Truth*.


### YOLO (You Only Look Once - Conteo Vehicular Automático y Tráfico Inteligente)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección de Objetos en Tiempo Real y Smart Cities)
- **Archivo:** `vision_computacional/YOLO.py`
- **Función básica:** Implementa un sistema completo de **Conteo Vehicular Automático y Monitoreo de Tráfico** utilizando la librería oficial de **Ultralytics YOLO** (`yolo11n.pt`). El pipeline genera un dataset sintético estructurado bajo el estándar oficial de YOLO (`dataset_trafico_yolo`) sobre pistas de asfalto inyectando recortes del dataset LFW People de scikit-learn, configura el archivo de metadatos `dataset_trafico.yaml` con tres clases personalizadas (*Auto_Sedán, Camión_Carga, Motocicleta*), ejecuta un proceso de entrenamiento por transferencia (*Fine-Tuning*) sobre las imágenes de entrenamiento, y realiza la inferencia para realizar un conteo automatizado de vehículos en tiempo real con exportación de cajas delimitadoras.
- **Entrada requerida:** Imágenes rasterizadas de tráfico a color (RGB). El script genera de manera automatizada las tomas viales sintéticas a partir de los datos incorporados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `ultralytics`, `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `os`, `shutil`, `time` y `warnings`).
- **Instalación específica:** `pip install torch ultralytics numpy matplotlib seaborn opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/YOLO.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el archivo de pesos livianos preentrenados (`yolo11n.pt`) desde los servidores oficiales de Ultralytics. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script incluye la instrucción activa para exportar el gráfico resumen de conteo vehicular y visualización de cajas bajo el archivo `03_yolo_conteo_vehicular.png`, culminando con una ventana interactiva `plt.show()`. Se recomienda disponer de un entorno compatible con aceleración por CPU/GPU para optimizar el ciclo de entrenamiento de las 5 épocas iniciales.

### SSD (Single Shot MultiBox Detector - Visión en Drones y Monitoreo Aéreo)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Detección en Una Sola Etapa y Edge AI para UAVs)
- **Archivo:** `vision_computacional/SSD.py`
- **Función básica:** Implementa un sistema de monitoreo aéreo para drones utilizando la arquitectura **SSD (Single Shot MultiBox Detector)** con un backbone VGG16 y pesos preentrenados en COCO (`torchvision.models.detection.ssd300_vgg16`). El flujo carga una imagen natural de alta resolución de scikit-learn (`china.jpg`) para emular una perspectiva cenital de vuelo, ejecuta la inferencia en un solo paso directo (*single pass*) para clasificar y localizar simultáneamente objetos a múltiples escalas, y filtra los resultados con un umbral de confianza optimizado para entornos aéreos. Produce consola informativa detallada y genera una visualización gráfica dual (`01_ssd_vision_drones.png`) con las cajas delimitadoras proyectadas sobre la escena y el gráfico estadístico de conteo.
- **Entrada requerida:** Imágenes rasterizadas en color (RGB) de alta resolución. El script carga de manera automatizada las muestras desde scikit-learn y los pesos preentrenados desde torchvision, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python vision_computacional/SSD.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar los pesos preentrenados de SSD300 desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) utiliza una ruta absoluta rígida (`/home/usuario/01_ssd_vision_drones.png`). **Es obligatorio modificar dicha ruta** por un directorio local accesible en tu entorno antes de la ejecución para asegurar que la imagen se guarde satisfactoriamente.

### U-Net (Ronneberger et al., 2015 - Segmentación Semántica Médica)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Segmentación Píxel a Píxel e Imágenes Médicas)
- **Archivo:** `vision_computacional/U-Net.py`
- **Función básica:** Implementa la arquitectura simétrica en forma de 'U' de **U-Net** en PyTorch para la segmentación semántica de tumores y lesiones cerebrales en imágenes de resonancia magnética (RM). El pipeline genera un dataset proxy a partir de las imágenes de Olivetti Faces de scikit-learn inyectando masas tumorales elípticas sintéticas y sus respectivas máscaras binarias de *Ground Truth*, entrena el modelo utilizando la función de pérdida **Dice Loss** (*Sørensen-Dice Coefficient*) para resolver de manera óptima el severo desbalance de clases, y ejecuta la inferencia generando mapas de probabilidad binarizados para comparar visualmente la superposición de las lesiones.
- **Entrada requerida:** Imágenes médicas rasterizadas en escala de grises. El script construye y adapta de manera automatizada las resonancias y máscaras a partir de los datos incorporados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `numpy`, `matplotlib`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch numpy matplotlib opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/U-Net.py`
- **Recursos externos:** No aplica (los datos y las máscaras tumorales se generan internamente mediante scikit-learn y OpenCV; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** El script ejecuta una visualización en cuadrícula interactiva mediante `plt.show()` para comparar la resonancia original, la máscara real (*Ground Truth*) y la segmentación predicha por la red U-Net.

### FPN (Feature Pyramid Networks - Mejora de Redes Base y Mapas de Características Multiescala)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Pirámides de Características y Arquitecturas de Extracción)
- **Archivo:** `vision_computacional/FPN.py`
- **Función básica:** Implementa una **Feature Pyramid Network (FPN)** utilizando PyTorch y torchvision (`torchvision.ops.FeaturePyramidNetwork`). FPN resuelve el problema fundamental de la detección multiescala combinando una ruta ascendente (*bottom-up*, basada en un backbone ResNet-18 para extraer características semánticas profundas) con una ruta descendente y conexiones laterales (*top-down*), unificando los mapas de características de diferentes capas (`layer1` a `layer4`) a un número fijo de canales (`out_channels=256`). El script carga el dataset real **STL-10**, ejecuta el paso de avance (*forward pass*) e inspecciona las dimensiones resultantes de los tensores generados a múltiples resoluciones.
- **Entrada requerida:** Imágenes rasterizadas a color (RGB). El script descarga de manera automatizada el dataset STL-10 y los pesos preentrenados de ResNet-18 desde los servidores oficiales, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`. (Incluye los submódulos `torch.nn`, `DataLoader`, y herramientas de transformaciones `torchvision.transforms`).
- **Instalación específica:** `pip install torch torchvision numpy`
- **Ejecución:** `python vision_computacional/FPN.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset STL-10 y los pesos preentrenados de ResNet-18. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script actúa como una demostración limpia y autocontenida de la arquitectura FPN, mostrando consola informativa detallada sobre la homogenización de canales y la adaptación espacial de los mapas de características.

### PSPNet (Pyramid Scene Parsing Network - Análisis de Escenas y Segmentación Semántica)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Análisis de Escenas y Contexto Global)
- **Archivo:** `vision_computacional/PSPNet.py`
- **Función básica:** Implementa los fundamentos conceptuales y arquitectónicos de **PSPNet (Zhao et al., CVPR 2017)** para la segmentación semántica píxel a píxel en imágenes de alta resolución. El pipeline genera un dataset proxy con escenas sintéticas complejas que integran múltiples regiones semánticas (*Cielo/Fondo, Estructura, Vegetación y Objeto Principal*), simula el funcionamiento del **Pyramid Pooling Module (PPM)** extrayendo características multiescala en subregiones de 1x1, 2x2, 3x3 y 6x6, evalúa el rendimiento mediante la métrica principal **mIoU (Mean Intersection over Union)**, y genera una visualización gráfica integral (`05_pspnet_analisis_escena.png`) detallando el diagrama del PPM y las máscaras semánticas predichas.
- **Entrada requerida:** Imágenes rasterizadas en alta resolución a color (RGB). El script genera y adapta de manera automatizada las escenas urbanas sintéticas a partir de los datos integrados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `matplotlib`, `seaborn`, `cv2` (opencv-python), `scikit-learn`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install numpy matplotlib seaborn opencv-python scikit-learn`
- **Ejecución:** `python vision_computacional/PSPNet.py`
- **Recursos externos:** No aplica (los datos y las máscaras provienen de scikit-learn y OpenCV; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta** por un directorio local accesible en tu entorno antes de la ejecución para garantizar que el archivo de imagen se guarde satisfactoriamente.

### Mask R-CNN (He et al., 2017 - Segmentación de Instancias y Conteo Celular Automático)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Segmentación de Instancias, Detección y Conteo Celular)
- **Archivo:** `vision_computacional/Mask R-CNN.py`
- **Función básica:** Implementa la arquitectura avanzada **Mask R-CNN** en PyTorch (`torchvision.models.detection.maskrcnn_resnet50_fpn`) para la detección, localización, segmentación precisa a nivel de píxel y conteo individual de células u núcleos en imágenes microscópicas. El pipeline genera un dataset proxy a partir de las imágenes de Olivetti Faces de scikit-learn inyectando micrografías de tejido de fondo y múltiples núcleos elípticos con sus respectivas máscaras de instancia y bounding boxes (*Ground Truth*), reemplaza las cabezas de predicción para adaptarlas a la clasificación binaria (*Fondo vs. Célula*), y ejecuta un entrenamiento multitarea basado en pérdidas combinadas (Caja + Máscara + RPN) para separar y contar células adyacentes o solapadas.
- **Entrada requerida:** Imágenes microscópicas rasterizadas a color (RGB). El script construye y adapta de manera automatizada las micrografías y máscaras celulares a partir de los datos integrados en scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `cv2` (opencv-python), `scikit-learn`, `matplotlib`. (Incluye también los módulos nativos `time` y `warnings`).
- **Instalación específica:** `pip install torch torchvision numpy opencv-python scikit-learn matplotlib`
- **Ejecución:** `python vision_computacional/Mask R-CNN.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar los pesos preentrenados del modelo desde los servidores oficiales de torchvision. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script ejecuta una visualización comparativa en cuadrícula interactiva mediante `plt.show()` para contrastar la micrografía original de entrada, el *Ground Truth* con bounding boxes y la segmentación de instancias predicha con colores independientes por célula.


### YOLACT (You Only Look At Coeffs - Realidad Aumentada Interactiva y Segmentación en Tiempo Real)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Segmentación de Instancias en Tiempo Real y AR)
- **Archivo:** `vision_computacional/YOLACT.py`
- **Función básica:** Implementa un pipeline completo de **Realidad Aumentada Interactiva** en tiempo real empleando un modelo de segmentación de instancias (*YOLOv8 nano segmentation* via Ultralytics). El script procesa fotogramas de video en directo (desde la webcam o archivos) o un *fallback* con imágenes de prueba de CIFAR-10, predice máscaras binarias de segmentación píxel a píxel para cada objeto detectado, y aplica un efecto holográfico/neón dinámico basado en un patrón de ondas sinusoidales combinado mediante alpha blending vectorizado con NumPy, resaltando el contorno de los elementos superpuestos con anotaciones visuales.
- **Entrada requerida:** Flujo de video continuo en tiempo real (Webcam) o imágenes estáticas rasterizadas (compatible con CIFAR-10). El script descarga de manera automatizada los pesos de segmentación preentrenados (`yolov8n-seg.pt`) y el dataset CIFAR-10, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `ultralytics`, `numpy`, `cv2` (opencv-python). (Incluye submódulos para manipulación de datasets y tensores).
- **Instalación específica:** `pip install torch torchvision ultralytics numpy opencv-python`
- **Ejecución:** `python vision_computacional/YOLACT.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar los pesos de segmentación de Ultralytics y el dataset CIFAR-10 en caso de que la cámara no esté disponible. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script incluye manejo de excepciones para entornos sin cámara física, descargando automáticamente una muestra de CIFAR-10, procesándola y guardando el resultado estático en un archivo de imagen (`ar_output_sample.jpg`). En entornos de ejecución interactiva con interfaz gráfica de escritorio, abrirá una ventana de OpenCV que se cerrará pulsando la tecla 'q'.

### Vision Transformer (ViT - Clasificación a Gran Escala y Atención Multinúcleo)

- **Categoría:** Aprendizaje Profundo / Visión por Computador (Transformers y Clasificación de Imágenes)
- **Archivo:** `vision_computacional/Vision Transformer.py`
- **Función básica:** Implementa la arquitectura completa de un **Vision Transformer (ViT)** en PyTorch para la clasificación a gran escala de rostros. El script carga el dataset real **LFW People (*Labeled Faces in the Wild*)** de scikit-learn, implementa el módulo personalizado `PatchEmbedding` mediante una capa convolucional 2D para proyectar la imagen en secuencias de parches, añade tokens de clasificación especiales (`[CLS] token`) e incrustaciones posicionales aprendibles (*Positional Embeddings*), y construye los bloques del codificador de Transformers estándar con atención multinúcleo (`nn.TransformerEncoderLayer`) para clasificar la identidad de los rostros mediante un cabezal MLP.
- **Entrada requerida:** Imágenes rasterizadas de rostros en escala de grises. El dataset base se descarga y adapta de forma automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `numpy`, `matplotlib`, `scikit-learn`. (Incluye los submódulos `torch.nn`, `DataLoader`, `TensorDataset` y herramientas de preprocesamiento).
- **Instalación específica:** `pip install torch numpy matplotlib scikit-learn`
- **Ejecución:** `python vision_computacional/Vision Transformer.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset LFW People desde scikit-learn. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script actúa como una implementación didáctica y autocontenida de ViT adaptada a resoluciones de rostros medianas, imprimiendo en consola el progreso del entrenamiento y la precisión final obtenida en el conjunto de prueba.


### PaLM & Gemini API (Búsqueda Semántica, Embeddings Vectoriales y Resumen de Textos)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP, Embeddings y Grandes Modelos de Lenguaje)
- **Archivo:** `ia_generativa/PaLM.py`
- **Función básica:** Implementa un motor de **búsqueda semántica mediante vectores de embedding y resumen automático de documentos**, actualizado con los estándares modernos de la **Gemini API** (sucesor del modelo legado PaLM). El pipeline descarga una selección del dataset **20 Newsgroups** de scikit-learn (`sci.med`, `sci.space`, `comp.graphics`, `talk.politics.guns`), procesa los textos mediante la API de Google para generar representaciones vectoriales densas, calcula la **similitud coseno** entre consultas de lenguaje natural y el corpus documental, e integra rutinas de resumen automático con soporte para modo de simulación (*DEMO*) o producción con clave de API real.
- **Entrada requerida:** Documentos de texto plano en lenguaje natural. El script descarga de forma automatizada los datos desde scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `scikit-learn`, `google-generativeai`, `tqdm`. (Incluye también los módulos nativos `os`, `warnings` y tipado de Python).
- **Instalación específica:** `pip install numpy pandas scikit-learn google-generativeai tqdm`
- **Ejecución:** `python ia_generativa/PaLM.py`
- **Recursos externos:** Requiere configurar la variable de entorno `GEMINI_API_KEY` (obtenida desde *Google AI Studio*) para interactuar con la API real de embeddings y generación de texto; de lo contrario, el script ejecutará automáticamente un modo de simulación didáctica con vectores aleatorios.
- **Observaciones:** El script destaca las ventajas clave de migración desde los modelos PaLM legacy hacia la infraestructura de Gemini (soporte multicanal/multimodal con *Gemini-Embedding-2*, mayor dimensionalidad de vectores de hasta 3072 y cobertura extendida de más de 100 idiomas).

### ChatGPT / GPT-5 (Resolución de Problemas Complejos, Chain-of-Thought y Razonamiento Analítico)

- **Categoría:** Inteligencia Artificial / Modelos de Lenguaje Avanzados (Razonamiento Complejo, CoT y APIs Multimodales)
- **Archivo:** `ia_generativa/ChatGPT-GPT-5.py`
- **Función básica:** Implementa un sistema de resolución de problemas complejos y diagnóstico médico automatizado utilizando la API de OpenAI (`openai.OpenAI`) bajo el patrón de **Cadena de Pensamiento (*Chain-of-Thought - CoT*)**. El script carga el dataset real **Breast Cancer Wisconsin** de scikit-learn, extrae las características multivariable de una muestra de prueba, estructura un prompt de razonamiento analítico paso a paso para forzar al modelo a descomponer el problema y solicita obligatoriamente una respuesta estructurada en formato JSON (`response_format={"type": "json_object"}`) para evaluar la atipicidad, ponderar riesgos y emitir un diagnóstico explicable.
- **Entrada requerida:** Datos tabulares multivariables numéricos. El dataset base se descarga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `openai`, `numpy`, `pandas`, `scikit-learn`. (Incluye también los módulos nativos `os` y `json`).
- **Instalación específica:** `pip install openai numpy pandas scikit-learn`
- **Ejecución:** `python ia_generativa/ChatGPT-GPT-5.py`
- **Recursos externos:** Requiere configurar la variable de entorno `OPENAI_API_KEY` con una clave activa de OpenAI para realizar consultas en vivo a los modelos (como `gpt-4o` o GPT-5); de lo contrario, capturará la excepción por falta de credenciales mostrando la estructura del código demostrativo.
- **Observaciones:** El script sirve como un patrón arquitectónico avanzado para integrar grandes modelos de lenguaje en flujos analíticos donde la explicabilidad y el razonamiento estructurado paso a paso son críticos.

### Claude (Anthropic Claude API - Análisis Inteligente de Contratos Legales y Tool Use)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (Modelos de Ventana Larga, Tool Calling y LegalTech)
- **Archivo:** `ia_generativa/Claude.py`
- **Función básica:** Implementa un sistema automatizado de **análisis de contratos legales, extracción de cláusulas críticas y evaluación de riesgos regulatorios** utilizando la API de Anthropic (`anthropic.Anthropic`). El pipeline procesa muestras de acuerdos comerciales (NDAs y SLAs), aprovecha las capacidades de *Long-Context Window* de Claude para analizar documentos extensos sin pérdida de coherencia, y aplica el mecanismo de **Tool Calling (Structured Outputs con JSON Schema)** para garantizar que la respuesta devuelva un objeto tabulado estricto con el tipo de contrato, jurisdicción, nivel de riesgo (*Risk Score*), cláusulas asimétricas y un resumen ejecutivo estructurado para integración en sistemas ERP o LegalTech.
- **Entrada requerida:** Contratos y acuerdos legales en texto plano. El script define un conjunto de muestras representativas integradas directamente en el código, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `anthropic`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`. (Incluye también los módulos nativos `json`, `time` y `warnings`).
- **Instalación específica:** `pip install anthropic numpy pandas seaborn matplotlib scikit-learn`
- **Ejecución:** `python ia_generativa/Claude.py`
- **Recursos externos:** Requiere configurar la variable de entorno `ANTHROPIC_API_KEY` con una clave activa de Anthropic para realizar consultas en vivo a la API de Claude; de lo contrario, el script incluye un modo de simulación (*mocking*) completamente funcional para demostrar la estructura de salida.
- **Observaciones:** El script genera un informe tabular consolidado en consola y despliega una visualización gráfica dual resumen con la distribución de niveles de riesgo y la cantidad de cláusulas desbalanceadas detectadas por el modelo.

### Gemini 3 Pro (Asistencia Técnica Avanzada y Function Calling con el SDK Oficial de Google GenAI)

- **Categoría:** Inteligencia Artificial / Modelos Multimodales Nativos (Agentes de Soporte Técnico y Function Calling)
- **Archivo:** `ia_generativa/Gemini pro.py`
- **Función básica:** Implementa un agente experto de asistencia técnica de nivel 3 utilizando el SDK moderno de Google (`google-genai`) con el modelo `gemini-3-pro-preview`. El script extrae un ticket de soporte real de las subcategorías técnicas del dataset **20 Newsgroups** de scikit-learn (`comp.sys.hardware`, `comp.graphics`, `comp.os.ms-windows.misc`), configura instrucciones de sistema especializadas y define herramientas personalizadas mediante **Function Calling** (`buscar_solucion_en_base_de_datos`) que el modelo puede invocar de forma autónoma para consultar bases de conocimiento técnicas y emitir diagnósticos estructurados paso a paso.
- **Entrada requerida:** Tickets de soporte en texto plano. El script descarga de manera automatizada las muestras técnicas desde scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `google-genai`, `scikit-learn`, `numpy`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install google-genai scikit-learn numpy`
- **Ejecución:** `python ia_generativa/Gemini pro.py`
- **Recursos externos:** Requiere configurar la variable de entorno `GEMINI_API_KEY` (obtenida desde *Google AI Studio*) para interactuar con la API real de Gemini; de lo contrario, requerirá credenciales válidas al ejecutar las llamadas al cliente.
- **Observaciones:** El script representa la forma estándar de integrar capacidades de ejecución de herramientas (*Function Calling*) y razonamiento de alta precisión con la infraestructura nativa multilingüe y multimodal de los modelos Gemini.

### Gemini.py / Gemini 3 Fast (Agentes Conversacionales Fluidos y Memoria Multiturno)

- **Categoría:** Inteligencia Artificial / Modelos Multimodales Rápidos (Agentes Conversacionales, Memoria de Sesión y Tool Use)
- **Archivo:** `ia_generativa/Gemini.py`
- **Función básica:** Implementa un **agente conversacional multiturno altamente fluido** utilizando el SDK moderno de Google (`google-genai`) y la sesión de chat con gestión automática de memoria (`client.chats.create` con el modelo `gemini-2.5-flash`). El script carga el dataset real **California Housing** de scikit-learn y procesa los precios, define una herramienta analítica personalizada (`consultar_estadisticas_vivienda`) para filtrar y calcular métricas inmobiliarias en tiempo real, y simula un diálogo multiturno coherente donde el agente recuerda el contexto previo mientras decide de forma autónoma cuándo invocar funciones.
- **Entrada requerida:** Interacciones de usuario en lenguaje natural en formato de texto plano (multiturno). El dataset base se descarga de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `google-genai`, `pandas`, `numpy`, `scikit-learn`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install google-genai pandas numpy scikit-learn`
- **Ejecución:** `python ia_generativa/Gemini.py`
- **Recursos externos:** Requiere configurar la variable de entorno `GEMINI_API_KEY` (obtenida desde *Google AI Studio*) para interactuar con la API real de chat y generación de contenido de Gemini; de lo contrario, capturará la excepción por falta de credenciales al iniciar los turnos de conversación.
- **Observaciones:** Este script demuestra el estándar óptimo para construir interfaces conversacionales con estado, baja latencia y ejecución integrada de herramientas de datos en arquitecturas impulsadas por Google GenAI.

### Perplexity AI (Sonar Search API - Investigación Académica y Búsqueda Confiable en Tiempo Real)

- **Categoría:** Inteligencia Artificial / Motores de Búsqueda Conversacionales y Grounded Research (Investigación Científica)
- **Archivo:** `ia_generativa/Perplexity AI.py`
- **Función básica:** Implementa un sistema automatizado de **mapeo de literatura científica, extracción de citas verificadas en tiempo real y análisis de consenso académico** utilizando los modelos de la familia **Sonar de Perplexity AI** (compatibles con el cliente OpenAI mediante *base_url="https://api.perplexity.ai"*). El script extrae palabras clave y tópicos científicos del dataset **20 Newsgroups** de scikit-learn para formular preguntas de investigación automatizadas sobre inteligencia artificial y visión por computador, ejecutando consultas orientadas a repositorios académicos con citación estricta y control de alucinaciones (*Web Grounding*).
- **Entrada requerida:** Preguntas de investigación y tópicos científicos en texto plano. El script extrae y procesa los temas de forma automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `openai`, `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`. (Incluye también los módulos nativos `json`, `time` y `warnings`).
- **Instalación específica:** `pip install openai numpy pandas seaborn matplotlib scikit-learn`
- **Ejecución:** `python ia_generativa/Perplexity AI.py`
- **Recursos externos:** Requiere configurar la variable de entorno `PERPLEXITY_API_KEY` con una clave activa de Perplexity para realizar llamadas directas a la API de Sonar; de lo contrario, incluye un simulador integrado (*mocking*) que emite respuestas de investigación estructuradas con enlaces de citación reales.
- **Observaciones:** El script genera un informe tabular consolidado en consola con las fuentes bibliográficas de soporte y despliega una visualización gráfica dual resumen comparando la cantidad de referencias científicas citadas y la longitud de las síntesis generadas.
### Grok 4.1 (xAI SDK - Análisis de Tendencias Financieras y Mercados en Tiempo Real)

- **Categoría:** Inteligencia Artificial / Modelos de Razonamiento y Análisis de Mercados (xAI SDK)
- **Archivo:** `ia_generativa/Grok.py`
- **Función básica:** Implementa un sistema automatizado de **análisis de tendencias bursátiles y evaluación de riesgo financiero** utilizando el SDK oficial de xAI (`xai_sdk`) con el modelo de razonamiento rápido `grok-4.1-fast-reasoning`. El script descarga datos históricos reales del índice S&P 500 (`sp500_values`) mediante `fetch_openml` de scikit-learn, calcula indicadores técnicos de tendencia (medias móviles simples *SMA 20* y *SMA 50*, retornos diarios y volatilidad anualizada), formatea un resumen cuantitativo estructurado y lo transmite al modelo de Grok para generar un reporte de análisis financiero experto.
- **Entrada requerida:** Datos tabulares financieros de series temporales. El script descarga de manera automatizada las series bursátiles desde OpenML, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `xai-sdk`, `pandas`, `numpy`, `scikit-learn`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install xai-sdk pandas numpy scikit-learn`
- **Ejecución:** `python ia_generativa/Grok.py`
- **Recursos externos:** Requiere configurar la variable de entorno `XAI_API_KEY` con una clave activa de xAI para interactuar con la plataforma de Grok; de lo contrario, capturará la excepción por falta de credenciales al intentar emitir la respuesta en tiempo real.
- **Observaciones:** El script demuestra cómo integrar capacidades avanzadas de modelado cuantitativo en Python con los modelos analíticos de xAI, automatizando la transformación de series de tiempo bursátiles en reportes ejecutivos estructurados.

### Midjourney (Creación de Arte Conceptual y Generación de Prompts Parametrizados)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Texto a Imagen (Arte Conceptual y Prompt Engineering)
- **Archivo:** `ia_generativa/Midjourney.py`
- **Función básica:** Implementa un sistema automatizado de **extracción de componentes estéticas latentes (*Eigenfaces*) mediante Análisis de Componentes Principales (PCA)** sobre el dataset real de rostros **LFW People (*Labeled Faces in the Wild*)** de scikit-learn, combinándolas dinámicamente para construir **prompts parametrizados y avanzados para Midjourney v6**. El script correlaciona la varianza explicada de las características visuales con parámetros avanzados de renderizado (`--v 6.0`, `--ar 16:9`, `--stylize`, `--chaos`), estructurando conceptos artísticos originales y exportando un mapa visual de estilos latentes (`midjourney_concept_art_style_map.png`).
- **Entrada requerida:** Imágenes rasterizadas de rostros en escala de grises. El dataset base se descarga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también la herramienta de descomposición `sklearn.decomposition.PCA`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python ia_generativa/Midjourney.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset LFW People desde scikit-learn. No necesita claves API, variables de entorno ni credenciales adicionales.
- **Observaciones:** El script genera de manera autónoma múltiples conceptos de arte estético (*Cyberpunk Neon, Dark Fantasy Chiaroscuro, Surrealist Impressionism*, etc.) y guarda localmente la visualización gráfica de los componentes latentes.

### DALL-E 3 (Automatización de Ilustraciones de Marketing y Generación de Recursos Gráficos)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Texto a Imagen (Marketing Digital y Publicidad Comercial)
- **Archivo:** `ia_generativa/DALL-E 3.py`
- **Función básica:** Implementa un sistema automatizado de **generación de recursos gráficos y comerciales para campañas de marketing digital** utilizando la API de OpenAI DALL-E 3 (`openai.OpenAI`). El script procesa un catálogo de productos estructurado, aplica técnicas de aprendizaje automático con scikit-learn (vectorización TF-IDF y clustering mediante `KMeans`) para agrupar los artículos por perfiles de audiencia, y construye dinámicamente prompts detallados orientados a anuncios comerciales de alta calidad con optimización de estilo, iluminación profesional y composición limpia para redes sociales.
- **Entrada requerida:** Datos tabulares de catálogos de producto con descripciones textuales. El script define un conjunto de datos simulado integrado directamente en el código, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `openai`, `pandas`, `numpy`, `scikit-learn`, `requests`, `pillow`. (Incluye también los módulos nativos `os`, `json` y la gestión de flujos binarios con `io.BytesIO`).
- **Instalación específica:** `pip install openai pandas numpy scikit-learn requests pillow`
- **Ejecución:** `python ia_generativa/DALL-E 3.py`
- **Recursos externos:** Requiere configurar la variable de entorno `OPENAI_API_KEY` con una clave activa de OpenAI para realizar solicitudes reales a la API de DALL-E 3 y descargar las ilustraciones resultantes; de lo contrario, operará en modo demostración mostrando los prompts listos para producción.
- **Observaciones:** El script destaca la capacidad avanzada de DALL-E 3 de revisar y expandir automáticamente los prompts del usuario para mejorar el realismo, la fidelidad de los detalles y la coherencia visual en recursos publicitarios.

### Stable Diffusion / Flux (Generación Masiva de Texturas PBR y Modelos de Difusión)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Difusión (Generación de Texturas 3D y PBR)
- **Archivo:** `ia_generativa/Stable Diffusion- Flux.py`
- **Función básica:** Implementa un sistema de **generación masiva de texturas sin costuras (*seamless tileable PBR textures*)** optimizado para motores de renderizado 3D mediante la librería Diffusers de Hugging Face y el modelo de flujo de alta velocidad **FLUX.1-schnell**. El script extrae nombres de materiales desde catálogos de scikit-learn / OpenML (con un conjunto de respaldo de texturas estándar), construye prompts estructurados bajo una vista ortográfica cenital (*top-down flat orthographic view*) e iluminación de estudio neutra, y ejecuta inferencia por lotes en GPUs con optimizaciones de administración de VRAM (*CPU offload*) y un bajo número de pasos (*4-step generation*).
- **Entrada requerida:** Nombres de materiales y descripciones de superficies en texto plano. El script gestiona de manera automatizada la carga de datos mediante scikit-learn o listas predefinidas, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `diffusers`, `transformers`, `accelerate`, `pandas`, `numpy`, `pillow`, `scikit-learn`.
- **Instalación específica:** `pip install torch diffusers transformers accelerate pandas numpy pillow scikit-learn`
- **Ejecución:** `python ia_generativa/Stable Diffusion- Flux.py`
- **Recursos externos:** Requiere una tarjeta GPU compatible con CUDA con suficiente VRAM y conexión a internet en su primera ejecución para descargar los pesos de FLUX.1-schnell desde el repositorio de Hugging Face (`black-forest-labs/FLUX.1-schnell`).
- **Observaciones:** El script procesa automáticamente el lote completo de prompts definido, guarda las texturas resultantes en formato PNG con nombres normalizados y muestra por consola las dimensiones de cada imagen generada.

### Nano Banana Pro (Publicidad Digital, AdTech y Optimización de Ofertas RTB)

- **Categoría:** Aprendizaje Automático / AdTech (Predicción de CTR, Clasificación y Real-Time Bidding)
- **Archivo:** `ia_generativa/Nano.py`
- **Función básica:** Implementa un modelo predictivo compacto y ultrarrápido (*Nano-ADS Predictor*) basado en `HistGradientBoostingClassifier` de scikit-learn para la **predicción de clics y conversiones (*CTR/CR*), optimización de pujas automatizadas y Real-Time Bidding (RTB)**. El script genera un dataset sintético representativo de impresiones publicitarias con 10 variables clave (edad, frecuencia, hora, dispositivo, historial, etc.), entrena un clasificador optimizado para tomar decisiones en milisegundos (< 5ms), y ejecuta una simulación económica de subastas calculando el valor esperado de la puja (*pCTR * Valor*), el filtrado por umbral de rentabilidad y el Retorno de Inversión Publicitaria (**ROAS**).
- **Entrada requerida:** Datos tabulares multivariables de impresiones publicitarias. El script genera de forma automatizada los datos mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python ia_generativa/Nano.py`
- **Recursos externos:** No aplica (los datos y las métricas provienen de la simulación ejecutada por scikit-learn; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta** por un directorio local accesible en tu entorno antes de la ejecución para garantizar que el archivo de imagen resumen se guarde correctamente.

### Adobe Firefly (Adobe Firefly Services - Diseño Gráfico Profesional y Generación de Contenido por API)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Medios (Diseño Gráfico Profesional, Edición de Imágenes y APIs de Adobe Firefly Services)
- **Archivo:** `ia_generativa/Adobe Firefly Services.py`
- **Función básica:** Implementa un pipeline automatizado para la **generación, expansión y edición inteligente de recursos gráficos profesionales** utilizando las APIs oficiales de *Adobe Firefly Services* y *Adobe Photoshop APIs*. El flujo procesa catálogos de diseño, construye solicitudes para la creación de contenido comercial con control estricto de propiedad intelectual y licencias empresariales (*Commercial Safety*), y automatiza tareas de post-producción gráfica como el relleno generativo (*Generative Fill*), la expansión de imagen (*Generative Expand*) y la vectorización avanzada para flujos de trabajo de diseño gráfico a escala.
- **Entrada requerida:** Archivos de imagen rasterizados (PNG/JPEG) o descripciones textuales detalladas orientadas a diseño gráfico comercial.
- **Dependencias:** `requests`, `pandas`, `numpy`, `pillow`. (Incluye los submódulos nativos `os`, `json` y `io.BytesIO`).
- **Instalación específica:** `pip install requests pandas numpy pillow`
- **Ejecución:** `python ia_generativa/Adobe Firefly Services.py`
- **Recursos externos:** Requiere credenciales de acceso válidas a *Adobe Developer Console* (Adobe Client ID / Client Secret y tokens de autenticación OAuth 2.0) para interactuar con los endpoints productivos de Firefly Services.
- **Observaciones:** Este script completa la biblioteca de documentación técnica del repositorio para modelos generativos gráficos profesionales, integrando el estándar de seguridad comercial y cumplimiento legal característico del ecosistema Adobe.

### Leonardo & Promethean (Leonardo.ai SDK - Diseño de Personajes 2D/3D y Hojas de Modelo)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Medios (Diseño de Videojuegos, Concept Art y Character Sheets)
- **Archivo:** `ia_generativa/LeonardoAi.py`
- **Función básica:** Implementa un pipeline automatizado para la **generación de hojas de personajes en vistas múltiples (*character turnaround views*) orientadas a modeladores 3D y artistas conceptuales 2D**, utilizando el SDK oficial de *Leonardo.ai* (`leonardo_ai_sdk`). El script extrae temas y arquetipos desde colecciones textuales de scikit-learn (`20newsgroups`), construye prompts estructurados con proyecciones ortográficas y estilo de renderizado de Unreal Engine 5, y gestiona una rutina asíncrona de sondeo (*Polling*) para consultar el estado de la tarea y descargar automáticamente los renders resultantes.
- **Entrada requerida:** Roles y descripciones de personajes en texto plano. El dataset base se carga de forma automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `leonardo-ai-sdk`, `requests`, `scikit-learn`, `numpy`. (Incluye también los submódulos nativos `os` y `time`).
- **Instalación específica:** `pip install leonardo-ai-sdk requests scikit-learn numpy`
- **Ejecución:** `python ia_generativa/LeonardoAi.py`
- **Recursos externos:** Requiere configurar la variable de entorno `LEONARDO_API_KEY` con una clave activa de Leonardo.ai para interactuar con la plataforma de generación; de lo contrario, capturará la excepción por falta de credenciales al intentar emitir la solicitud a la API.
- **Observaciones:** Este script completa la biblioteca de referencias técnicas para pipelines de diseño gráfico y animación, automatizando la creación de recursos (*game assets*) listos para producción tridimensional.


### Luma Photon (VFX, Animación y Simulación de Dinámicas para Generación Cinematográfica)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y VFX (Simulación Cinematográfica y Renderizado 3D)
- **Archivo:** `ia_generativa/Luma Photon.py`
- **Función básica:** Implementa un pipeline de **extracción de trayectorias de partículas y dinámicas de fluidos (VFX)** utilizando scikit-learn (`make_blobs` y agrupamiento mediante `KMeans`) para calcular centroides de emisión tridimensionales en un sistema de 3,000 partículas simuladas, traduciendo coordenadas espaciales en directivas de cámara, iluminación volumétrica y prompts cinematográficos optimizados para el motor de renderizado **Luma Photon**.
- **Entrada requerida:** Coordenadas espaciales de partículas en 3D. El script genera y procesa los datos volumétricos de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`. (Incluye también la proyección tridimensional de Matplotlib `Axes3D`).
- **Instalación específica:** `pip install numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python ia_generativa/Luma Photon.py`
- **Recursos externos:** No aplica (los datos volumétricos y los centroides provienen de la simulación local ejecutada con scikit-learn; no requiere descarga de archivos remotos, claves API ni credenciales adicionales).
- **Observaciones:** En el código original provisto, la instrucción para exportar la visualización gráfica resumen (`plt.savefig`) se encuentra comentada (`#`). **Es obligatorio descomentarla y modificar la ruta** por un directorio local accesible en tu entorno antes de la ejecución para garantizar que el archivo de imagen resumen se guarde correctamente.

### Seedream (Seedream AI / ByteDance - Generación de Conceptos Creativos y Arte Conceptual)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Medios y Text-to-Image (ByteDance Seedream API)
- **Archivo:** `ia_generativa/Seedream.py`
- **Función básica:** Implementa un pipeline integrado de **clustering temático y extracción de conceptos visuales** utilizando scikit-learn (`TfidfVectorizer`, `TruncatedSVD` y `KMeans`) sobre el dataset de textos **20 Newsgroups** (`sci.space`, `comp.graphics`, `sci.electronics`), mapeando los grupos temáticos a direcciones de arte específicas (*Sci-Fi Cinematic*, *Minimalist Architectural*, *Retro-Futurism Graphic Art*) y conectándolas con la API REST de **Seedream AI (ByteDance)** para automatizar la generación de recursos gráficos avanzados en resoluciones de hasta 2K/4K con relaciones de aspecto personalizadas.
- **Entrada requerida:** Dataset documental de scikit-learn procesado mediante análisis semántico latente (LSA). El script descarga y agrupa los temas de forma automatizada, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `requests`, `pandas`, `matplotlib`, `scikit-learn`. (Incluye también los submódulos nativos `os` y `json`).
- **Instalación específica:** `pip install requests pandas matplotlib scikit-learn`
- **Ejecución:** `python ia_generativa/Seedream.py`
- **Recursos externos:** Requiere configurar la variable de entorno `SEEDREAM_API_KEY` (o `SEGMIND_API_KEY`) con una clave activa para interactuar con los endpoints productivos de Seedream; de lo contrario, el cliente incluye un modo de simulación que valida la correcta construcción del pipeline y los prompts generados.
- **Observaciones:** Este script corona la biblioteca de documentación técnica de modelos generativos multimodales y visuales del repositorio, integrando la potencia de los clústeres estadísticos con la síntesis estética de ByteDance.

### Canva Magic Media (Presentaciones y Redes Sociales - Automatización con Canva Connect API)

- **Categoría:** Inteligencia Artificial / Diseño Gráfico y Marketing de Contenidos (Canva Connect & Magic Media API)
- **Archivo:** `ia_generativa/Canva Magic Media.py`
- **Función básica:** Implementa un pipeline automatizado de **creación de contenidos para redes sociales y presentaciones** mediante la integración de analítica de datos y la API REST de Canva (`Canva Connect API` y su motor de generación visual *Magic Media*). El script carga un dataset real de marketing/ventas (*Bank Marketing* vía OpenML/scikit-learn), extrae los segmentos demográficos clave y formatea titulares, subtítulos y llamadas a la acción (*CTA*), estructurando prompts optimizados para rellenar automáticamente plantillas de marca (*Brand Templates* a través del endpoint de Autofill).
- **Entrada requerida:** Datos tabulares de marketing y segmentación de clientes. El script gestiona de forma automatizada la carga desde scikit-learn / OpenML, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/Canva Magic Media.py`
- **Recursos externos:** Requiere configurar la variable de entorno `CANVA_ACCESS_TOKEN` (con un token OAuth 2.0 válido de Canva Developers) y `CANVA_TEMPLATE_ID` para interactuar con los endpoints de producción; de al contrario, el script simula de forma estructurada la preparación del payload y las llamadas a la API.
- **Observaciones:** Este script añade la última pieza clave a la biblioteca técnica de documentación para herramientas de diseño asistido por IA, cerrando el ciclo entre análisis de datos y despliegue masivo en plataformas de contenido gráfico.

### Runway (Gen-3 Alpha - Producción de Cine y Generación de Video por IA)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video (Producción Cinematográfica y Text/Image-to-Video)
- **Archivo:** `ia_generativa/Runway Gen-3 Alpha.py`
- **Función básica:** Implementa un pipeline completo para la **producción cinematográfica automatizada y la estructuración de prompts para Runway Gen-3 Alpha** (`gen3a_turbo`). El script procesa el dataset real de rostros **LFW People (*Labeled Faces in the Wild*)** de scikit-learn, entrena un modelo de clasificación con *Eigenfaces (PCA)* y SVM para modelar atributos de personajes, y genera directivas de movimiento de cámara, encuadre e iluminación estructuradas bajo la sintaxis recomendada por Runway (`[Camera] + [Subject] + [Environment & Lighting]`), exportando el payload JSON listo para su integración con la API REST de producción.
- **Entrada requerida:** Imágenes rasterizadas de rostros y metadatos de casting. El dataset base se carga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `scikit-learn`. (Incluye también el submódulo nativo `json`).
- **Instalación específica:** `pip install numpy pandas scikit-learn`
- **Ejecución:** `python ia_generativa/Runway Gen-3 Alpha.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset LFW People desde scikit-learn. Para interactuar con el endpoint productivo de Runway (`https://api.dev.runwayml.com/v1/tasks`), se requiere configurar una clave API oficial de desarrollador.
- **Observaciones:** Este script añade la última integración clave para completar el bloque de modelos de video y producción de cine de la biblioteca técnica del repositorio.

### Synthesia (Videos de Capacitación y Avatares de IA - Automatización con Synthesia API v2)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y Avatares Digitales (Capacitación Corporativa y E-Learning)
- **Archivo:** `ia_generativa/Synthesia.py`
- **Función básica:** Implementa un pipeline automatizado para la **creación de videos de capacitación corporativa con avatares de IA en lenguaje natural** utilizando la API REST v2 de *Synthesia*. El script procesa colecciones documentales de soporte técnico mediante scikit-learn (`TfidfVectorizer` y agrupamiento `KMeans`), extrae palabras clave para redactar guiones pedagógicos estructurados de forma dinámica, y gestiona las solicitudes de renderizado de video mediante peticiones HTTP autenticadas con soporte para sondeo de estado (*polling*).
- **Entrada requerida:** Guiones en texto plano y metadatos de módulos educativos. El dataset base se carga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también los submódulos nativos `os`, `json` y `time`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/Synthesia.py`
- **Recursos externos:** Requiere configurar la variable de entorno `SYNTHESIA_API_KEY` con una clave activa de Synthesia para interactuar con los endpoints de producción; de lo contrario, el cliente incluye un modo de simulación que valida la correcta construcción del payload y la estructuración de los guiones.
- **Observaciones:** Este script añade la última integración clave para completar el bloque de modelos de generación de video con avatares corporativos y capacitación en la biblioteca técnica del repositorio.

### Colossyan (Marketing y RRHH - Generación de Videos y Onboarding Automatizado con Colossyan API)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y Avatares Digitales (Recursos Humanos y Marketing Corporativo)
- **Archivo:** `ia_generativa/Colossyan.py`
- **Función básica:** Implementa un pipeline automatizado para la **creación de videos personalizados de bienvenida y onboarding corporativo utilizando avatares de IA narradores** mediante la integración con la API REST de *Colossyan*. El script carga datos analíticos de RR.HH. (*HR Analytics* vía OpenML/scikit-learn o mediante estructuras de respaldo detalladas), procesa los perfiles de nuevos empleados y genera guiones de bienvenida personalizados en idioma español, estructurando el payload técnico estandarizado (*escenas, avatares, scripts y fondos institucionales*) para su envío automatizado.
- **Entrada requerida:** Datos tabulares de empleados (nombres, roles y departamentos) y textos de guiones en lenguaje natural. El script gestiona de manera automatizada la carga de datos mediante scikit-learn o estructuras relacionales locales, por lo que no requiere cargar archivos externos manuales.
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/Colossyan.py`
- **Recursos externos:** Requiere configurar la variable de entorno `COLOSSYAN_API_KEY` con una clave activa de Colossyan para interactuar con los endpoints de producción; de lo contrario, el script simula de forma estructurada la preparación del payload y las peticiones a la API.
- **Observaciones:** Este script añade una nueva pieza clave al ecosistema de documentación técnica de modelos generativos de video orientados a la automatización de procesos internos y gestión de talento en el repositorio.

### Pictory (Shorts y Reels Automáticos - Generación de Video desde Textos y Artículos)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y Edición de Contenido (Automatización de Shorts, Reels y TikToks)
- **Archivo:** `ia_generativa/Pictory AI.py`
- **Función básica:** Implementa un pipeline automatizado para la **extracción de frases clave y resúmenes estructurados mediante Procesamiento de Lenguaje Natural (LSA / TruncatedSVD y TF-IDF)** sobre artículos técnicos y noticias del dataset **20 Newsgroups** de scikit-learn, transformándolos en guiones optimizados con formato vertical (9:16) y generando el payload JSON formal bajo la especificación *Script-to-Video* de la API REST de *Pictory*.
- **Entrada requerida:** Artículos en texto plano o documentos extensos. El script extrae y procesa los textos de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `scikit-learn`. (Incluye también los submódulos nativos `json` y `re`).
- **Instalación específica:** `pip install numpy pandas scikit-learn requests`
- **Ejecución:** `python ia_generativa/Pictory AI.py`
- **Recursos externos:** Requiere configurar los encabezados de autenticación (`X-Pictory-User-Id` y `Authorization: Bearer`) con credenciales activas de Pictory para enviar los trabajos de renderizado al endpoint de producción (`https://api.pictory.ai/v1/jobs/scriptToVideo`); de lo contrario, el script simula la construcción del payload estructurado y muestra el cliente HTTP listo para integración.
- **Observaciones:** Este script completa la biblioteca técnica de documentación para modelos generativos de video de formato corto del repositorio, integrando la selección inteligente de fragmentos de impacto mediante Machine Learning con la síntesis audiovisual automatizada.

### DeepBrain AI (AI Studios API - Educación Online y Generación de Cursos en Video)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y Avatares de IA (Educación en Línea y E-Learning Académico)
- **Archivo:** `ia_generativa/DeepBrain AI.py`
- **Función básica:** Implementa un pipeline automatizado para la **creación de cursos y lecciones educativas narradas por avatares de IA** utilizando la API REST v2 de *DeepBrain AI (AI Studios)*. El script procesa colecciones documentales técnicas de scikit-learn (`20newsgroups` en categorías de gráficos y sistemas), agrupa los conceptos pedagógicos mediante `KMeans` y TF-IDF para redactar guiones de clase estructurados de forma automatizada, y conecta con los endpoints de DeepBrain para generar los proyectos de video en idioma español.
- **Entrada requerida:** Guiones pedagógicos y metadatos de lecciones. El dataset base se carga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también los submódulos nativos `os`, `json` y `time`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/DeepBrain AI.py`
- **Recursos externos:** Requiere configurar la variable de entorno `DEEPBRAIN_API_KEY` (o `AISTUDIOS_API_KEY`) con una clave activa para interactuar con los endpoints productivos de AI Studios; de lo contrario, el cliente incluye un modo de simulación que valida la correcta construcción del payload y la estructura del guión.
- **Observaciones:** Este script añade la última integración clave para completar el bloque de modelos de generación de video con avatares orientados a la educación en línea en la biblioteca técnica del repositorio.

### InVideo (Redes sociales y automatización de videos verticales con InVideo AI API)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video y Creación de Contenido para Redes Sociales (TikTok, Instagram Reels y YouTube Shorts)
- **Archivo:** `ia_generativa/InVideo.py`
- **Función básica:** Implementa un pipeline automatizado para la **extracción de métricas inmobiliarias y estadísticas económicas** desde el dataset real **California Housing** de scikit-learn, transformando los hallazgos cuantitativos en guiones narrativos dinámicos de 30 segundos optimizados con ganchos de atención (*hooks*), transiciones rápidas, música de fondo y subtítulos virales para su envío automatizado mediante la API REST de *InVideo AI*.
- **Entrada requerida:** Datos analíticos y series tabulares de mercado. El dataset base se descarga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/InVideo.py`
- **Recursos externos:** Requiere configurar la variable de entorno `INVIDEO_API_KEY` con una clave activa de InVideo para interactuar con los endpoints productivos de generación de video; de lo contrario, el cliente gestiona la excepción y simula de forma estructurada la preparación del payload y las peticiones HTTP.
- **Observaciones:** Este script añade la última integración clave para completar el bloque de modelos de generación de video y contenido enfocado en redes sociales dentro de la biblioteca técnica del repositorio.


### Fliki (Creación de Contenido Rápido - Texto a Voz y Video con Fliki AI API)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Medios y Text-to-Video / Text-to-Speech (TTS)
- **Archivo:** `ia_generativa/Fliki AI.py`
- **Función básica:** Implementa un pipeline automatizado para la **creación rápida de contenidos de video y locuciones de alta calidad** utilizando la API REST de *Fliki AI*. El script analiza datos analíticos reales del mercado inmobiliario mediante scikit-learn (cargando el dataset `California Housing` y aplicando segmentación `KMeans` y escalado con `StandardScaler`), extrae los indicadores clave por segmento y traduce los hallazgos cuantitativos en un arreglo estructurado de escenas con texto para síntesis de voz (*TTS*) y consultas de medios de fondo (*B-Roll*), exportando el payload JSON listo para su integración.
- **Entrada requerida:** Datos analíticos inmobiliarios y métricas tabulares. El script descarga y procesa el dataset de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `numpy`, `pandas`, `scikit-learn`. (Incluye también el submódulo nativo `json`).
- **Instalación específica:** `pip install numpy pandas scikit-learn requests`
- **Ejecución:** `python ia_generativa/Fliki AI.py`
- **Recursos externos:** Requiere configurar la cabecera de autenticación con una clave activa de Fliki (`Bearer TU_FLIKI_API_KEY`) para interactuar con los endpoints de producción (`https://api.fliki.ai/v1/content/generate/text-to-video`); de lo contrario, el script simula la construcción completa de la carga útil y muestra el cliente HTTP preparado para su ejecución.
- **Observaciones:** Este script añade la última integración clave para completar el bloque de modelos de generación de contenido rápido, locución automatizada y video en la biblioteca técnica del repositorio.

### Sora y Veo (Simuladores y Cine Avanzado - Generación de Video Largo y Cinematográfico)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Video de Alta Gama (Simulación Física, Cine Avanzado y Control de Cámaras)
- **Archivo:** `ia_generativa/Sora API.py`
- **Función básica:** Implementa un pipeline cinematográfico avanzado para la **creación programática de tomas de alta gama utilizando OpenAI Sora 2 y Google Veo 3.1**. El script procesa colecciones documentales técnicas de scikit-learn (`20newsgroups` en categorías de ciencia y computación), agrupa los tópicos temáticos mediante `KMeans` y TF-IDF para redactar prompts cinemáticos estructurados, y conecta simultáneamente con los SDKs oficiales de OpenAI (`openai.OpenAI`) y Google GenAI (`google.genai`) para automatizar la solicitud de videos de larga duración con audio nativo y consistencia física fidedigna.
- **Entrada requerida:** Guiones y especificaciones de tomas cinematográficas. El dataset base se carga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `openai`, `google-genai`, `requests`, `pandas`, `scikit-learn`. (Incluye también los submódulos nativos `os`, `json` y `time`).
- **Instalación específica:** `pip install openai google-genai requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/Sora API.py`
- **Recursos externos:** Requiere configurar las variables de entorno `OPENAI_API_KEY` y `GEMINI_API_KEY` (o `GOOGLE_API_KEY`) con claves activas para interactuar con los endpoints de producción de Sora y Veo; de lo contrario, los clientes gestionan la excepción y simulan de forma estructurada la preparación de los trabajos asíncronos y operaciones de larga duración.
- **Observaciones:** Este script corona la biblioteca técnica de documentación para modelos generativos de video del repositorio, cerrando el ciclo de integración con las arquitecturas de simulación y generación cinematográfica más avanzadas del estado del arte.

### Humata AI (Análisis de Contratos e Informes - RAG y Procesamiento de Documentos PDF Largos)

- **Categoría:** Inteligencia Artificial / Procesamiento de Documentos, RAG (*Retrieval-Augmented Generation*) y Análisis Legal / Financiero
- **Archivo:** `ia_generativa/HumataAI.py`
- **Función básica:** Implementa un pipeline automatizado para el **análisis inteligente, extracción de cláusulas y consultas RAG sobre contratos e informes corporativos extensos** utilizando la API REST de *Humata AI*. El script extrae discusiones y documentos reales de scikit-learn (`20newsgroups`), construye un archivo de texto estructurado simulando un contrato corporativo con cláusulas legales y financieras, y gestiona tanto la subida del documento (*File Upload*) como la ejecución de preguntas analíticas en lenguaje natural con retorno de citas exactas.
- **Entrada requerida:** Documentos extensos en formato de texto plano o PDF (en este caso, generado de forma automatizada a partir de colecciones documentales de scikit-learn).
- **Dependencias:** `requests`, `pandas`, `scikit-learn`. (Incluye también los submódulos nativos `os` y `time`).
- **Instalación específica:** `pip install requests pandas scikit-learn`
- **Ejecución:** `python ia_generativa/HumataAI.py`
- **Recursos externos:** Requiere configurar la variable de entorno `HUMATA_API_KEY` con una clave activa de Humata AI para interactuar con los endpoints de producción (`https://api.humata.ai/v1`); de lo contrario, el script captura la excepción e incluye un modo de simulación que detalla la estructura de la carga útil y las peticiones preparadas.
- **Observaciones:** Este script incorpora una nueva capacidad clave de análisis documental y auditoría legal a la biblioteca técnica del repositorio, permitiendo procesar y extraer conocimiento automatizado de contratos complejos a escala.


### VAE (Variational Autoencoder - Reducción Dimensional Probabilística e Interpolación de Rostros)

- **Categoría:** Inteligencia Artificial / Modelos Generativos Profundos y Espacios Latentes Probabilísticos (Deep Learning / Visión por Computador)
- **Archivo:** `ia_generativa/VAE.py`
- **Función básica:** Implementa una arquitectura completa de **Autoencoder Variacional (VAE)** en PyTorch para la **generación, reconstrucción e interpolación suave en el espacio latente** utilizando el dataset real de rostros **LFW People (*Labeled Faces in the Wild*)** de scikit-learn. El pipeline incluye la normalización de tensores de píxeles, la parametrización mediante codificador probabilístico (*mu y logvar*) con el truco de reparametrización (*reparameterization trick*), la función de pérdida combinada de reconstrucción y divergencia de Kullback-Leibler (BCE + KLD), el entrenamiento iterativo y la interpolación lineal (*Spherical/Linear Interpolation*) entre dos identidades latentes distintas.
- **Entrada requerida:** Imágenes rasterizadas de rostros en escala de grises o color. El dataset base se descarga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `matplotlib`, `scikit-learn`.
- **Instalación específica:** `pip install torch torchvision numpy matplotlib scikit-learn`
- **Ejecución:** `python ia_generativa/VAE.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset LFW People desde scikit-learn. No requiere claves de API propietarias al ejecutarse localmente con PyTorch.
- **Observaciones:** En el script original, la línea para exportar la visualización gráfica (`plt.savefig`) se encuentra comentada (`#`). **Es recomendable descomentarla y asegurar una ruta de almacenamiento local** si deseas conservar la secuencia de interpolación facial en formato de imagen PNG de alta resolución.

### GAN (Generative Adversarial Networks - Deepfakes, Súper Resolución y Ruido Latente)

- **Categoría:** Inteligencia Artificial / Redes Neuronales Adversarias y Modelos Generativos Profundos (Deep Learning / Visión por Computador)
- **Archivo:** `ia_generativa/GAN.py`
- **Función básica:** Implementa una arquitectura completa de **Redes Adversarias Generativas (GAN)** en PyTorch para la **generación de rostros sintéticos (*Deepfakes*) y modelado de súper resolución a partir de ruido latente (espacio latente gaussiano)**, utilizando el dataset real de rostros **Olivetti Faces** de scikit-learn. El script configura el juego minimax competitivo entre el **Generador** (transforma vectores de ruido en imágenes sintéticas de 64x64 píxeles) y el **Discriminador** (clasificador binario entrenado para distinguir entre datos reales y falsificaciones), simulando la convergencia adversarial y exportando curvas de pérdida junto con una cuadrícula visual de rostros inexistentes generados por la IA.
- **Entrada requerida:** Imágenes en escala de grises de rostros humanos. El dataset base se descarga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python ia_generativa/GAN.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset Olivetti Faces desde scikit-learn. No requiere claves de API propietarias al ejecutarse localmente con PyTorch.
- **Observaciones:** El script guarda automáticamente el gráfico resumen con las curvas de entrenamiento minimax y la cuadrícula de Deepfakes en la ruta `/home/usuario/01_gan_deepfakes_latentes.png`. Asegúrate de contar con permisos de escritura en dicho directorio o modifica la ruta en el código si ejecutas el script en otro entorno local.

### Modelos de Difusión (DPM - Diffusion Probabilistic Models / Síntesis de Imágenes Fotorrealistas)

- **Categoría:** Inteligencia Artificial / Modelos Generativos de Difusión y Síntesis Basada en Denoisers (Deep Learning / Visión por Computador)
- **Archivo:** `ia_generativa/Modelos de Difusión (DPM).py`
- **Función básica:** Implementa un modelo probabilístico de difusión desentrenado desde cero (*DDPM*) en PyTorch para la **síntesis de imágenes de dígitos manuscritos** a partir de ruido gaussiano puro, utilizando el dataset real `digits` de scikit-learn (`load_digits`). El script estructura el calendario de ruido (*Diffusion Schedule* con $\beta$ y $\alpha$ acumulados), el proceso directo (*Forward Process* agregando ruido paso a paso), una red convolucional temporal para estimación de ruido (*SimpleDenoisingModel*), el ciclo de entrenamiento mediante minimización de error cuadrático medio (MSE), y el proceso inverso (*Reverse Process* / muestreo iterativo) para recuperar imágenes estructuradas.
- **Entrada requerida:** Matrices de imágenes normalizadas en escala de grises. El dataset base se carga y procesa de manera automatizada mediante scikit-learn, por lo que no requiere cargar archivos locales externos.
- **Dependencias:** `torch`, `numpy`, `matplotlib`, `scikit-learn`.
- **Instalación específica:** `pip install torch numpy matplotlib scikit-learn`
- **Ejecución:** `python ia_generativa/Modelos de Difusión (DPM).py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el dataset digits desde scikit-learn. Funciona de manera local tanto en CPU como en GPU mediante PyTorch sin necesidad de claves de API externas.
- **Observaciones:** El script finaliza ejecutando una rutina de muestreo que visualiza mediante Matplotlib una cuadrícula con los dígitos sintetizados a partir de ruido estocástico puro.

### BERT (Bidirectional Encoder Representations from Transformers - Atención al Cliente y Chatbots)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Modelos Basados en Transformers y Representaciones Bidireccionales
- **Archivo:** `procesamiento_lenguaje_natural/bert.py`
- **Función básica:** Implementa un pipeline completo con la librería `transformers` y `datasets` de Hugging Face para la **tokenización, carga de corpus documentales a gran escala y extracción de embeddings contextuales** utilizando el modelo pre-entrenado `bert-base-uncased`. El script tokeniza muestras de texto agregando tokens especiales (como `[CLS]` y `[SEP]`), procesa lotes masivos de datos descargados desde Wikimedia Wikipedia, y pasa los tensores resultantes a través de `BertModel` para generar representaciones vectoriales densas de alta dimensionalidad (`[batch_size, sequence_length, hidden_size]`) ideales para tareas de clasificación, chatbots y sistemas de atención al cliente.
- **Entrada requerida:** Corpus textuales o colecciones documentales (en este caso, descargadas y procesadas de forma automatizada mediante el conjunto de datos de Wikimedia en español).
- **Dependencias:** `transformers`, `datasets`, `torch`.
- **Instalación específica:** `pip install transformers datasets torch`
- **Ejecución:** `python procesamiento_lenguaje_natural/bert.py`
- **Recursos externos:** Requiere conexión a internet para descargar el modelo pre-entrenado `bert-base-uncased` desde el Hub de Hugging Face y los fragmentos del dataset de Wikipedia. No requiere claves de API propietarias.
- **Observaciones:** Este script consolida las capacidades de procesamiento de lenguaje natural y modelos basados en atención dentro de la biblioteca técnica del repositorio, facilitando la auditoría y despliegue de embeddings para motores de búsqueda y asistentes conversacionales.

### StructBERT (Gestión Documental Corporativa - Fine-Tuning y Clasificación de Texto)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Modelos de Lenguaje Basados en Enmascaramiento y Estructura Sintáctica (*StructBERT*)
- **Archivo:** `procesamiento_lenguaje_natural/structbert.py`
- **Función básica:** Implementa un pipeline completo de **ajuste fino (*fine-tuning*) y evaluación para clasificación de secuencias** utilizando la arquitectura `bayartsogt/structbert-large` de Hugging Face. El script construye un dataset local en memoria con ejemplos de texto y etiquetas binarias, inicializa el tokenizador y el modelo de clasificación de secuencias, preprocesa y tokeniza los textos, configura los argumentos de entrenamiento optimizados (`TrainingArguments`, `Trainer`, `DataCollatorWithPadding`) y ejecuta la optimización junto con la evaluación métrica de precisión (*accuracy*) mediante la librería `evaluate`.
- **Entrada requerida:** Datos textuales estructurados en memoria (diccionarios con textos y etiquetas). El script genera y procesa un dataset local de prueba de manera automatizada.
- **Dependencias:** `transformers`, `datasets`, `torch`, `numpy`, `evaluate`.
- **Instalación específica:** `pip install transformers datasets torch evaluate`
- **Ejecución:** `python procesamiento_lenguaje_natural/structbert.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el punto de control pre-entrenado `bayartsogt/structbert-large` desde el Hub de Hugging Face. No requiere claves de API propietarias.
- **Observaciones:** Este script añade una valiosa herramienta de clasificación documental basada en estructuras sintácticas avanzadas al repositorio, permitiendo automatizar la gestión y categorización de archivos de texto corporativos.


### GPT-3 (Asistentes Virtuales y Redacción Automática - Integración con OpenAI API)

- **Categoría:** Inteligencia Artificial / Modelos de Lenguaje Grande (LLMs), Asistentes Conversacionales y Automatización de Redacción
- **Archivo:** `procesamiento_lenguaje_natural/GPT3.py`
- **Función básica:** Implementa un script cliente interactivo utilizando la librería oficial `openai` para conectar con el modelo `gpt-3.5-turbo` a través del endpoint de Chat Completions. El programa solicita de forma dinámica la clave de API del usuario y el prompt en lenguaje natural, configura un rol de sistema (*system prompt*) para garantizar respuestas útiles y concisas, y gestiona los parámetros de generación como la longitud máxima de tokens (`max_tokens`) y la temperatura creativa (`temperature`).
- **Entrada requerida:** Prompts de texto en lenguaje natural y una clave de API válida de OpenAI.
- **Dependencias:** `openai`. (Incluye también el submódulo nativo `os`).
- **Instalación específica:** `pip install openai`
- **Ejecución:** `python procesamiento_lenguaje_natural/GPT3.py`
- **Recursos externos:** Requiere una clave de API activa de OpenAI para interactuar con los servidores de producción de chat completions.
- **Observaciones:** Este script se integra en la biblioteca técnica del repositorio para cubrir los requerimientos de asistentes conversacionales ligeros y tareas de redacción automática basadas en LLMs.

### Chatbots (Soporte Técnico de Nivel 1 - Clasificación de Texto e Inferencia Interactiva con StructBERT)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Chatbots Conversacionales y Ajuste Fino de Modelos Basados en Transformers
- **Archivo:** `procesamiento_lenguaje_natural/chatbot.py`
- **Función básica:** Implementa un pipeline integral de **entrenamiento, ajuste fino (*fine-tuning*) y despliegue interactivo de un chatbot de análisis de sentimiento y soporte de nivel 1** utilizando el modelo `bayartsogt/structbert-large` de Hugging Face. El script estructura un dataset local en memoria, configura el proceso de optimización mediante `Trainer` y `TrainingArguments`, evalúa el rendimiento con métricas de precisión, guarda los pesos del modelo localmente y despliega un bucle interactivo de inferencia en tiempo real basado en la abstracción `pipeline("text-classification")`.
- **Entrada requerida:** Dataset textual estructurado en memoria con etiquetas binarias. El script genera y procesa los datos de entrenamiento de manera automatizada.
- **Dependencias:** `transformers`, `datasets`, `torch`, `numpy`, `evaluate`.
- **Instalación específica:** `pip install transformers datasets torch evaluate`
- **Ejecución:** `python procesamiento_lenguaje_natural/chatbot.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el punto de control pre-entrenado `bayartsogt/structbert-large` desde el Hub de Hugging Face. No requiere claves de API propietarias.
- **Observaciones:** Este script corona la sección de asistentes conversacionales y atención al cliente, integrando el ciclo completo de machine learning desde el preprocesamiento y el fine-tuning hasta la interacción en consola mediante un pipeline de clasificación de texto.

### RoBERTa (Análisis de Sentimiento Avanzado - Modelos de Lenguaje Robustos con Hugging Face)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Modelos Basados en Transformers y Clasificación Avanzada de Sentimientos
- **Archivo:** `procesamiento_lenguaje_natural/roberta.py`
- **Función básica:** Implementa un pipeline automatizado para el **análisis de sentimiento avanzado en texto en bruto** utilizando la arquitectura `cardiffnlp/twitter-roberta-base-sentiment-latest` de Hugging Face. El script inicializa de forma dinámica el tokenizador y el modelo de clasificación optimizados sobre tweets y textos informales, procesa la entrada proporcionada por el usuario y evalúa la polaridad del mensaje retornando la etiqueta de sentimiento (negativo, neutral o positivo) junto con su puntaje de confianza exacto.
- **Entrada requerida:** Texto en bruto ingresado interactivamente por el usuario a través de la consola.
- **Dependencias:** `transformers`, `torch`.
- **Instalación específica:** `pip install transformers torch`
- **Ejecución:** `python procesamiento_lenguaje_natural/roberta.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el modelo pre-entrenado y su tokenizador desde el Hub de Hugging Face. No requiere claves de API propietarias.
- **Observaciones:** Este script enriquece la sección de procesamiento de lenguaje natural y modelos avanzados de atención en la biblioteca técnica del repositorio, permitiendo auditar con alta precisión la polaridad emocional en corpus documentales y flujos de redes sociales.

### Parsing (Extracción de Entidades Nombradas y Procesamiento de Salidas Estructuradas)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Extracción de Información y Normalización de Salidas de Modelos
- **Archivo:** `procesamiento_lenguaje_natural/parsing.py`
- **Función básica:** Implementa un módulo modular de **procesamiento y normalización de salidas (*parsing*)** para transformar las respuestas crudas de los pipelines de Hugging Face en estructuras de datos limpias y reutilizables. El script utiliza un modelo RoBERTa especializado para procesar texto ingresado por el usuario, aísla la lógica de extracción mediante una función dedicada (`parsear_salida`) y devuelve un diccionario normalizado con la etiqueta de clasificación y el puntaje de confianza exacto.
- **Entrada requerida:** Texto estructurado o en bruto ingresado interactivamente por el usuario.
- **Dependencias:** `transformers`, `torch`.
- **Instalación específica:** `pip install transformers torch`
- **Ejecución:** `python procesamiento_lenguaje_natural/parsing.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el modelo pre-entrenado desde Hugging Face. No requiere claves de API propietarias.
- **Observaciones:** Este script fortalece la infraestructura de preprocesamiento y post-procesamiento de datos del repositorio, asegurando que las inferencias de los modelos de lenguaje se traduzcan en estructuras limpias para bases de datos o sistemas downstream.

### XLNet (Análisis de Textos Largos - Modelos Autorregresivos Generalizados con Transformer-XL)

- **Categoría:** Inteligencia Artificial / Procesamiento de Lenguaje Natural (NLP), Modelos de Lenguaje Basados en Permutaciones y Arquitecturas de Contexto Extendido
- **Archivo:** `procesamiento_lenguaje_natural/XLnet.py`
- **Función básica:** Implementa un pipeline completo para la **clasificación de textos largos y análisis de documentos extensos** utilizando la arquitectura `xlnet-base-cased` de Hugging Face combinada con el dataset real `20 newsgroups` de scikit-learn. El script procesa categorías temáticas de noticias, realiza tokenización avanzada con `XLNetTokenizer` (manejando longitudes de secuencia optimizadas de hasta 512 tokens), configura los argumentos de entrenamiento con la clase `Trainer`, ejecuta inferencias personalizadas para calcular métricas de confianza y genera una matriz de confusión detallada junto con un reporte de clasificación completo.
- **Entrada requerida:** Colecciones documentales y artículos de texto extenso. El script descarga y procesa de manera automatizada las categorías seleccionadas del dataset de scikit-learn.
- **Dependencias:** `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install torch transformers datasets numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python procesamiento_lenguaje_natural/XLnet.py`
- **Recursos externos:** Requiere conexión a internet en su primera ejecución para descargar el modelo pre-entrenado `xlnet-base-cased` desde el Hub de Hugging Face y las particiones del dataset `20 newsgroups` desde scikit-learn. No requiere claves de API propietarias.
- **Observaciones:** Este script añade una de las capacidades más potentes de modelado de texto largo al repositorio, aprovechando el *Permutation Language Modeling* (PLM) y las ventajas arquitectónicas de Transformer-XL frente a las limitaciones de longitud típicas de BERT.

### RNN / LSTM (Traducción Automática y Forecasting - Redes Neuronales Recurrentes y Memoria a Largo Plazo)

- **Categoría:** Inteligencia Artificial / Redes Neuronales Recurrentes, Procesamiento de Secuencias Temporales y Modelos Seq2Seq (Deep Learning / Time Series Forecasting)
- **Archivo:** `procesamiento_lenguaje_natural/RNN-LSTM.py`
- **Función básica:** Implementa un pipeline completo de **predicción de series temporales (*forecasting*) y modelado secuencial** utilizando TensorFlow/Keras y datos reales del mercado financiero descargados mediante la librería `yfinance` (histórico de acciones de Apple / `AAPL`). El script abarca el preprocesamiento crítico de escalado al rango [0, 1] mediante `MinMaxScaler` de scikit-learn, la estructuración de ventanas temporales deslizantes (*lookback windows* de 60 días), la construcción de una arquitectura neuronal profunda con celdas **LSTM** y capas de regularización *Dropout*, el entrenamiento iterativo con optimizador Adam, la desnormalización de predicciones a valores en dólares reales (USD), y la evaluación estadística completa calculando métricas de error (MAE, RMSE y $R^2$ Score).
- **Entrada requerida:** Series temporales de precios o valores financieros históricos. El script descarga los datos financieros de manera automatizada a través de la API de Yahoo Finance, por lo que no requiere archivos locales previos.
- **Dependencias:** `tensorflow`, `yfinance`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install tensorflow yfinance numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python procesamiento_lenguaje_natural/RNN-LSTM.py`
- **Recursos externos:** Requiere conexión a internet activa para descargar el dataset histórico de cotizaciones bursátiles desde Yahoo Finance. No requiere claves de API propietarias.
- **Observaciones:** El script genera automáticamente un gráfico analítico que muestra la evolución de las pérdidas (*loss*) durante el entrenamiento y una comparativa visual detallada con zoom entre los precios reales del mercado y la proyección generada por el modelo LSTM. Al igual que en scripts anteriores, la línea para guardar la figura (`plt.savefig`) se encuentra comentada (`#`), por lo que puedes activarla si deseas exportar el gráfico en alta resolución.

### Transformers (Generación y Traducción Avanzada - Mecanismo de Atención y Tokens Posicionales)

- **Categoría:** Inteligencia Artificial / Arquitecturas de Atención Global, Modelos Generativos de Lenguaje y Traducción Automática de Alta Gama
- **Archivo:** `procesamiento_lenguaje_natural/Transformers.py`
- **Función básica:** Implementa un pipeline completo de **generación de texto causal (*Text Generation*) y traducción automática neuronal (*Seq2Seq*)** utilizando la librería `transformers` de Hugging Face y datos reales extraídos del dataset `20 newsgroups` de scikit-learn (`sci.space`). El script toma un fragmento de texto científico como *prompt* semilla, lo alimenta a un modelo generativo basado en Decoder (`gpt2`) para la inferencia de continuación estocástica, procesa el texto resultante a través de un modelo de traducción Encoder-Decoder de alta precisión (`Helsinki-NLP/opus-mt-en-es`) para convertirlo del inglés al español, y simula analíticamente el mecanismo de **auto-atención (*Self-Attention*)** y la complejidad computacional cuadrática característica de las arquitecturas de atención.
- **Entrada requerida:** Fragmentos de texto en bruto o colecciones documentales. El script descarga de manera automatizada las noticias espaciales de scikit-learn para utilizarlas como semilla de generación.
- **Dependencias:** `transformers`, `torch`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
- **Instalación específica:** `pip install transformers torch numpy pandas matplotlib seaborn scikit-learn`
- **Ejecución:** `python procesamiento_lenguaje_natural/Transformers.py`
- **Recursos externos:** Requiere conexión a internet activa en su primera ejecución para descargar los modelos pre-entrenados (`gpt2` y `opus-mt-en-es`) desde el Hub de Hugging Face y las particiones de scikit-learn. No requiere claves de API propietarias.
- **Observaciones:** Este script consolida y sella la biblioteca técnica de documentación para modelos avanzados de procesamiento de lenguaje natural y arquitecturas de atención del repositorio, interconectando de forma práctica la teoría fundacional de los Transformers con pipelines reales de inferencia y traducción.



## Notas de documentación

- La columna de aplicaciones es una referencia inicial y no sustituye la inspección del código.
- No se exige documentar el dataset cuando no se conoce o no está registrado.
- Las dependencias deben extraerse de los imports y verificarse contra el entorno real; no listar una librería solo porque sea común en esa categoría.
- Los comandos de ejecución deben probarse o verificarse antes de marcarlos como definitivos.
- Si un elemento es una plataforma o servicio externo y no un script Python, documentar su acceso y requisitos en lugar de inventar un archivo `.py`.
- Mantener la misma ficha para todas las implementaciones ayuda a que el repositorio sea consistente y fácil de mantener.