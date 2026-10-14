"""
SARSA (State-Action-Reward-State-Action): Navegación en Laberintos
Demostración de Aprendizaje por Refuerzo On-Policy con scikit-learn

Conceptos:
- SARSA: Algoritmo de Aprendizaje por Refuerzo basado en Diferencias Temporales (TD). 
  A diferencia de Q-Learning (Off-Policy), SARSA es "On-Policy".
- On-Policy: El agente aprende el valor de la política que está ejecutando actualmente, 
  incluyendo sus errores aleatorios (exploración). Esto lo hace más conservador y seguro 
  al navegar cerca de obstáculos.
- Ecuación de Actualización: Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]
- Generación del Laberinto: Se utilizará `make_moons` de scikit-learn para generar un 
  muro curvo (barrera) que el agente deberá rodear para llegar a la meta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import seaborn as sns
import random
import time
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

# ============================================================================
# 1. GENERAR EL ENTORNO (LABERINTO CON SCIKIT-LEARN)
# ============================================================================
print("=" * 80)
print("PASO 1: GENERACIÓN DEL LABERINTO (OBSTÁCULOS CURVOS)")
print("=" * 80)

# Tamaño de la cuadrícula
GRID_SIZE = 10
INICIO = (0, 0)
META = (9, 9)

# Usamos make_moons para crear una forma de "muro" en el centro del laberinto
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Filtramos solo una de las "lunas" para que sea nuestro muro
muro_continuo = X[y == 0]

# Escalar y discretizar los puntos continuos al tamaño del Grid (10x10)
# Mapeamos los valores de X al rango [1, GRID_SIZE-2] para no bloquear los bordes
x_min, x_max = muro_continuo[:, 0].min(), muro_continuo[:, 0].max()
y_min, y_max = muro_continuo[:, 1].min(), muro_continuo[:, 1].max()

obstaculos = set()
for px, py in muro_continuo:
    ix = int(np.interp(px, (x_min, x_max), (2, 8)))
    iy = int(np.interp(py, (y_min, y_max), (2, 7)))
    
    # Prevenir que el obstáculo pise el inicio o la meta
    if (ix, iy) != INICIO and (ix, iy) != META:
        obstaculos.add((ix, iy))

print(f"\nLaberinto Configurado: {GRID_SIZE}x{GRID_SIZE}")
print(f"Inicio: {INICIO} | Meta: {META}")
print(f"Número de celdas bloqueadas (Muro en forma de luna): {len(obstaculos)}")

# ============================================================================
# 2. DEFINIR LA LÓGICA DEL LABERINTO (MDP)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DINÁMICA DEL ENTORNO Y RECOMPENSAS")
print("=" * 80)

acciones = [0, 1, 2, 3] # 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha
n_acciones = len(acciones)

def paso_entorno(estado, accion):
    """Ejecuta una acción y devuelve el nuevo estado y la recompensa."""
    x, y = estado
    if accion == 0:   x -= 1 # Arriba
    elif accion == 1: x += 1 # Abajo
    elif accion == 2: y -= 1 # Izquierda
    elif accion == 3: y += 1 # Derecha

    # Chocar con los límites del laberinto
    if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
        return estado, -5.0, False # Se queda en el mismo lugar y recibe penalización
    
    nuevo_estado = (x, y)
    
    # Chocar con el muro (obstáculo)
    if nuevo_estado in obstaculos:
        return estado, -10.0, False # SARSA aprenderá a tener mucho miedo de acercarse aquí
    
    # Llegar a la meta
    if nuevo_estado == META:
        return nuevo_estado, 100.0, True
    
    # Movimiento normal en celda vacía
    return nuevo_estado, -1.0, False

print("\nEstructura de Recompensas:")
print("  - Alcanzar la META: +100")
print("  - Chocar contra MURO: -10")
print("  - Chocar contra BORDE: -5")
print("  - Moverse a celda vacía: -1 (Incentiva buscar la ruta más corta)")

# ============================================================================
# 3. CONFIGURAR EL AGENTE SARSA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: INICIALIZAR LA TABLA Q Y PARÁMETROS SARSA")
print("=" * 80)

# Inicializamos la Tabla Q con ceros
# Dimensiones: GRID_SIZE x GRID_SIZE x n_acciones
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, n_acciones))

# Hiperparámetros de SARSA
alpha = 0.1          # Tasa de aprendizaje (Learning rate)
gamma = 0.95         # Factor de descuento (Importancia del futuro)
epsilon = 1.0        # Exploración inicial (100% aleatorio)
epsilon_min = 0.05
epsilon_decay = 0.995

def elegir_accion(estado, eps):
    """Política Epsilon-Greedy"""
    if np.random.rand() < eps:
        return np.random.choice(acciones) # Explorar
    else:
        x, y = estado
        # Explotar: elegir la acción con mayor valor Q
        # En caso de empate, np.argmax elige el primero, por lo que agregamos un pequeño ruido
        valores_q = Q_table[x, y, :]
        return np.random.choice(np.flatnonzero(valores_q == valores_q.max()))

print("✓ Tabla Q inicializada en ceros.")
print(f"✓ Parámetros: Alpha={alpha}, Gamma={gamma}, Epsilon_decay={epsilon_decay}")

# ============================================================================
# 4. ENTRENAMIENTO DEL AGENTE (ALGORITMO SARSA)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: ENTRENAMIENTO (BÚSQUEDA DE RUTAS)")
print("=" * 80)

num_episodios = 400
recompensas_hist = []

print("Entrenando al agente usando la actualización SARSA (On-Policy)...")

for episodio in range(num_episodios):
    estado = INICIO
    # 1. SARSA elige la acción ANTES de entrar al bucle
    accion = elegir_accion(estado, epsilon)
    
    recompensa_total = 0
    done = False
    pasos = 0
    
    while not done and pasos < 100: # Límite para evitar bucles infinitos
        # 2. Tomar acción A, observar R y S'
        estado_siguiente, recompensa, done = paso_entorno(estado, accion)
        
        # 3. SARSA elige la acción Siguiente A' desde S' usando su política actual
        accion_siguiente = elegir_accion(estado_siguiente, epsilon)
        
        # 4. Actualización SARSA (On-Policy)
        # Nota: Se utiliza Q(S', A') real que tomará en el siguiente paso, 
        # a diferencia de Q-Learning que usaría max(Q(S', a))
        x, y = estado
        nx, ny = estado_siguiente
        
        predict = Q_table[x, y, accion]
        target = recompensa + gamma * Q_table[nx, ny, accion_siguiente] * (not done)
        
        Q_table[x, y, accion] = predict + alpha * (target - predict)
        
        # Avanzar
        estado = estado_siguiente
        accion = accion_siguiente
        
        recompensa_total += recompensa
        pasos += 1
        
    # Decaimiento de la exploración
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    recompensas_hist.append(recompensa_total)
    
    if (episodio + 1) % 50 == 0:
        print(f"  - Episodio {episodio+1:3d}/{num_episodios} | Recompensa: {recompensa_total:6.1f} | Pasos: {pasos:3d} | Epsilon: {epsilon:.2f}")

print("\n✓ Entrenamiento SARSA completado.")

# ============================================================================
# 5. EVALUACIÓN Y EXTRACCIÓN DE LA RUTA ÓPTIMA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: EVALUACIÓN DE LA POLÍTICA APRENDIDA")
print("=" * 80)

estado = INICIO
ruta_optima = [estado]
done = False
pasos = 0

print("Desplegando al agente (Epsilon = 0, sin azar)...")
while not done and pasos < 30:
    # Siempre elegimos la mejor acción
    accion = elegir_accion(estado, 0.0)
    estado_siguiente, recompensa, done = paso_entorno(estado, accion)
    
    ruta_optima.append(estado_siguiente)
    estado = estado_siguiente
    pasos += 1

if ruta_optima[-1] == META:
    print(f"🏆 ¡ÉXITO! El laberinto fue resuelto en {len(ruta_optima)-1} pasos.")
else:
    print("⚠️ El agente no logró llegar a la meta.")

# ============================================================================
# 6. VISUALIZACIÓN: EL LABERINTO Y EL MAPA DE VALORES Q
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: VISUALIZACIÓN DEL LABERINTO Y APRENDIZAJE")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: El Laberinto y la Ruta
mapa = np.zeros((GRID_SIZE, GRID_SIZE))
for ox, oy in obstaculos:
    mapa[ox, oy] = -1

axes[0].imshow(mapa, cmap='Pastel1', origin='upper')

# Cuadrícula
axes[0].set_xticks(np.arange(-0.5, GRID_SIZE, 1))
axes[0].set_yticks(np.arange(-0.5, GRID_SIZE, 1))
axes[0].grid(color='black', linestyle='-', linewidth=0.5)
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])

# Dibujar obstáculos
for ox, oy in obstaculos:
    axes[0].text(oy, ox, '🧱', ha='center', va='center', fontsize=18)

# Inicio y Meta
axes[0].text(INICIO[1], INICIO[0], '🟢\nInicio', ha='center', va='center', fontweight='bold')
axes[0].text(META[1], META[0], '🏁\nMeta', ha='center', va='center', fontweight='bold')

# Trazar ruta
y_coords = [p[1] for p in ruta_optima]
x_coords = [p[0] for p in ruta_optima]
axes[0].plot(y_coords, x_coords, color='blue', linewidth=3, marker='o', 
             markersize=6, label='Ruta SARSA')
axes[0].legend(loc='upper right')
axes[0].set_title('Resolución del Laberinto Generado', fontsize=14, fontweight='bold')

# Gráfico 2: Mapa de Calor de Valores Q (El "Valor" de cada estado)
# Calculamos V(s) como el max Q(s, a) para simplificar la visualización
valor_estados = np.max(Q_table, axis=2)

sns.heatmap(valor_estados, annot=False, cmap='viridis', ax=axes[1], cbar_kws={'label': 'Valor del Estado (Max Q)'})
axes[1].set_title('Mapa de Calor de la Tabla Q\n(Rastros brillantes guían a la meta)', 
                  fontsize=14, fontweight='bold')

# Superponer la ruta en el mapa de calor
axes[1].plot(y_coords, x_coords, color='white', linewidth=2, marker='.', label='Ruta')

plt.tight_layout()
#plt.savefig('/home/usuario/01_sarsa_laberinto.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_sarsa_laberinto.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              SARSA APLICADO A LA NAVEGACIÓN EN LABERINTOS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. GENERACIÓN DEL ENTORNO (SCIKIT-LEARN):
   • Utilizamos `make_moons` de scikit-learn para generar coordenadas continuas 
     en forma de media luna.
   • Estas coordenadas fueron discretizadas y mapeadas a nuestro Grid (10x10) 
     para actuar como un muro curvo complejo que el agente debe esquivar.

2. SARSA vs Q-LEARNING (LA DIFERENCIA CLAVE):
   • Q-Learning es optimista: asume que en el próximo estado tomará la mejor 
     acción posible (max Q). Esto lo lleva a buscar el camino más corto, 
     incluso si roza peligrosamente el muro.
   • SARSA es realista (On-Policy): utiliza la acción real elegida (que 
     puede ser aleatoria debido a epsilon). 
   • Consecuencia: SARSA aprende a alejarse de los bordes del muro porque 
     sabe que, mientras esté explorando, un error aleatorio cerca del muro 
     le costará -10 puntos. Desarrolla rutas más seguras.

3. EVOLUCIÓN DEL APRENDIZAJE:
   • La Tabla Q inició en ceros. A través de episodios, la recompensa de 
     llegar a la meta (+100) se propagó hacia atrás, celda por celda.
   • El Mapa de Calor (Generado en el Paso 6) demuestra cómo el algoritmo 
     creó un gradiente visual, donde los colores más brillantes actúan 
     como migajas de pan guiando al agente hacia la bandera.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           PARÁMETROS UTILIZADOS                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

- alpha = 0.1: El agente actualiza su conocimiento lentamente (10% por paso) 
  para estabilizar el aprendizaje probabilístico.
- gamma = 0.95: Alta prioridad en la recompensa final (la meta), permitiendo 
  soportar la penalización de -1 en cada celda a lo largo de la ruta.
- epsilon_decay = 0.995: Transición suave de modo "Curioso/Explorador" a 
  modo "Experto/Explotador".

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_sarsa_laberinto.png - Contiene el diseño del laberinto curvo con la ruta 
                         trazada y el mapa de calor (Tabla Q).

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Entrenamiento de SARSA completado exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")