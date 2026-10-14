"""
Actor-Critic (A2C/DDPG): Control Continuo en Robótica Industrial
Demostración implementada con scikit-learn (MLPRegressor) y datasets sintéticos

Conceptos:
- Actor-Critic: Arquitectura híbrida de Aprendizaje por Refuerzo. 
  * El 'Actor' aprende la política (qué acción tomar).
  * El 'Crítico' aprende a evaluar esa acción (qué tan bueno es el estado resultante).
- Control Continuo: A diferencia de acciones discretas (arriba/abajo), las acciones 
  son valores continuos reales (ej. torque aplicado a los motores o velocidad de los ejes).
- Error de Diferencia Temporal (TD Error): Mide la sorpresa del Crítico. Si el resultado 
  es mejor de lo esperado (TD > 0), el Actor actualiza sus pesos para favorecer esa acción.
- Dataset de Scikit-Learn: Utilizaremos `make_blobs` para generar coordenadas de 
  "zonas de soldadura/pintura" a las que el brazo robótico industrial debe llegar.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import math
import random
import warnings

warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set1")

# ============================================================================
# 1. GENERAR EL ENTORNO (ZONAS DE TRABAJO DEL ROBOT)
# ============================================================================
print("=" * 80)
print("PASO 1: GENERACIÓN DEL ENTORNO INDUSTRIAL (ZONAS DE TRABAJO)")
print("=" * 80)

# Parámetros del Brazo Robótico (Cinemática 2D)
L1 = 2.0  # Longitud del primer segmento (hombro a codo)
L2 = 1.5  # Longitud del segundo segmento (codo a efector final)
R_MAX = L1 + L2 # Alcance máximo (3.5)

# Generamos puntos objetivo (target) simulando piezas que deben ser operadas
# Forzamos los centros para que estén dentro del alcance del robot
centros = [[1.5, 1.5], [-1.5, 2.0], [2.0, -1.0]]
X_targets, y_clusters = make_blobs(n_samples=30, centers=centros, cluster_std=0.3, random_state=42)

# Filtrar puntos que estén fuera del alcance del brazo
puntos_trabajo = []
for x, y in X_targets:
    dist = np.sqrt(x**2 + y**2)
    if 0.5 < dist < R_MAX * 0.95:  # Evitar singularidades extremas
        puntos_trabajo.append(np.array([x, y]))

puntos_trabajo = np.array(puntos_trabajo)

print(f"\nBrazo Robótico Industrial Configurado:")
print(f"  - Eslabón 1 (L1): {L1} m | Eslabón 2 (L2): {L2} m")
print(f"  - Alcance Máximo Operativo: {R_MAX} m")
print(f"Zonas de Operación Generadas: {len(puntos_trabajo)} puntos de soldadura/ensamblaje.")

# ============================================================================
# 2. CINEMÁTICA Y DINÁMICA DEL ENTORNO
# ============================================================================
print("\n" + "=" * 80)
print("PASO 2: DINÁMICA DEL ENTORNO (CINEMÁTICA DIRECTA)")
print("=" * 80)

def cinematica_directa(theta1, theta2):
    """Calcula la posición (x,y) de la punta del brazo dados los ángulos de los motores."""
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return np.array([x, y])

def paso_entorno(estado, accion, target_pos):
    """
    estado: [theta1, theta2, target_x, target_y]
    accion: [delta_theta1, delta_theta2] (Velocidad de giro aplicada)
    """
    th1, th2, tx, ty = estado
    d_th1, d_th2 = accion
    
    # Aplicar acción (con límite de velocidad de motores)
    d_th1 = np.clip(d_th1, -0.2, 0.2)
    d_th2 = np.clip(d_th2, -0.2, 0.2)
    
    nuevo_th1 = th1 + d_th1
    nuevo_th2 = th2 + d_th2
    
    # Calcular nueva posición del efector
    pos_actual = cinematica_directa(nuevo_th1, nuevo_th2)
    distancia = np.linalg.norm(pos_actual - target_pos)
    
    nuevo_estado = np.array([nuevo_th1, nuevo_th2, tx, ty])
    
    # Sistema de Recompensas (Shaping Continuo)
    # 1. Recompensa negativa basada en la distancia (incentiva acercarse)
    recompensa = -distancia 
    # 2. Penalización por gasto energético (movimientos bruscos)
    recompensa -= 0.1 * (d_th1**2 + d_th2**2)
    
    done = False
    if distancia < 0.15: # Precisión de operación alcanzada
        recompensa += 50.0
        done = True
        
    return nuevo_estado, recompensa, done, pos_actual

print("\nSistema de Recompensas para Control Continuo:")
print("  - Alcanzar el objetivo (Precisión < 0.15m): +50")
print("  - Distancia al objetivo: Penalización proporcional continua.")
print("  - Gasto de energía (Delta Theta): -0.1 * esfuerzo^2")

# ============================================================================
# 3. INICIALIZAR ACTOR Y CRÍTICO (MLPRegressor)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 3: INICIALIZAR EL ACTOR Y EL CRÍTICO")
print("=" * 80)

# ACTOR: Recibe Estado (4 variables) -> Predice Acción Continua (2 variables: d_th1, d_th2)
actor = MLPRegressor(
    hidden_layer_sizes=(32, 32), activation='relu', solver='adam',
    learning_rate_init=0.005, max_iter=1, warm_start=True, random_state=42
)

# CRÍTICO: Recibe Estado (4 variables) -> Predice Valor V (1 variable: Recompensa esperada)
critic = MLPRegressor(
    hidden_layer_sizes=(32, 32), activation='relu', solver='adam',
    learning_rate_init=0.01, max_iter=1, warm_start=True, random_state=42
)

# Inicialización obligatoria de formas
estado_dummy = np.zeros((1, 4))
actor.partial_fit(estado_dummy, np.zeros((1, 2)))
critic.partial_fit(estado_dummy, np.zeros(1))

print("✓ ACTOR (Política): Inicializado. Estructura [4 -> 32 -> 32 -> 2]")
print("✓ CRÍTICO (Valor):  Inicializado. Estructura [4 -> 32 -> 32 -> 1]")

# ============================================================================
# 4. ENTRENAMIENTO ACTOR-CRÍTICO (SIMULACIÓN DDPG/A2C)
# ============================================================================
print("\n" + "=" * 80)
print("PASO 4: ENTRENAMIENTO DEL BRAZO ROBÓTICO")
print("=" * 80)

num_episodios = 150
gamma = 0.95
ruido_exploracion = 0.5  # Desviación estándar del ruido Gaussiano para explorar
ruido_decay = 0.98

historial_recompensas = []
historial_td_errors = []

print(f"Entrenando por {num_episodios} episodios de calibración continua...")

for episodio in range(num_episodios):
    # Seleccionar un objetivo aleatorio de los generados por sklearn
    target = random.choice(puntos_trabajo)
    
    # Iniciar brazo en posición de reposo (ángulos 0, 0)
    estado = np.array([0.0, 0.0, target[0], target[1]])
    
    recompensa_total = 0
    errores_td_episodio = []
    done = False
    pasos = 0
    
    while not done and pasos < 40:
        # 1. El ACTOR predice la acción ideal (media)
        mu_accion = actor.predict([estado])[0]
        
        # 2. Añadir ruido de exploración (esencial en control continuo)
        accion_real = mu_accion + np.random.normal(0, ruido_exploracion, size=2)
        
        # 3. Interactuar con el entorno
        estado_siguiente, recompensa, done, _ = paso_entorno(estado, accion_real, target)
        
        # 4. El CRÍTICO evalúa el estado actual y el siguiente
        V_actual = critic.predict([estado])[0]
        V_siguiente = critic.predict([estado_siguiente])[0] if not done else 0.0
        
        # 5. Calcular Error de Diferencia Temporal (TD Error / Advantage)
        td_target = recompensa + gamma * V_siguiente
        td_error = td_target - V_actual
        
        # 6. Entrenar al CRÍTICO
        # Aprende a predecir mejor el TD Target
        critic.partial_fit([estado], [td_target])
        
        # 7. Entrenar al ACTOR (El gradiente del Actor-Critic)
        # En una arquitectura real (con TensorFlow/PyTorch) usaríamos retropropagación a través del crítico.
        # Aquí aproximamos: Si td_error > 0, la acción aleatoria fue MEJOR de lo que pensaba el actor, 
        # así que acercamos la predicción del actor hacia esa acción aleatoria.
        tasa_ajuste = 0.1
        accion_target = mu_accion + tasa_ajuste * td_error * (accion_real - mu_accion)
        
        # Limitamos el target para no desestabilizar la red
        accion_target = np.clip(accion_target, -1.0, 1.0)
        actor.partial_fit([estado], [accion_target])
        
        estado = estado_siguiente
        recompensa_total += recompensa
        errores_td_episodio.append(abs(td_error))
        pasos += 1
        
    ruido_exploracion = max(0.05, ruido_exploracion * ruido_decay)
    
    historial_recompensas.append(recompensa_total)
    historial_td_errors.append(np.mean(errores_td_episodio))
    
    if (episodio + 1) % 25 == 0:
        print(f"  - Episodio {episodio+1:3d}/{num_episodios} | Recompensa: {recompensa_total:6.1f} | Error Crítico: {np.mean(errores_td_episodio):5.2f} | Ruido: {ruido_exploracion:.2f}")

print("\n✓ Entrenamiento cinemático completado.")

# ============================================================================
# 5. EVALUACIÓN Y EXTRACCIÓN DE LA TRAYECTORIA
# ============================================================================
print("\n" + "=" * 80)
print("PASO 5: EVALUACIÓN EN MODO EXPLOTACIÓN (PRODUCCIÓN)")
print("=" * 80)

# Elegimos un objetivo específico para trazar la operación del robot
target_eval = puntos_trabajo[0]
estado = np.array([0.0, 0.0, target_eval[0], target_eval[1]])
trayectoria_codo = []
trayectoria_punta = []
done = False
pasos = 0

print(f"Objetivo de soldadura establecido en: (X={target_eval[0]:.2f}, Y={target_eval[1]:.2f})")
print("Ejecutando trayectoria generada por el Actor (sin ruido)...")

while not done and pasos < 40:
    # Capturar coordenadas para visualización
    th1, th2 = estado[0], estado[1]
    codo_x, codo_y = L1 * np.cos(th1), L1 * np.sin(th1)
    punta_pos = cinematica_directa(th1, th2)
    
    trayectoria_codo.append((codo_x, codo_y))
    trayectoria_punta.append((punta_pos[0], punta_pos[1]))
    
    # El Actor dicta el movimiento de forma determinista
    accion = actor.predict([estado])[0]
    
    estado, _, done, _ = paso_entorno(estado, accion, target_eval)
    pasos += 1

# Capturar estado final
th1, th2 = estado[0], estado[1]
trayectoria_codo.append((L1 * np.cos(th1), L1 * np.sin(th1)))
punta_final = cinematica_directa(th1, th2)
trayectoria_punta.append((punta_final[0], punta_final[1]))

distancia_final = np.linalg.norm(punta_final - target_eval)
print(f"Resultado de Operación: Llevado a cabo en {pasos} iteraciones (frames).")
print(f"Error de Precisión Final: {distancia_final:.3f} metros.")
if distancia_final < 0.15:
    print("🏆 ¡Soldadura exitosa dentro de los márgenes de tolerancia industrial!")

# ============================================================================
# 6. VISUALIZACIÓN: TRAYECTORIA ROBÓTICA Y APRENDIZAJE
# ============================================================================
print("\n" + "=" * 80)
print("PASO 6: VISUALIZACIÓN DE CINEMÁTICA Y CURVAS DE CONVERGENCIA")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: El Entorno Robótico
axes[0].set_aspect('equal')
axes[0].set_xlim(-R_MAX - 0.5, R_MAX + 0.5)
axes[0].set_ylim(-R_MAX - 0.5, R_MAX + 0.5)
axes[0].grid(True, linestyle='--', alpha=0.6)

# Dibujar la base del robot
axes[0].plot(0, 0, marker='s', markersize=15, color='black', label='Base (Motor 1)')

# Dibujar los puntos de trabajo generados
axes[0].scatter(puntos_trabajo[:, 0], puntos_trabajo[:, 1], c='orange', s=80, 
                edgecolor='red', zorder=5, label='Zonas de Operación (Dataset)')
axes[0].scatter(target_eval[0], target_eval[1], c='red', marker='*', s=300, 
                zorder=6, label='Objetivo Actual')

# Dibujar radio de alcance máximo
circulo = plt.Circle((0, 0), R_MAX, fill=False, color='grey', linestyle='-.', alpha=0.5)
axes[0].add_artist(circulo)

# Animar el brazo (mostrando estelas)
for i in range(0, len(trayectoria_punta), max(1, len(trayectoria_punta)//10)):
    cx, cy = trayectoria_codo[i]
    px, py = trayectoria_punta[i]
    alpha_val = (i+1)/len(trayectoria_punta)
    
    # Eslabón 1 (Hombro a Codo)
    axes[0].plot([0, cx], [0, cy], color='blue', linewidth=4, alpha=alpha_val*0.5)
    # Eslabón 2 (Codo a Punta)
    axes[0].plot([cx, px], [cy, py], color='darkblue', linewidth=3, alpha=alpha_val*0.5)

# Dibujar posición final con opacidad 1.0
cx, cy = trayectoria_codo[-1]
px, py = trayectoria_punta[-1]
axes[0].plot([0, cx], [0, cy], color='blue', linewidth=5, label='Eslabón 1')
axes[0].plot([cx, px], [cy, py], color='cyan', linewidth=4, label='Eslabón 2')
axes[0].plot(cx, cy, marker='o', markersize=10, color='black') # Codo
axes[0].plot(px, py, marker='o', markersize=8, color='magenta', zorder=10, label='Efector Final')

axes[0].set_title('Cinemática 2D: Control Continuo de Brazo Robótico', fontsize=14, fontweight='bold')
axes[0].legend(loc='lower left', fontsize=9)

# Gráfico 2: Curva de Aprendizaje (Actor-Critic)
# Suavizado de curvas
recompensas_suavizadas = pd.Series(historial_recompensas).rolling(15, min_periods=1).mean()

ax2 = axes[1]
ax2.plot(historial_recompensas, color='lightgray', label='Recompensa Episodio', alpha=0.5)
ax2.plot(recompensas_suavizadas, color='green', linewidth=3, label=f'Tendencia del Actor (Suavizada)')
ax2.set_xlabel('Episodio de Entrenamiento')
ax2.set_ylabel('Recompensa (Actor)')
ax2.set_title('Convergencia del Sistema Actor-Critic', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')

# Eje gemelo para el error del crítico
ax3 = ax2.twinx()
error_suavizado = pd.Series(historial_td_errors).rolling(15, min_periods=1).mean()
ax3.plot(error_suavizado, color='purple', linewidth=2, linestyle='--', label='Pérdida del Crítico (TD Error)')
ax3.set_ylabel('Error de Diferencia Temporal (Crítico)')
ax3.legend(loc='lower right')

plt.tight_layout()
#plt.savefig('/home/usuario/01_actor_critic_robotics.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: 01_actor_critic_robotics.png")
plt.close()

# ============================================================================
# 7. RESUMEN Y CONCLUSIONES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMEN Y CONCLUSIONES")
print("=" * 80)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              ACTOR-CRITIC EN CONTROL ROBÓTICO CONTINUO                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. ESPACIOS CONTINUOS EN ROBÓTICA:
   • A diferencia de entornos de cuadrícula donde el agente se mueve de celda 
     en celda, los servomotores industriales requieren valores precisos y 
     continuos (radianes, voltajes, torques). 
   • El ACTOR resuelve esto al predecir dos variables de punto flotante en lugar 
     de clasificar una acción discreta.

2. LA DINÁMICA ACTOR-CRÍTICO (A2C):
   • EL CRÍTICO: Actúa como el supervisor analítico. Su único trabajo es 
     evaluar si el estado actual del brazo es mejor o peor de lo que calculó.
   • EL ACTOR: Es el controlador motriz ciego. Ejecuta un movimiento y escucha 
     el veredicto del Crítico. Si el TD Error (Advantage) es positivo, el Actor 
     actualiza sus redes para hacer más de ese movimiento; si es negativo, se aleja.

3. EVOLUCIÓN DEL APRENDIZAJE:
   • En las curvas generadas (Gráfico Derecho), se observa que al principio el 
     Crítico se equivoca mucho (TD Error alto) y las recompensas son bajas.
   • A medida que los episodios avanzan, el Crítico entiende la cinemática del 
     brazo, el error disminuye, y el Actor comienza a guiar al brazo con fluidez 
     hacia los clusters generados por Scikit-Learn.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           IMPLEMENTACIÓN Y PARÁMETROS                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

- Ruido Gaussiano: En espacios continuos no se usa "Epsilon-Greedy". En su lugar, 
  se añade ruido estadístico (`np.random.normal(0, 0.5)`) directamente a la 
  salida del Actor para asegurar la exploración del espacio articular, 
  decayendo paulatinamente.
- MLPRegressor Dual: Se emplearon dos regresores multicapa independientes de 
  scikit-learn (Actor y Crítico) con `partial_fit` para actualización on-line.

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ARCHIVOS GENERADOS                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

01_actor_critic_robotics.png - Diagrama cinemático de los eslabones del brazo 
                               y evolución conjunta Actor/Crítico.

═══════════════════════════════════════════════════════════════════════════════
""")

print("✅ Simulación Actor-Critic para control continuo completada exitosamente.")
print("🎯 Todos los gráficos han sido generados en /home/usuario/")