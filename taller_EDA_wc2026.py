"""
Taller EDA - Análisis Exploratorio de Datos
Dataset: ELO Ratings - Mundial 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================
# CARGA DE DATOS
# =============================================================

df = pd.read_csv("elo_ratings_wc2026.csv")

# =============================================================
# EXPLORACIÓN INICIAL
# =============================================================

print("=== PRIMERAS FILAS ===")
print(df.head())

print("\n=== INFORMACIÓN GENERAL ===")
print(df.info())

print("\n=== RESUMEN ESTADÍSTICO ===")
print(df.describe())

print(f"\nFilas: {df.shape[0]}, Columnas: {df.shape[1]}")

print("\n=== VALORES NULOS ===")
print(df.isnull().sum())
# El dataset no tiene valores nulos, lo que indica buena calidad de datos.

print("\n=== DUPLICADOS ===")
print(f"Filas duplicadas: {df.duplicated().sum()}")
# No hay filas completamente duplicadas.

# =============================================================
# LIMPIEZA DE DATOS
# =============================================================

# Convertir snapshot_date a datetime
df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

# Crear columna win_rate: porcentaje de victorias sobre partidos totales
df['win_rate'] = (df['wins'] / df['matches_total'] * 100).round(2)

# Crear columna goal_diff: diferencia de goles (ataque - defensa)
df['goal_diff'] = df['goals_for'] - df['goals_against']

# Filtrar solo el snapshot más reciente (2026) para análisis actuales
# Se eliminan snapshots anteriores para evitar contar países varias veces
df2026 = df[df['year'] == 2026].drop_duplicates(subset='country').copy()

print(f"\nPaíses en el ranking 2026: {len(df2026)}")

# =============================================================
# PREGUNTAS DE ANÁLISIS
# =============================================================

# ------------------------------------------------------------------
# PREGUNTA 1: ¿Cuáles son los 10 países con mayor ELO rating en 2026?
# ------------------------------------------------------------------

top10 = df2026.sort_values('rating', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top10['country'][::-1], top10['rating'][::-1], color='steelblue')
ax.set_xlabel('ELO Rating', fontsize=12)
ax.set_ylabel('País', fontsize=12)
ax.set_title('Top 10 Selecciones por ELO Rating - Mundial 2026', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%d', padding=3, fontsize=10)
ax.set_xlim(1800, 2250)
plt.tight_layout()
plt.savefig('pregunta1_top10_rating.png', dpi=150)
plt.show()

print("\n=== PREGUNTA 1: Top 10 por ELO Rating ===")
print(top10[['country', 'rating', 'rank', 'confederation']])
"""
Interpretación: España lidera el ranking con 2165 puntos, seguida de Argentina (2113) 
y Francia (2081). Europa (UEFA) domina el top 10 con 7 representantes, 
mientras CONMEBOL aporta 2 (Argentina y Brasil).
"""

# ------------------------------------------------------------------
# PREGUNTA 2: ¿Cuál es el ELO rating promedio por confederación?
# ------------------------------------------------------------------

rating_conf = df2026.groupby('confederation')['rating'].mean().sort_values(ascending=False).round(1)

fig, ax = plt.subplots(figsize=(9, 5))
colores = ['#2ecc71' if c in ['UEFA', 'CONMEBOL'] else '#95a5a6' for c in rating_conf.index]
bars = ax.bar(rating_conf.index, rating_conf.values, color=colores, edgecolor='white')
ax.set_xlabel('Confederación', fontsize=12)
ax.set_ylabel('ELO Rating Promedio', fontsize=12)
ax.set_title('ELO Rating Promedio por Confederación - 2026', fontsize=14, fontweight='bold')
ax.bar_label(bars, fmt='%.1f', padding=3, fontsize=10)
ax.set_ylim(1200, 1900)
plt.tight_layout()
plt.savefig('pregunta2_rating_confederacion.png', dpi=150)
plt.show()

print("\n=== PREGUNTA 2: Rating promedio por confederación ===")
print(rating_conf)
"""
Interpretación: UEFA y CONMEBOL son las confederaciones más fuertes con ratings 
promedio superiores a 1700. OFC (Oceanía) tiene el promedio más bajo, 
reflejando la menor competitividad de esa zona.
"""

# ------------------------------------------------------------------
# PREGUNTA 3: ¿Cómo ha evolucionado el ELO rating de Colombia a lo largo del tiempo?
# ------------------------------------------------------------------

colombia = df[df['country'] == 'Colombia'].drop_duplicates(subset='year').sort_values('year')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(colombia['year'], colombia['rating'], color='#f39c12', linewidth=2, marker='o', markersize=3)
ax.fill_between(colombia['year'], colombia['rating'], alpha=0.15, color='#f39c12')
ax.set_xlabel('Año', fontsize=12)
ax.set_ylabel('ELO Rating', fontsize=12)
ax.set_title('Evolución del ELO Rating de Colombia (1901 - 2026)', fontsize=14, fontweight='bold')
ax.axhline(colombia['rating'].mean(), color='red', linestyle='--', linewidth=1, label=f"Promedio: {colombia['rating'].mean():.0f}")
ax.legend(fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('pregunta3_evolucion_colombia.png', dpi=150)
plt.show()

print("\n=== PREGUNTA 3: Evolución ELO Colombia ===")
print(f"Rating actual (2026): {colombia[colombia['year']==2026]['rating'].values[0]}")
print(f"Rating máximo histórico: {colombia['rating'].max()} ({colombia.loc[colombia['rating'].idxmax(), 'year']})")
"""
Interpretación: Colombia ha crecido significativamente desde mediados del siglo XX.
Su peak histórico refleja las generaciones doradas del fútbol colombiano.
El rating actual de 2026 la ubica en una posición competitiva a nivel mundial.
"""

# ------------------------------------------------------------------
# PREGUNTA 4: ¿Hay valores atípicos en el ELO rating actual?
# ------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot
ax1.boxplot(df2026['rating'], patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.6),
            medianprops=dict(color='red', linewidth=2))
ax1.set_ylabel('ELO Rating', fontsize=12)
ax1.set_title('Distribución del ELO Rating - Boxplot', fontsize=13, fontweight='bold')
ax1.set_xticks([])

# Histograma
ax2.hist(df2026['rating'], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(df2026['rating'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f"Media: {df2026['rating'].mean():.0f}")
ax2.axvline(df2026['rating'].median(), color='green', linestyle='--', linewidth=1.5,
            label=f"Mediana: {df2026['rating'].median():.0f}")
ax2.set_xlabel('ELO Rating', fontsize=12)
ax2.set_ylabel('Frecuencia', fontsize=12)
ax2.set_title('Histograma del ELO Rating - 2026', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)

plt.suptitle('Análisis de Outliers en ELO Rating', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('pregunta4_outliers_rating.png', dpi=150)
plt.show()

# Identificar outliers con IQR
Q1 = df2026['rating'].quantile(0.25)
Q3 = df2026['rating'].quantile(0.75)
IQR = Q3 - Q1
outliers = df2026[(df2026['rating'] < Q1 - 1.5*IQR) | (df2026['rating'] > Q3 + 1.5*IQR)]
print("\n=== PREGUNTA 4: Outliers en ELO Rating ===")
print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Países outliers: {len(outliers)}")
print(outliers[['country', 'rating', 'confederation']])
"""
Interpretación: Los países con rating muy alto (España, Argentina, Francia) 
aparecen como outliers superiores, confirmando que son selecciones 
estadísticamente excepcionales respecto al resto del mundo.
"""

# ------------------------------------------------------------------
# PREGUNTA 5: ¿Qué porcentaje de selecciones tiene un win rate superior al promedio?
# ------------------------------------------------------------------

promedio_wr = df2026['win_rate'].mean()
sobre_promedio = df2026[df2026['win_rate'] > promedio_wr]
bajo_promedio = df2026[df2026['win_rate'] <= promedio_wr]
pct_sobre = len(sobre_promedio) / len(df2026) * 100

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    [len(sobre_promedio), len(bajo_promedio)],
    labels=[f'Sobre el promedio\n({len(sobre_promedio)} países)',
            f'Bajo el promedio\n({len(bajo_promedio)} países)'],
    autopct='%1.1f%%',
    colors=['#2ecc71', '#e74c3c'],
    startangle=90,
    explode=(0.05, 0),
    textprops={'fontsize': 11}
)
ax.set_title(f'Selecciones con Win Rate > Promedio ({promedio_wr:.1f}%)\nMundial 2026',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pregunta5_winrate_promedio.png', dpi=150)
plt.show()

print("\n=== PREGUNTA 5: Win Rate vs Promedio ===")
print(f"Win rate promedio mundial: {promedio_wr:.2f}%")
print(f"Países sobre el promedio: {len(sobre_promedio)} ({pct_sobre:.1f}%)")
print("\nTop 5 con mayor win rate:")
print(df2026.nlargest(5, 'win_rate')[['country', 'win_rate', 'wins', 'matches_total']])
"""
Interpretación: Solo el 40% de las selecciones supera el win rate promedio.
Esto es esperado: en fútbol, las selecciones de élite acumulan muchas más 
victorias que el promedio, generando una distribución sesgada hacia la derecha.
"""

print("\n Taller EDA completado. Gráficas guardadas como archivos PNG.")
