"""
Taller - Ventas Sucias
"""

import pandas as pd
import numpy as np

# =============================================================
# PARTE 1 — EXPLORACIÓN DE DATOS
# =============================================================

# Cargamos el dataset
df = pd.read_csv("ventas_sucias_5000.csv")

# Primeras filas del dataset
print("=== PRIMERAS FILAS ===")
print(df.head())

# Información general: tipos de datos y valores nulos
print("\n=== INFORMACIÓN GENERAL ===")
print(df.info())

# Resumen estadístico de columnas numéricas
print("\n=== RESUMEN ESTADÍSTICO ===")
print(df.describe())

# Dimensiones del dataset
print(f"\nFilas: {df.shape[0]}, Columnas: {df.shape[1]}")

"""
RESPUESTAS PARTE 1:
- El dataset tiene 5000 filas y 7 columnas.
- Tipos de datos: cliente (texto), producto (texto), precio (float),
  cantidad (float), pais (texto), metodo_pago (texto), fecha (texto/object).
- Problemas encontrados:
    * precio y cantidad tienen 50 valores nulos cada uno.
    * pais tiene inconsistencias: 'peru', 'Perú', 'Colombia', 'COL', 'col', 'chile', 'Chile'.
    * metodo_pago tiene inconsistencias: 'Efectivo', 'Tarjeta', 'TRANSFERENCIA', 'transferencia'.
    * precio tiene un outlier extremo: valor máximo de 999999.
    * fecha está como texto (object), debe convertirse a datetime.
    * cantidad debería ser entero, no float.
"""

# =============================================================
# PARTE 2 — LIMPIEZA DE DATOS
# =============================================================

# --- Valores nulos ---
# Se eliminan filas donde precio o cantidad son nulos,
# ya que sin estos valores no podemos calcular ventas.
df = df.dropna(subset=['precio', 'cantidad'])
print(f"\nRegistros tras eliminar nulos: {len(df)}")

# --- Tipos de datos ---
# cantidad tiene valores de texto como 'three' → los convertimos a NaN y eliminamos
df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce')
df = df.dropna(subset=['cantidad'])
df['cantidad'] = df['cantidad'].astype(int)

# fecha debe ser datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# --- Estandarizar columna 'pais' ---
# Convertimos a minúsculas, eliminamos espacios
# y unificamos variantes del mismo país
df['pais'] = df['pais'].str.strip().str.lower()
df['pais'] = df['pais'].replace({
    'perú': 'peru',
    'col': 'colombia',
    'chile': 'chile'
})
# Capitalizamos para presentación final
df['pais'] = df['pais'].str.capitalize()

print("\nPaíses únicos tras limpieza:", df['pais'].unique())

# --- Estandarizar columna 'metodo_pago' ---
df['metodo_pago'] = df['metodo_pago'].str.strip().str.capitalize()
print("Métodos de pago únicos:", df['metodo_pago'].unique())

# --- Outliers en precio ---
# El precio máximo es 999999, lo cual es claramente erróneo.
# Eliminamos registros con precio superior a 5000 (valor atípico extremo).
antes = len(df)
df = df[df['precio'] <= 5000]
print(f"\nRegistros eliminados por outliers en precio: {antes - len(df)}")
print(f"Registros finales: {len(df)}")

"""
RESPUESTAS PARTE 2:
- Problemas encontrados:
    * 50 nulos en precio y 50 en cantidad → se eliminaron esas filas (100 total).
    * pais tenía 7 variantes para 3 países → unificado con replace y capitalize.
    * metodo_pago tenía mayúsculas inconsistentes → unificado con capitalize.
    * fecha era texto → convertida a datetime con pd.to_datetime().
    * cantidad era float → convertida a int.
    * precio tenía valores de 999999 (outliers extremos) → eliminados (> 5000).
- Se eliminaron filas con nulos porque sin precio/cantidad no hay venta válida.
- Se eliminaron outliers extremos porque distorsionan el análisis estadístico.
"""

# =============================================================
# PARTE 3 — ANÁLISIS CON PANDAS
# =============================================================

# Nueva columna total
df['total'] = df['precio'] * df['cantidad']

# Estadísticas generales
print("\n=== ANÁLISIS DE VENTAS ===")
print(f"Total vendido:     ${df['total'].sum():,.2f}")
print(f"Promedio de ventas:${df['total'].mean():,.2f}")
print(f"Venta máxima:      ${df['total'].max():,.2f}")
print(f"Venta mínima:      ${df['total'].min():,.2f}")

# Top 5 productos con mayor valor vendido
print("\n=== TOP 5 PRODUCTOS MÁS VENDIDOS ===")
top5 = df.groupby('producto')['total'].sum().sort_values(ascending=False).head(5)
print(top5)

# País con más ventas
print("\n=== PAÍS CON MÁS VENTAS ===")
pais_top = df.groupby('pais')['total'].sum().sort_values(ascending=False)
print(pais_top)
print(f"\nEl país con más ventas es: {pais_top.idxmax()}")

# =============================================================
# PARTE 4 — INTRODUCCIÓN A NUMPY
# =============================================================

# Convertir columnas numéricas a array de NumPy
data = df[['precio', 'cantidad']].to_numpy()

# Separar columnas
precios    = data[:, 0]   # Todas las filas, columna 0 (precio)
cantidades = data[:, 1]   # Todas las filas, columna 1 (cantidad)

# Calcular ventas totales con vectorización
totales = precios * cantidades

print("\n=== ARRAY NUMPY (primeras 5 filas) ===")
print(data[:5])

# =============================================================
# PARTE 5 — ANÁLISIS CON NUMPY
# =============================================================

print("\n=== ANÁLISIS CON NUMPY ===")
print(f"Suma total de ventas:          ${np.sum(totales):,.2f}")
print(f"Promedio de ventas:            ${np.mean(totales):,.2f}")
print(f"Venta máxima:                  ${np.max(totales):,.2f}")
print(f"Ventas superiores a $1000:     {(totales > 1000).sum()}")

"""
RESPUESTAS PARTE 5:

¿Qué ventajas observas al usar NumPy?
- NumPy es más rápido que Python puro porque opera directamente sobre 
  bloques de memoria, sin bucles interpretados. Es ideal para cálculos
  sobre grandes volúmenes de datos numéricos.

¿Qué significa "vectorización"?
- Significa aplicar una operación a todos los elementos de un array de una 
  sola vez, sin usar un bucle for. Por ejemplo, precios * cantidades multiplica
  cada par de valores simultáneamente, de forma más rápida y con menos código.

¿Qué hace la expresión data[:, 0]?
- Selecciona TODAS las filas (:) de la columna 0 del array bidimensional.
  Es decir, extrae todos los precios como un array unidimensional.
"""

# =============================================================
# PARTE 6 — INTERPRETACIÓN DE RESULTADOS
# =============================================================

"""
RESPUESTAS PARTE 6:

¿Los resultados obtenidos tienen sentido?
- Sí, tras la limpieza los datos son coherentes. Los precios van de $10 a $5000,
  las cantidades son enteros positivos y los países están unificados.

¿Detectaste valores sospechosos?
- Sí. El precio máximo original era 999999, claramente un error de ingreso de datos.
  También había precios muy bajos (ej: $55 para un Laptop) que podrían ser errores,
  aunque en este taller se conservaron salvo los extremos obvios.

¿El promedio representa correctamente los datos?
- Solo después de eliminar outliers. Con el valor 999999 incluido, el promedio
  era de $5053, completamente distorsionado. Tras la limpieza, el promedio 
  refleja mejor el comportamiento real de las ventas.

¿Qué decisiones tomarías si esta fuera información real de negocio?
- Investigar el origen de los valores nulos (¿error del sistema? ¿datos faltantes?).
- Validar con el equipo comercial los precios atípicos antes de eliminarlos.
- Crear alertas para detectar precios fuera de un rango aceptable en tiempo real.
- Analizar por período (mes/trimestre) para detectar tendencias de ventas.
- Segmentar por país y producto para identificar oportunidades de mercado.
"""
