#Ejemplo para desplegar en streamlit
# Se despliega así: `streamlit run main.py` dentro de la ruta del archivo sstreamlit


#Este archivo es una prueba para mostrar como se puede montar una pagina web con streamlit
# Streamlit es una libreria de python que permite crear aplicaciones web de manera sencilla y rapida

# Para instalar streamlit, puedes usar el siguiente comando:
# pip install streamlit

# Para ejecutar este archivo, puedes usar el siguiente comando:
# streamlit run <nombredelarchivo>.py

# Importar librerías
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

st.title("📊 Aplicación Interactiva de Gráficos")

st.subheader("Como montar una pagina Streamlit?")

iframe_code = """
<iframe width="100%" height="400" src="https://s3-us-west-2.amazonaws.com/assets.streamlit.io/videos/hero-video.mp4"
title="Streamlit Tutorial" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe>
"""

st.components.v1.html(iframe_code, height=420, scrolling=False)

# *** GRÁFICO DEL APARTDO 3 ***

# Título de sección
st.header("3. Pedidos por ciudad")

# Cargar los datos
olist_order_customer = pd.read_csv('../datasets_limpios/olist_order_customer.csv')
olist_orders_dataset = pd.read_csv('../datasets_limpios/olist_orders_dataset.csv')

# Juntar las dos tablas con un merge
merge_tablas = pd.merge(olist_order_customer, olist_orders_dataset, on="customer_id")

# Convertir fechas a formato datetime
merge_tablas["order_delivered_customer_date"] = pd.to_datetime(merge_tablas["order_delivered_customer_date"])
merge_tablas["order_estimated_delivery_date"] = pd.to_datetime(merge_tablas["order_estimated_delivery_date"])

# Calcular retraso
merge_tablas["dias_retraso"] = (merge_tablas["order_delivered_customer_date"] - merge_tablas["order_estimated_delivery_date"]).dt.days

# Filtrar los pedidos tardíos
pedidos_tarde = merge_tablas[merge_tablas["dias_retraso"] > 0]

# Calcular número de pedidos por ciudad
pedidos_por_ciudad = merge_tablas.groupby("customer_city")["order_delivered_customer_date"].count()

# Crear el dataframe con los datos agregados
merge_tablas = pedidos_tarde.groupby("customer_city").agg(
    num_pedidos_tarde=("dias_retraso", "count"),
    tiempo_medio_retraso=("dias_retraso", "mean")
).reset_index()

# Calcular porcentaje de pedidos tardíos por ciudad
# calcular porcentaje
porcentaje_tarde = []
for ciudad in merge_tablas["customer_city"]:
    total_pedidos = pedidos_por_ciudad.loc[ciudad]
    pedidos_tardios = merge_tablas.loc[merge_tablas["customer_city"] == ciudad, "num_pedidos_tarde"].values[0]
    porcentaje_tarde.append((pedidos_tardios / total_pedidos) * 100)

merge_tablas["porcentaje_tarde"] = porcentaje_tarde

# Ordenar y seleccionar los más representativos
df_resultado_num_pedidos_tarde = merge_tablas.sort_values(by="num_pedidos_tarde", ascending=False)
df_top_15 = df_resultado_num_pedidos_tarde.head(15)

# Ordenar ciudades por tiempo medio de retraso (de mayor a menor)
#df_resultado_num_pedidos_tarde = merge_tablas.sort_values(by="tiempo_medio_retraso", ascending=False)

# Crear figura con Seaborn - Boxplot para visualizar distribución de retrasos
fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df_resultado_num_pedidos_tarde, 
    x="customer_city", 
    y="num_pedidos_tarde", 
    palette="Blues",
    width=0.6  # Ajusta el ancho de las cajas para mayor claridad
)

sns.stripplot(  # Agregar puntos sobre el boxplot para mejor visualización
    data=df_resultado_num_pedidos_tarde, 
    x="customer_city", 
    y="num_pedidos_tarde", 
    color="black", 
    size=6,  # Hace los puntos más grandes
    alpha=0.7,  # Transparencia para mejor visibilidad
    jitter=True  # Evita superposición exacta
)

# Configurar etiquetas y título
ax.set_xlabel("Ciudades")
ax.set_ylabel("Número de pedidos tardíos")
ax.set_title("Distribución de pedidos que llegan tarde por ciudad")
ax.set_xticklabels([])

# Mostrar gráfico en Streamlit
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df_top_15, x="customer_city", y="num_pedidos_tarde", palette="Blues_r")
ax.set_xticklabels(df_top_15["customer_city"], rotation=45, ha="right")
ax.set_xlabel("Ciudad")
ax.set_ylabel("Pedidos tardíos")
ax.set_title("Top 15 ciudades con más pedidos tardíos")
st.pyplot(fig)


# Ordenar datos
df_resultado_porcentaje_tarde = merge_tablas.sort_values(by="porcentaje_tarde", ascending=False)

# Establecer fondo completamente blanco
sns.set_style("white")

# Crear figura
fig, ax = plt.subplots(figsize=(12, 6))

# Crear gráfico de dispersión con puntos negros
sns.scatterplot(data=df_resultado_porcentaje_tarde, x="customer_city", y="porcentaje_tarde", color="black", alpha=0.8, ax=ax)

# Etiquetas
ax.set_xlabel("Ciudad")
ax.set_ylabel("Porcentaje de pedidos tardíos")
ax.set_title("Porcentaje de pedidos tardíos por ciudad")

# Ocultar etiquetas del eje X si hay demasiadas ciudades
ax.set_xticklabels([])

# Mostrar gráfico en Streamlit
st.pyplot(fig)






# Filtrar y ordenar los datos
df_resultado_tiempo_medio_retraso = merge_tablas.sort_values(by="tiempo_medio_retraso", ascending=False)

# Configurar el estilo de Seaborn
sns.set_style("whitegrid")

# Primer gráfico: Retraso promedio por ciudad

import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo sin fondo gris
sns.set_style("white")

# Crear la figura y el eje
fig, ax = plt.subplots(figsize=(6, 4))

# Crear gráfico de línea con puntos negros
sns.lineplot(
    data=df_resultado_tiempo_medio_retraso, 
    x="customer_city", 
    y="tiempo_medio_retraso", 
    marker="o", 
    color="black",  # Color negro para todos los elementos
    linewidth=2,
    markeredgecolor="black",  # Bordes negros en los puntos
    markeredgewidth=1  # Grosor del borde
)

# Ajustar etiquetas y título
ax.set_xticklabels([])
ax.set_xlabel("Ciudades")
ax.set_ylabel("Días de retraso promedio")
ax.set_title("Retraso promedio en la entrega por ciudad")

# Mostrar la gráfica en Streamlit
st.pyplot(fig)



# Filtrar las 15 ciudades con más retraso
df_top_15 = df_resultado_tiempo_medio_retraso.head(15)


fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(data=df_top_15, x="customer_city", y="tiempo_medio_retraso", marker="o", color="red", linewidth=2)
ax.set_xticklabels(df_top_15["customer_city"], rotation=45, ha="right")
ax.set_xlabel("Ciudad")
ax.set_ylabel("Días de retraso promedio")
ax.set_title("Retraso promedio en la entrega por ciudad (Top 15)")
st.pyplot(fig)





#***EJERCICIO 4****

# Cargar los datos
olist_order_reviews_dataset_clean = pd.read_csv('../datasets_limpios/olist_order_reviews_dataset_clean.csv')
olist_orders_dataset = pd.read_csv('../datasets_limpios/olist_orders_dataset.csv')
olist_order_customer = pd.read_csv('../datasets_limpios/olist_order_customer.csv')

# Merge de los 3 datasets
merged_dataset = olist_order_reviews_dataset_clean.merge(olist_orders_dataset, on='order_id').merge(olist_order_customer, on='customer_id')

# Unir los datos
merge_tablas = pd.merge(olist_order_customer, olist_orders_dataset, on="customer_id")

# Convertir fechas a formato datetime
merge_tablas["order_delivered_customer_date"] = pd.to_datetime(merge_tablas["order_delivered_customer_date"])
merge_tablas["order_estimated_delivery_date"] = pd.to_datetime(merge_tablas["order_estimated_delivery_date"])

# Calcular retraso
merge_tablas["dias_retraso"] = (merge_tablas["order_delivered_customer_date"] - merge_tablas["order_estimated_delivery_date"]).dt.days

# Filtrar pedidos tardíos
pedidos_tarde = merge_tablas[merge_tablas["dias_retraso"] > 0]

# Filtrar reseñas de pedidos entregados a tiempo
merged_dataset_filtrado = merged_dataset[~merged_dataset['customer_id'].isin(pedidos_tarde['customer_id'])]

# Ordenar reviews por estado
reviews_estado = merged_dataset_filtrado.groupby("customer_state")["review_score"].count().reset_index()
reviews_estado = reviews_estado.sort_values(by="review_score", ascending=False)

# Configurar estilo de Seaborn
sns.set_style("whitegrid")

# Gráfico de barras: cantidad de reseñas por estado
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=reviews_estado, x="customer_state", y="review_score", palette="Blues_r")
ax.set_xlabel("Estado")
ax.set_ylabel("Cantidad de reseñas")
ax.set_title("Cantidad de reseñas por estado")
plt.xticks(rotation=90)
st.pyplot(fig)

# Calcular la puntuación media de las reviews por estado
reviews_estado_media_score = merged_dataset_filtrado.groupby("customer_state")["review_score"].mean().reset_index()
reviews_estado_media_score = reviews_estado_media_score.sort_values(by="review_score", ascending=False)

# Gráfico de pastel: puntuación media de las reviews por estado
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(reviews_estado_media_score["review_score"], labels=reviews_estado_media_score["customer_state"], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
ax.set_title("Puntuación media de las reviews por estado")
st.pyplot(fig)