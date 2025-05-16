import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Importo las funciones propias de pipelines
from pipelines import *


st.title("Proyecto 01 - Olist Data")

st.markdown(
    """
## 👥 Integrantes del grupo

- **Javier Sánchez de las Heras** - [@javierstemdo](https://github.com/javierstemdo)  
- **Mariam Guadalupe Núñez** - [@mariamgn-stemdo](https://github.com/mariamgn-stemdo)  
- **Sergio Fernández Nevado** - [@srgfdz](https://github.com/srgfdz)
"""
)

# *** GRÁFICO DEL APARTDO 1 ***
st.header("1. Número de Clientes por Estado")

# Cargar datos para el punto 1 (clientes y pedidos)
df_customers = pd.read_csv("./datasets_limpios/olist_order_customer.csv")
df_orders = pd.read_csv("./datasets_limpios/olist_orders_dataset.csv")

# Hago uun merge outer, pero en este caso los registros de ambos dataset coinciden exactamente:
# no hay clientes que no tengan ningún pedido y cada cliente tiene un único pedido, por lo que no necesitamos transformar el df mergeado
df_customers_orders = df_customers.merge(df_orders, on="customer_id", how="outer")
# print(df_customers_orders)

# Convierto las fechas fechas con el pipeline personalizado
datetime_fields = {
    "order_purchase_timestamp": "%Y-%m-%d %H:%M:%S",
    "order_approved_at": "%Y-%m-%d %H:%M:%S",
    "order_delivered_carrier_date": "%Y-%m-%d %H:%M:%S",
    "order_delivered_customer_date": "%Y-%m-%d %H:%M:%S",
    "order_estimated_delivery_date": "%Y-%m-%d",
}

df_customers_orders = clean_datetime_columns_pandas(
    df_customers_orders, datetime_fields
)
# df_customers_orders['order_purchase_timestamp'] = pd.to_datetime(df_customers_orders['order_purchase_timestamp'])

# Pongo estos inputs en la misma línea
col1, col2 = st.columns(2)

# Input con el número de estados a mostrar
with col1:
    top_n = st.selectbox("¿Nº de Estados a mostrar?", options=[3, 5, 10, 20], index=1)

# Input para invertir el orden (añado estilos para centrarl overticalmente)
with col2:
    st.markdown("<div style='height: 2.5em;'></div>", unsafe_allow_html=True)
    orden_invertido = st.checkbox("Estados con menos clientes")


# Filtro de fechas
# fecha_min = df_customers_orders['order_purchase_timestamp'].min()
# fecha_max = df_customers_orders['order_purchase_timestamp'].max()

# fecha_inicio, fecha_fin = st.date_input(
#     "Rango de fechas",
#     value=(fecha_min, fecha_max),
#     min_value=fecha_min,
#     max_value=fecha_max
# )

fecha_min = df_customers_orders["order_purchase_timestamp"].min().to_pydatetime()
fecha_max = df_customers_orders["order_purchase_timestamp"].max().to_pydatetime()


fecha_inicio, fecha_fin = st.slider(
    "Rango de fechas de compra",
    min_value=fecha_min,
    max_value=fecha_max,
    value=(fecha_min, fecha_max),
    format="DD/MM/YYYY",
)


# Aplicar filtro
df_filtrado = df_customers_orders[
    (df_customers_orders["order_purchase_timestamp"] >= pd.to_datetime(fecha_inicio))
    & (df_customers_orders["order_purchase_timestamp"] <= pd.to_datetime(fecha_fin))
]


# Agrupar por estado y cliente y mostrar solo los que indica el filtro desplegable de Nº y según el orden seleccionado
clientes_por_estado = (
    df_filtrado.groupby("customer_state")["customer_id"]
    .nunique()
    .sort_values(ascending=orden_invertido)
    .head(top_n)
    .reset_index()
    .rename(columns={"customer_id": "count_customers"})
)


# Comprobar si hay datos para los filtros seleccionados
if clientes_por_estado.empty:
    st.markdown(
        """
        <div style="padding: 20px; margin: 20px; border: 2px solid white; border-radius: 10px; background-color: #f8d7da; color: #721c24; text-align: center;">
            <strong>*** No hay registros para los filtros aplicados, ajusta los filtros y prueba nuevamente ***</strong>
        </div>
    """,
        unsafe_allow_html=True,
    )
else:
    # Convertir los nombres de los estados a mayúsculas
    clientes_por_estado["customer_state"] = clientes_por_estado[
        "customer_state"
    ].str.upper()
    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=clientes_por_estado,
        x="customer_state",
        y="count_customers",
        palette="Reds_r",
        ax=ax,
    )
    ax.set_title(
        f"{'Estados con menos' if orden_invertido else 'Estados con más clientes'}"
    )
    ax.set_xlabel("Estado")
    ax.set_ylabel("Nº clientes")

    st.pyplot(fig)


# *** GRÁFICO DEL APARTDO 2 ***

# Ampliar el ancho de la página
# st.set_page_config(layout="wide")

# Título de sección
st.header("2. Estados con mayor porcentaje de pedidos")

# Relacionar los datasets por 'customer_id'
df_merged = pd.merge(df_customers, df_orders, on="customer_id", how="inner")

# Asegurar mayúsculas en estados y formato título en ciudades
df_merged["customer_state"] = df_merged["customer_state"].astype(str).str.upper()
df_merged["customer_city"] = df_merged["customer_city"].astype(str).str.title()

# Calcular número de clientes únicos por ciudad
clientes_por_ciudad = (
    df_merged.groupby(["customer_state", "customer_city"])["customer_unique_id"]
    .nunique()
    .reset_index()
)
clientes_por_ciudad.columns = ["Estado", "Ciudad", "Número de clientes"]

# Calcular número de pedidos por ciudad (conteo de order_id por ciudad)
pedidos_por_ciudad = (
    df_merged.groupby(["customer_state", "customer_city"])["order_id"]
    .count()
    .reset_index()
)
pedidos_por_ciudad.columns = ["Estado", "Ciudad", "Número de pedidos"]

# Combinar ambas tablas
tabla = pedidos_por_ciudad.merge(clientes_por_ciudad, on=["Estado", "Ciudad"])

# Calcular ratio como número entero
tabla["Ratio de Pedidos por Cliente"] = (
    (tabla["Número de pedidos"] / tabla["Número de clientes"]).round(0).astype(int)
)

# Select de estado
estados_unicos = sorted(tabla["Estado"].unique())
estado_seleccionado = st.selectbox(
    "Selecciona un estado para ver sus ciudades", estados_unicos
)

# Filtrar por estado y preparar Top 10
datos_estado = tabla[tabla["Estado"] == estado_seleccionado].copy()
datos_estado = datos_estado.sort_values(by="Número de pedidos", ascending=False)

top10_estados = datos_estado.head(10).copy()
total_top10 = top10_estados["Número de pedidos"].sum()

# Calcular porcentaje dentro del top 10
top10_estados["Porcentaje"] = (
    top10_estados["Número de pedidos"] / total_top10 * 100
).round(2)

# Gráfico circular
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.PuBu(np.linspace(0.3, 1, len(top10_estados)))

ax.pie(
    top10_estados["Porcentaje"],
    labels=top10_estados["Ciudad"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    textprops={"fontsize": 9},
    wedgeprops=dict(edgecolor="white"),
    labeldistance=1.05,
)

ax.set_title(f"Top 10 ciudades por pedidos en {estado_seleccionado}", fontsize=14)
ax.axis("equal")

# Mostrar gráfico y tabla
col1, col2 = st.columns([1.3, 1.7])

with col1:
    st.pyplot(fig)

with col2:
    st.subheader(f"Detalle de ciudades en {estado_seleccionado}")
    tabla_filtrada = top10_estados[
        [
            "Ciudad",
            "Número de pedidos",
            "Número de clientes",
            "Porcentaje",
            "Ratio de Pedidos por Cliente",
        ]
    ].reset_index(drop=True)

    st.dataframe(tabla_filtrada, use_container_width=True)


# *** GRÁFICO DEL APARTDO 3 ***

# Título de sección
st.header("3. Pedidos por ciudad")

# Cargar los datos
# df_customers = pd.read_csv("./datasets_limpios/olist_order_customer.csv")
# df_orders = pd.read_csv("./datasets_limpios/olist_orders_dataset.csv")

# Juntar las dos tablas con un merge
merge_tablas = pd.merge(df_customers, df_orders, on="customer_id")

# Convertir fechas a formato datetime
merge_tablas["order_delivered_customer_date"] = pd.to_datetime(
    merge_tablas["order_delivered_customer_date"]
)
merge_tablas["order_estimated_delivery_date"] = pd.to_datetime(
    merge_tablas["order_estimated_delivery_date"]
)

# Calcular retraso
merge_tablas["dias_retraso"] = (
    merge_tablas["order_delivered_customer_date"]
    - merge_tablas["order_estimated_delivery_date"]
).dt.days

# Filtrar los pedidos tardíos
pedidos_tarde = merge_tablas[merge_tablas["dias_retraso"] > 0]

# Calcular número de pedidos por ciudad
pedidos_por_ciudad = merge_tablas.groupby("customer_city")[
    "order_delivered_customer_date"
].count()

# Crear el dataframe con los datos agregados
merge_tablas = (
    pedidos_tarde.groupby("customer_city")
    .agg(
        num_pedidos_tarde=("dias_retraso", "count"),
        tiempo_medio_retraso=("dias_retraso", "mean"),
    )
    .reset_index()
)

# calcular porcentaje pedidos tardíos por ciudad
porcentaje_tarde = []
for ciudad in merge_tablas["customer_city"]:
    total_pedidos = pedidos_por_ciudad.loc[ciudad]
    pedidos_tardios = merge_tablas.loc[
        merge_tablas["customer_city"] == ciudad, "num_pedidos_tarde"
    ].values[0]
    porcentaje_tarde.append((pedidos_tardios / total_pedidos) * 100)

merge_tablas["porcentaje_tarde"] = porcentaje_tarde

# Obtener el número de pedidos tarde ordenados
df_resultado_num_pedidos_tarde = merge_tablas.sort_values(
    by="num_pedidos_tarde", ascending=False
)
df_top_15 = df_resultado_num_pedidos_tarde.head(15)
"""
# Gráfico num pedidos tarde por ciudad
fig, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df_resultado_num_pedidos_tarde,
    x="customer_city",
    y="num_pedidos_tarde",
    palette="Blues",
    width=0.6,
)

# Configurar representación puntos del boxplot
sns.stripplot(
    data=df_resultado_num_pedidos_tarde,
    x="customer_city",
    y="num_pedidos_tarde",
    color="black",
    size=6,
    alpha=0.7,
    jitter=True,
)

# Etiquetas y título
ax.set_xlabel("Ciudades")
ax.set_ylabel("Número de pedidos tardíos")
ax.set_title("Distribución de pedidos que llegan tarde por ciudad")
ax.set_xticklabels([])

# Mostrar gráfico
st.pyplot(fig)
"""
# Gráfico para ciudades con más pedidos tardios
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=df_top_15, x="customer_city", y="num_pedidos_tarde", palette="Blues_r")
ax.set_xticklabels(df_top_15["customer_city"], rotation=45, ha="right")
ax.set_xlabel("Ciudad")
ax.set_ylabel("Pedidos tardíos")
ax.set_title("Top 15 ciudades con más pedidos tardíos")
st.pyplot(fig)

# Ordenar datos
df_resultado_porcentaje_tarde = merge_tablas.sort_values(
    by="porcentaje_tarde", ascending=False
)

# Establecer fondo blanco
sns.set_style("white")

# Crear figura
fig, ax = plt.subplots(figsize=(12, 6))

# Configurar puntos del gráfico
sns.scatterplot(
    data=df_resultado_porcentaje_tarde,
    x="customer_city",
    y="porcentaje_tarde",
    color="black",
    alpha=0.8,
    ax=ax,
)

# Etiquetas
ax.set_xlabel("Ciudad")
ax.set_ylabel("Porcentaje de pedidos tardíos")
ax.set_title("Porcentaje de pedidos tardíos por ciudad")

# Ocultar etiquetas del eje X
ax.set_xticklabels([])

# Mostrar gráfico
st.pyplot(fig)

# Mostrar tabla con los datos más detallados
st.subheader(f"Número y porcentaje de pedidos tarde por ciudad")
tabla_filtrada = (
    merge_tablas[
        [
            "customer_city",
            "porcentaje_tarde",
            "num_pedidos_tarde",
        ]
    ]
    .rename(
        columns={
            "customer_city": "Ciudad",
            "porcentaje_tarde": "Porcentaje de pedidos tardíos",
            "num_pedidos_tarde": "Número de pedidos tardíos",
        }
    )
    .reset_index(drop=True)
)

# Mostrar tabla
st.dataframe(tabla_filtrada, use_container_width=True)

# Ordenar los datos tiempo_medio_retraso
df_resultado_tiempo_medio_retraso = merge_tablas.sort_values(
    by="tiempo_medio_retraso", ascending=False
)

# Estilo color blanco para el fondo
sns.set_style("white")

# Crear figura
fig, ax = plt.subplots(figsize=(6, 4))

# Crear gráfico
sns.lineplot(
    data=df_resultado_tiempo_medio_retraso,
    x="customer_city",
    y="tiempo_medio_retraso",
    marker="o",
    color="black",
    linewidth=2,
    markeredgecolor="black",
    markeredgewidth=1,
)

# Etiquetas y título
ax.set_xticklabels([])
ax.set_xlabel("Ciudades")
ax.set_ylabel("Días de retraso promedio")
ax.set_title("Retraso promedio en la entrega por ciudad")

# Mostrar la gráfica
st.pyplot(fig)

# Obtener las 15 ciudades con más retraso
df_top_15 = df_resultado_tiempo_medio_retraso.head(15)

# Crear figura
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=df_top_15,
    x="customer_city",
    y="tiempo_medio_retraso",
    marker="o",
    color="red",
    linewidth=2,
)

ax.set_xticklabels(df_top_15["customer_city"], rotation=45, ha="right")
ax.set_xlabel("Ciudad")
ax.set_ylabel("Días de retraso promedio")
ax.set_title("Retraso promedio en la entrega por ciudad (Top 15)")

# Mostrar gráfica
st.pyplot(fig)

# *** GRÁFICO DEL APARTDO 4 ***

# Título de sección
st.header("4. Cantidad de reseñas y score medio.")

# Cargar los datos
olist_order_reviews_dataset_clean = pd.read_csv(
    "./datasets_limpios/olist_order_reviews_dataset_clean.csv"
)
# olist_orders_dataset = pd.read_csv("./datasets_limpios/olist_orders_dataset.csv")
# olist_order_customer = pd.read_csv("./datasets_limpios/olist_order_customer.csv")

# Merge de los 3 datasets
merged_dataset = olist_order_reviews_dataset_clean.merge(
    df_orders, on="order_id"
).merge(df_customers, on="customer_id")

# Filtrar reseñas de pedidos entregados a tiempo
merged_dataset_filtrado = merged_dataset[
    ~merged_dataset["customer_id"].isin(pedidos_tarde["customer_id"])
]

# Ordenar reviews por estado
reviews_estado = (
    merged_dataset_filtrado.groupby("customer_state")["review_score"]
    .count()
    .reset_index()
)
reviews_estado = reviews_estado.sort_values(by="review_score", ascending=False)

# Calcular la puntuación media de las reviews por estado
reviews_estado_media_score = (
    merged_dataset_filtrado.groupby("customer_state")["review_score"]
    .mean()
    .reset_index()
)
reviews_estado_media_score = reviews_estado_media_score.sort_values(
    by="review_score", ascending=False
)

# Crear figura
fig, ax1 = plt.subplots(figsize=(12, 6))

# Gráfico de barras
sns.barplot(
    data=reviews_estado, x="customer_state", y="review_score", palette="Blues_r", ax=ax1
)
ax1.set_xlabel("Estado")
ax1.set_ylabel("Cantidad de reseñas")
ax1.set_title("Cantidad de reseñas y puntuación media por estado")
ax1.tick_params(axis="x", rotation=90)

# Añadir un segundo eje para la puntuación media
ax2 = ax1.twinx()
sns.lineplot(
    data=reviews_estado_media_score,
    x="customer_state",
    y="review_score",
    color="red",
    marker="o",
    ax=ax2,
)
ax2.set_ylabel("Puntuación media")

# Mostrar gráfico
st.pyplot(fig)
