import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


#Importo las funciones propias de pipelines
from pipelines import *
st.set_page_config(layout="wide")




st.title("Proyecto 01 - Olist Dataaaaaa")

st.markdown("""
## 👥 Integrantes del grupo

- **Javier Sánchez de las Heras** - [@javierstemdo](https://github.com/javierstemdo)  
- **Mariam Guadalupe Núñez** - [@mariamgn-stemdo](https://github.com/mariamgn-stemdo)  
- **Sergio Fernández Nevado** - [@srgfdz](https://github.com/srgfdz)
""")



# Cargar datos 
df_customers = pd.read_csv('./../datasets_limpios/olist_order_customer.csv')
df_orders = pd.read_csv('./../datasets_limpios/olist_orders_dataset.csv')
df_sellers = pd.read_csv('./../datasets_limpios/olist_sellers_dataset.csv')
df_reviews = pd.read_csv('./../datasets_limpios/olist_order_reviews_dataset_clean.csv')
df_order_items = pd.read_csv('./../datasets_limpios/olist_order_items_dataset.csv')
df_products = pd.read_csv('./../datasets_limpios/olist_products_dataset.csv')

# *** GRÁFICO DEL APARTDO 1 ***
st.header("1. Número de Clientes por Estado")





# Convierto los nombres de los estados a mayúsculas, ya que en la normalización los he pasado a minúsculas
df_customers['customer_state'] = df_customers['customer_state'].str.upper()
df_customers['customer_city'] = df_customers['customer_city'].str.title()
df_sellers['seller_state'] = df_sellers['seller_state'].str.upper()
df_sellers['seller_city'] = df_sellers['seller_city'].str.title()

df_products['product_category_name'] = df_products['product_category_name'].str.replace('_', ' ').str.title()



# no hay clientes que no tengan ningún pedido, por lo que puedo usar inner y no perder registro de ningún cliente
df_customers_orders = df_customers.merge(df_orders, on='customer_id', how='inner')
# print(df_customers_orders)

# Convierto las fechas fechas con el pipeline personalizado
datetime_fields = {
        "order_purchase_timestamp": "%Y-%m-%d %H:%M:%S",
        "order_approved_at": "%Y-%m-%d %H:%M:%S",
        "order_delivered_carrier_date": "%Y-%m-%d %H:%M:%S",
        "order_delivered_customer_date": "%Y-%m-%d %H:%M:%S",
        "order_estimated_delivery_date": "%Y-%m-%d",
    }

df_customers_orders = clean_datetime_columns_pandas(df_customers_orders, datetime_fields)
# df_customers_orders['order_purchase_timestamp'] = pd.to_datetime(df_customers_orders['order_purchase_timestamp'])

#Pongo estos inputs en la misma línea
col1, col2 = st.columns(2)

# Input con el número de estados a mostrar
with col1:
        orden_opcion = st.selectbox(
        "Información a mostrar",
        options=[
            "Estados con mayor número de clientes",
            "Estados con menor número de clientes",
            "Estados con mayor número de vendedores",
            "Estados con menor número de vendedores"
        ],
        index=0
    )


# Input para invertir el orden (añado estilos para centrarl overticalmente)
with col2:
    top_n = st.selectbox("Nº de Estados a mostrar", options=[3, 5, 10, 20], index=1)


# Filtrado por fecha (solo aplica a clientes)
fecha_min = df_customers_orders['order_purchase_timestamp'].min().to_pydatetime()
fecha_max = df_customers_orders['order_purchase_timestamp'].max().to_pydatetime()

fecha_inicio, fecha_fin = st.slider(
    "Rango de fechas de compra",
    min_value=fecha_min,
    max_value=fecha_max,
    value=(fecha_min, fecha_max),
    format="DD/MM/YYYY"
)

# Parámetros de agrupación según lo seleccionado por el usuario
tipo = "clientes" if "cliente" in orden_opcion else "vendedores"
orden_invertido = "menor" in orden_opcion

if tipo == "clientes":
    df_filtrado = df_customers_orders[
        (df_customers_orders['order_purchase_timestamp'] >= pd.to_datetime(fecha_inicio)) &
        (df_customers_orders['order_purchase_timestamp'] <= pd.to_datetime(fecha_fin))
    ]

    agrupado = (
        df_filtrado.groupby('customer_state')['customer_unique_id']
        .nunique()
        .sort_values(ascending=orden_invertido)
        .head(top_n)
        .reset_index()
        .rename(columns={'customer_unique_id': 'count'})
    )

    total = df_filtrado['customer_unique_id'].nunique()
    agrupado['percentage'] = (agrupado['count'] / total * 100).round(2)

    estados_mostrados = agrupado['customer_state'].tolist()

    df_tabla = df_filtrado[df_filtrado['customer_state'].isin(estados_mostrados)].copy()
    clientes_por_ciudad = (
        df_tabla.groupby(['customer_state', 'customer_city'])['customer_unique_id']
        .nunique()
        .reset_index()
        .rename(columns={'customer_state': 'Estado', 'customer_city': 'Ciudad', 'customer_unique_id': 'Nº de Clientes'})
    )
    clientes_por_ciudad['Estado'] = pd.Categorical(clientes_por_ciudad['Estado'], categories=estados_mostrados, ordered=True)
    clientes_por_ciudad.sort_values(by=['Estado', 'Nº de Clientes'], ascending=[True, False], inplace=True)

else:
    agrupado = (
        df_sellers.groupby('seller_state')['seller_id']
        .nunique()
        .sort_values(ascending=orden_invertido)
        .head(top_n)
        .reset_index()
        .rename(columns={'seller_id': 'count'})
    )

    total = df_sellers['seller_id'].nunique()
    agrupado['percentage'] = (agrupado['count'] / total * 100).round(2)

    estados_mostrados = agrupado['seller_state'].tolist()
    df_tabla = df_sellers[df_sellers['seller_state'].isin(estados_mostrados)].copy()

    clientes_por_ciudad = (
        df_tabla.groupby(['seller_state', 'seller_city'])['seller_id']
        .nunique()
        .reset_index()
        .rename(columns={'seller_state': 'Estado', 'seller_city': 'Ciudad', 'seller_id': 'Nº de Vendedores'})
    )
    clientes_por_ciudad['Estado'] = pd.Categorical(clientes_por_ciudad['Estado'], categories=estados_mostrados, ordered=True)
    clientes_por_ciudad.sort_values(by=['Estado', 'Nº de Vendedores'], ascending=[True, False], inplace=True)

# Gráfico y tabla
if agrupado.empty:
    st.markdown("""
        <div style="padding: 20px; margin: 20px; border: 2px solid white; border-radius: 10px; background-color: #f8d7da; color: #721c24; text-align: center;">
            <strong>*** No hay registros para los filtros aplicados, ajusta los filtros y prueba nuevamente ***</strong>
        </div>
    """, unsafe_allow_html=True)
else:
    col_grafico, col_tabla = st.columns([2, 3])

    with col_grafico:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=agrupado,
            x='customer_state' if tipo == "clientes" else 'seller_state',
            y='count',
            palette='Reds_r',
            ax=ax
        )

        ax.set_title(orden_opcion)
        ax.set_xlabel('Estado')
        ax.set_ylabel('Nº ' + ('Clientes' if tipo == "clientes" else 'Vendedores'))

        for index, row in agrupado.iterrows():
            ax.text(
                x=index,
                y=row['count'],
                s=f"{row['percentage']}%",
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='black'
            )

        st.pyplot(fig)

    with col_tabla:
        st.markdown(f"#### Número de {'Clientes' if tipo == 'clientes' else 'Vendedores'} por Ciudad")
        st.dataframe(clientes_por_ciudad, use_container_width=True)




st.title("Mapa Brasil - Densidad de clientes y vendedores por estado")

# Crear dos columnas para mostrar los mapas en paralelo
col1, col2 = st.columns(2)

# URL del GeoJSON
geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'

#El filtrado de fechas solo actuará sobre el mapa de clientess
if tipo == "clientes":
    df_filtrado = df_customers_orders[
        (df_customers_orders['order_purchase_timestamp'] >= pd.to_datetime(fecha_inicio)) &
        (df_customers_orders['order_purchase_timestamp'] <= pd.to_datetime(fecha_fin))
    ]
else:
    # Si no se usa clientes, aseguramos que df_filtrado exista para los mapas sin romper el flujo
    df_filtrado = df_customers_orders.copy()

# Mapa de clientes

with col1:
    st.subheader("Clientes por estado")

    # Agrupar por estado sin limitar por top_n para el mapa
    clientes_por_estado_mapa = (
        df_filtrado.groupby('customer_state')['customer_unique_id']
        .nunique()
        .reset_index()
        .rename(columns={'customer_unique_id': 'count_customers'})
    )
    clientes_por_estado_mapa['customer_state'] = clientes_por_estado_mapa['customer_state'].str.upper()

    fig_clientes = px.choropleth(
        clientes_por_estado_mapa,
        geojson=geojson_url,
        locations='customer_state',
        featureidkey='properties.sigla',
        color='count_customers',
        color_continuous_scale='OrRd',
        scope='south america',
        labels={'count_customers': 'Nº Clientes'}
    )
    fig_clientes.update_geos(fitbounds="locations", visible=False)
    fig_clientes.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig_clientes, use_container_width=True)

# Mapa de los vendedores
with col2:
    st.subheader("Vendedores por estado")

    # Agrupar vendedores únicos por estado
    vendedores_por_estado_mapa = (
        df_sellers.groupby('seller_state')['seller_id']
        .nunique()
        .reset_index()
        .rename(columns={'seller_id': 'count_sellers'})
    )
    vendedores_por_estado_mapa['seller_state'] = vendedores_por_estado_mapa['seller_state'].str.upper()

    fig_vendedores = px.choropleth(
        vendedores_por_estado_mapa,
        geojson=geojson_url,
        locations='seller_state',
        featureidkey='properties.sigla',
        color='count_sellers',
        color_continuous_scale='Blues',
        scope='south america',
        labels={'count_sellers': 'Nº Vendedores'}
    )
    fig_vendedores.update_geos(fitbounds="locations", visible=False)
    fig_vendedores.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig_vendedores, use_container_width=True)






    # *** GRÁFICO DEL APARTDO 4 ***
st.header("4.Reviews por estado")

#Primero preparo el dataframe con todos los datos que necesito:


df_customers_reviews = pd.merge(
    pd.merge(df_reviews, df_orders, on='order_id', how='inner'),
    df_customers, on='customer_id', how='inner'
)

# print(df_customers_reviews.head(5))
# df_customers_reviews.info()

datetime_params = {
        "review_creation_date": "%Y-%m-%dT%H:%M:%S.%f",
        "review_answer_timestamp": "%Y-%m-%dT%H:%M:%S.%f",
        "order_purchase_timestamp": "%Y-%m-%d %H:%M:%S",
        "order_approved_at": "%Y-%m-%d %H:%M:%S",
        "order_delivered_carrier_date": "%Y-%m-%d %H:%M:%S",
        "order_delivered_customer_date": "%Y-%m-%d %H:%M:%S",
        "order_estimated_delivery_date": "%Y-%m-%d",
    }

df_customers_reviews = clean_datetime_columns_pandas(df_customers_reviews, datetime_params)

# df_customers_reviews.info()

# Añado una columna con los días de retraso (si es positivo habrá tenido retraso en la entrega)
df_customers_reviews['dias_retraso'] = (
    df_customers_reviews['order_delivered_customer_date'].dt.floor('D') - df_customers_reviews['order_estimated_delivery_date']
).dt.days.astype('Int64')




# Inputs de filtrado
c1, c2 = st.columns([6, 1])

with c1:
    min_date = df_customers_reviews['review_creation_date'].min().date()
    max_date = df_customers_reviews['review_creation_date'].max().date()
    fecha_inicio, fecha_fin = st.slider(
        "Selecciona el rango de fechas de review",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="DD/MM/YYYY"
    )

with c2:
    excluir_retrasos = st.checkbox("Excluir retrasos", value=True)

col1, col2 = st.columns(2)

with col1:
    top_n = st.selectbox("Nº de estados a mostrar", options=[5, 10, 15, 20], index=1)

with col2:
    orden = st.selectbox(
        "Estados a mostrar",
        options=[
            "Mayor número de reviews",
            "Menor número de reviews",
            "Mayor puntuación media",
            "Menor puntuación media"
        ],
        index=0
    )


# Aplicar filtros
filtro_fecha = (df_customers_reviews['review_creation_date'].dt.date >= fecha_inicio) & \
               (df_customers_reviews['review_creation_date'].dt.date <= fecha_fin)
df_filtrado = df_customers_reviews[filtro_fecha]

if excluir_retrasos:
    df_filtrado = df_filtrado[~(df_filtrado['dias_retraso'] > 0)]

# Agrupar
df_grouped = df_filtrado.groupby('customer_state').agg(
    num_reviews=('review_score', 'count'),
    avg_score=('review_score', 'mean')
).reset_index()

# Ordenar 
if orden == "Mayor número de reviews":
    df_grouped = df_grouped.sort_values(by='num_reviews', ascending=False)
elif orden == "Menor número de reviews":
    df_grouped = df_grouped.sort_values(by='num_reviews', ascending=True)
elif orden == "Mayor puntuación media":
    df_grouped = df_grouped.sort_values(by='avg_score', ascending=False)
elif orden == "Menor puntuación media":
    df_grouped = df_grouped.sort_values(by='avg_score', ascending=True)


df_grouped = df_grouped.head(top_n)


fig, ax1 = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")

# Eje izquierdo: número de reviews
sns.barplot(
    data=df_grouped,
    x='customer_state',
    y='num_reviews',
    color='skyblue',
    ax=ax1,
    label="Número de Reviews"
)

ax1.set_ylabel("Número de Reviews")
ax1.set_xlabel("Estado")
max_reviews = df_grouped['num_reviews'].max()
ax1.set_ylim(0, max_reviews * 1.2)

# Eje derecho (oculto): puntuación media y oculto también su grid
ax2 = ax1.twinx()
ax2.grid(False)  

ax2.plot(
    df_grouped['customer_state'],
    df_grouped['avg_score'],
    color='orange',
    marker='o',
    linewidth=2,
    linestyle='-',
    label='Puntuación Media'
)

# Etiquetas de puntuación junto a los puntos
for x, y in zip(range(len(df_grouped)), df_grouped['avg_score']):
    ax2.text(
        x=x,
        y=y + 0.05,
        s=f"{y:.2f}",
        ha='center',
        va='bottom',
        fontsize=9,
        color='black'
    )

# Ocultar eje derecho (pero mantener escala para que se vean bien los puntos)
max_score = df_grouped['avg_score'].max()
ax2.set_ylim(0, max_score * 1.2)
ax2.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
ax2.set_ylabel("")

# Leyenda manual
ax1.legend(["Número de Reviews"], loc='upper left')
ax1.plot([], [], color='orange', marker='o', linestyle='-', label='Puntuación Media')
ax1.legend(loc='upper right')

col1, col2, col3 = st.columns([1, 5, 1])  # Relación de anchos

with col2:
    st.pyplot(fig)

print('df_customers_reviews')
print(df_customers_reviews.info())




    
st.header("5. Reviews por categoría de producto")

# Unimos order_items con productos
df_items_productos = pd.merge(df_order_items, df_products, on='product_id', how='left')

# Aseguramos que el campo price sea numérico
df_items_productos['price'] = pd.to_numeric(df_items_productos['price'], errors='coerce')

# Unimos df_customers_reviews con order_items + productos (por order_id)
df_reviews_categoria = pd.merge(
    df_customers_reviews,
    df_items_productos[['order_id', 'product_category_name', 'price']],
    on='order_id',
    how='left'
)

# Filtros de Streamlit
c1, c2 = st.columns([6, 1])

with c1:
    min_date = df_reviews_categoria['review_creation_date'].min().date()
    max_date = df_reviews_categoria['review_creation_date'].max().date()
    fecha_inicio, fecha_fin = st.slider(
        "Selecciona el rango de fechas de review",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="DD/MM/YYYY",
        key="slider_fecha_reviews_categoria"
    )

with c2:
    excluir_retrasos = st.checkbox("Excluir retrasos", value=True, key="checkbox_excluir_retrasos_categoria")

col1, col2 = st.columns(2)

with col1:
    top_n = st.selectbox("Nº de categorías a mostrar", options=[5, 10, 15, 20], index=1)

with col2:
    orden = st.selectbox(
        "Métrica a mostrar",
        options=[
            "Mayor número de reviews",
            "Menor número de reviews",
            "Mayor puntuación media",
            "Menor puntuación media",
            "Mayor facturación",
            "Menor facturación"
        ],
        index=0
    )

# Filtro de fechas y retrasos
filtro_fecha = (df_reviews_categoria['review_creation_date'].dt.date >= fecha_inicio) & \
               (df_reviews_categoria['review_creation_date'].dt.date <= fecha_fin)
df_filtrado = df_reviews_categoria[filtro_fecha]

if excluir_retrasos:
    df_filtrado = df_filtrado[~(df_filtrado['dias_retraso'] > 0)]

# Agrupación por categoría incluyendo facturación
df_grouped = df_filtrado.groupby('product_category_name').agg(
    num_reviews=('review_score', 'count'),
    avg_score=('review_score', 'mean'),
    total_facturacion=('price', 'sum')
).reset_index()

# Limpieza de posibles nulos en facturación
df_grouped['total_facturacion'] = df_grouped['total_facturacion'].fillna(0)

# Ordenamiento según selección
if orden == "Mayor número de reviews":
    df_grouped = df_grouped.sort_values(by='num_reviews', ascending=False)
elif orden == "Menor número de reviews":
    df_grouped = df_grouped.sort_values(by='num_reviews', ascending=True)
elif orden == "Mayor puntuación media":
    df_grouped = df_grouped.sort_values(by='avg_score', ascending=False)
elif orden == "Menor puntuación media":
    df_grouped = df_grouped.sort_values(by='avg_score', ascending=True)
elif orden == "Mayor facturación":
    df_grouped = df_grouped.sort_values(by='total_facturacion', ascending=False)
elif orden == "Menor facturación":
    df_grouped = df_grouped.sort_values(by='total_facturacion', ascending=True)

# Nos quedamos con las N categorías seleccionadas
df_grouped = df_grouped.head(top_n)

# Layout: tabla y gráfico en paralelo
col_grafico, col_tabla = st.columns([2, 2])

with col_grafico:
    # Determinar qué variable mostrar en el eje Y
    if "facturación" in orden:
        y_var = "total_facturacion"
        y_label = "Facturación Total"
        legend_label = "Facturación Total"
    else:
        y_var = "num_reviews"
        y_label = "Número de Reviews"
        legend_label = "Número de Reviews"

    # Gráfico
    fig, ax1 = plt.subplots(figsize=(10, 5))  # proporción adecuada al layout
    sns.set_style("whitegrid")

    # Barras (siempre mismo color)
    sns.barplot(
        data=df_grouped,
        x='product_category_name',
        y=y_var,
        color='skyblue',
        ax=ax1
    )

    # Mostrar etiqueta en eje Y izquierdo
    ax1.set_ylabel(y_label)

    ax1.set_xlabel("Categoría de Producto")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    max_y = df_grouped[y_var].max()
    ax1.set_ylim(0, max_y * 1.2)

    # Línea de puntuación media
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.plot(
        df_grouped['product_category_name'],
        df_grouped['avg_score'],
        color='orange',
        marker='o',
        linewidth=2,
        linestyle='-'
    )

    # Etiquetas
    for x, y in zip(range(len(df_grouped)), df_grouped['avg_score']):
        ax2.text(
            x=x,
            y=y + 0.05,
            s=f"{y:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

    ax2.set_ylim(0, df_grouped['avg_score'].max() * 1.2)
    ax2.tick_params(axis='y', left=False, right=False, labelleft=False, labelright=False)
    ax2.set_ylabel("")

    st.pyplot(fig)


with col_tabla:
    # Calculamos totales para porcentaje
    total_reviews = df_grouped['num_reviews'].sum()
    total_facturacion = df_grouped['total_facturacion'].sum()

    # Añadimos columnas de porcentaje
    df_grouped['% Número de Reviews'] = df_grouped['num_reviews'] / total_reviews * 100
    df_grouped['% Facturación Total'] = df_grouped['total_facturacion'] / total_facturacion * 100

    # Redondeamos para mejor visualización
    df_grouped['% Número de Reviews'] = df_grouped['% Número de Reviews'].round(2)
    df_grouped['% Facturación Total'] = df_grouped['% Facturación Total'].round(2)

    # Renombramos columnas para la tabla
    df_mostrar = df_grouped.rename(columns={
        'product_category_name': 'Categoría de Producto',
        'num_reviews': 'Número de Reviews',
        'avg_score': 'Puntuación Media',
        'total_facturacion': 'Facturación Total',
        '% Número de Reviews': '% Número de Reviews',
        '% Facturación Total': '% Facturación Total'
    })

    # Mostramos tabla con nuevas columnas de porcentaje
    st.dataframe(
        df_mostrar[['Categoría de Producto', 'Número de Reviews', '% Número de Reviews',
                    'Puntuación Media', 'Facturación Total', '% Facturación Total']],
        use_container_width=True
    )



   
st.header("6. Métricas de la evolución del negocio")


# Columnas
col1, col2 = st.columns(2)

# Selector de métricas a mostrar
with col1:
    metrica = st.selectbox(
        "Selecciona la métrica a mostrar",
        options=[
            'Volumen de Facturación',
            'Volumen de pedidos',
            'Volumen de clientes',
            'Puntuación media de las Reviews',
            'Porcentaje de pedidos con retraso',
            'Días de hasta la entrega de un pedido (Mediana)',
            'Días de retraso sobre el total de pedidos con retraso (Mediana)',
        ]
    )

# Selector de periodo de agrupación
with col2:
    periodo = st.selectbox(
        "Selecciona el periodo de agrupación",
        options=['Diario', 'Semanal', 'Mensual', 'Trimestral'],
        index=2
    )

# Selector estados
estados_disponibles = sorted(df_customers_orders['customer_state'].unique())
estados_seleccionados = st.multiselect(
    "Estados a mostrar (Por defecto todos)",
    options=estados_disponibles,
    default=estados_disponibles
)

# Slider de fechas
fecha_min = df_customers_orders['order_purchase_timestamp'].dt.date.min()
fecha_max = df_customers_orders['order_purchase_timestamp'].dt.date.max()

fecha_inicio, fecha_fin = st.slider(
    "Selecciona rango de fechas",
    min_value=fecha_min,
    max_value=fecha_max,
    value=(fecha_min, fecha_max),
    format="DD/MM/YYYY"
)

# Filtrado datos
df_filtrado = df_customers_orders[
    (df_customers_orders['order_purchase_timestamp'].dt.date >= fecha_inicio) &
    (df_customers_orders['order_purchase_timestamp'].dt.date <= fecha_fin) &
    (df_customers_orders['customer_state'].isin(estados_seleccionados))
].copy()

if df_filtrado.empty:
    st.warning("No hay datos para los filtros seleccionados.")
else:
    df_filtrado.set_index('order_purchase_timestamp', inplace=True)

    # Mapear periodo a frecuencia para resample
    if periodo == 'Diario':
        freq = 'D'
    elif periodo == 'Semanal':
        freq = 'W-MON'
    elif periodo == 'Trimestral':
        freq = 'Q'
    else:
        freq = 'M'

    # Índice según la opción seleccionada para switch
    opciones = [
        'Volumen de Facturación',
        'Volumen de pedidos',
        'Volumen de clientes',
        'Puntuación media de las Reviews',
        'Porcentaje de pedidos con retraso',
        'Días de hasta la entrega de un pedido (Mediana)',
        'Días de retraso sobre el total de pedidos con retraso (Mediana)',
    ]
    idx = opciones.index(metrica)

    switch = {
        # Volumen de Facturación (suma de precios de pedidos en el periodo)
        0: lambda: df_filtrado.reset_index().merge(
                df_order_items[['order_id', 'price']],
                on='order_id', how='left'
            ).set_index('order_purchase_timestamp')['price'].resample(freq).sum(),

        # Volumen de pedidos (número de pedidos en el periodo)
        1: lambda: df_filtrado['order_id'].resample(freq).count(),

        # Volumen de clientes únicos (número de clientes únicos en el periodo)
        2: lambda: df_filtrado['customer_unique_id'].resample(freq).nunique(),

        # Puntuación media de las reviews (media del score de review en el periodo)
        3: lambda: df_filtrado.reset_index().merge(
                df_customers_reviews[['order_id', 'review_score']],
                on='order_id', how='left'
            ).set_index('order_purchase_timestamp')['review_score'].resample(freq).mean(),

        # Porcentaje de pedidos con retraso (pedidos con días de retraso > 0 sobre total pedidos en el periodo)
        4: lambda: df_filtrado.reset_index().merge(
                df_customers_reviews[['order_id', 'dias_retraso']],
                on='order_id', how='left'
            ).set_index('order_purchase_timestamp').pipe(lambda d: (
                d[d['dias_retraso'] > 0]['order_id'].resample(freq).count() /
                d['order_id'].resample(freq).count() * 100
            )).fillna(0),

        # Mediana de días hasta la entrega de un pedido (mediana de diferencia entre entrega al cliente y compra)
        5: lambda: df_filtrado.assign(
                dias_entrega = (df_filtrado['order_delivered_customer_date'] - df_filtrado.index).dt.days
            )['dias_entrega'].resample(freq).median(),

        # Mediana de días de retraso sobre pedidos con retraso (mediana de días de retraso en pedidos retrasados)
        6: lambda: df_filtrado.reset_index().merge(
                df_customers_reviews[['order_id', 'dias_retraso']],
                on='order_id', how='left'
            ).set_index('order_purchase_timestamp').pipe(lambda d: (
                d[d['dias_retraso'] > 0]['dias_retraso'].resample(freq).median()
            ))
    }

    agrupado = switch.get(idx, lambda: pd.Series(dtype=float))()

    # Preparar DataFrame para gráfica
    evolucion = agrupado.reset_index().rename(columns={'order_purchase_timestamp': 'periodo'})
    evolucion.rename(columns={evolucion.columns[1]: 'valor'}, inplace=True)

    titulo = f"Evolución {periodo.lower()} de {metrica.lower()}"
    eje_y = metrica

    # Crear gráfica con plotly
    fig = px.line(
        evolucion,
        x='periodo',
        y='valor',
        labels={'periodo': 'Fecha', 'valor': eje_y},
        title=titulo,
        markers=True
    )
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title=eje_y,
        margin={"r":0, "t":40, "l":0, "b":0}
    )
    st.plotly_chart(fig, use_container_width=True)















