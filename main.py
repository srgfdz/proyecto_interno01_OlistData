import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Importo las funciones propias de pipelines
from pipelines import *


st.title("Proyecto 01 - Olist Data")

st.markdown("""
## 👥 Integrantes del grupo

- **Javier Sánchez de las Heras** - [@javierstemdo](https://github.com/javierstemdo)  
- **Mariam Guadalupe Núñez** - [@mariamgn-stemdo](https://github.com/mariamgn-stemdo)  
- **Sergio Fernández Nevado** - [@srgfdz](https://github.com/srgfdz)
""")

# *** GRÁFICO DEL APARTDO 1 ***
st.header("1. Número de Clientes por Estado")

# Cargar datos para el punto 1 (clientes y pedidos)
df_customers = pd.read_csv('./datasets_limpios/olist_order_customer.csv')
df_orders = pd.read_csv('./datasets_limpios/olist_orders_dataset.csv')

#Hago uun merge outer, pero en este caso los registros de ambos dataset coinciden exactamente:
# no hay clientes que no tengan ningún pedido y cada cliente tiene un único pedido, por lo que no necesitamos transformar el df mergeado
df_customers_orders = df_customers.merge(df_orders, on='customer_id', how='outer')
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

fecha_min = df_customers_orders['order_purchase_timestamp'].min().to_pydatetime()
fecha_max = df_customers_orders['order_purchase_timestamp'].max().to_pydatetime()


fecha_inicio, fecha_fin = st.slider(
    "Rango de fechas de compra",
    min_value=fecha_min,
    max_value=fecha_max,
    value=(fecha_min, fecha_max),
    format="DD/MM/YYYY"
)


# Aplicar filtro
df_filtrado = df_customers_orders[
    (df_customers_orders['order_purchase_timestamp'] >= pd.to_datetime(fecha_inicio)) &
    (df_customers_orders['order_purchase_timestamp'] <= pd.to_datetime(fecha_fin))
]


# Agrupar por estado y cliente y mostrar solo los que indica el filtro desplegable de Nº y según el orden seleccionado
clientes_por_estado = (
    df_filtrado.groupby('customer_state')['customer_id']
    .nunique()
    .sort_values(ascending=orden_invertido)
    .head(top_n)
    .reset_index()
    .rename(columns={'customer_id': 'count_customers'})
)




# Comprobar si hay datos para los filtros seleccionados
if clientes_por_estado.empty:
   st.markdown("""
        <div style="padding: 20px; margin: 20px; border: 2px solid white; border-radius: 10px; background-color: #f8d7da; color: #721c24; text-align: center;">
            <strong>*** No hay registros para los filtros aplicados, ajusta los filtros y prueba nuevamente ***</strong>
        </div>
    """, unsafe_allow_html=True)
else:
    # Convertir los nombres de los estados a mayúsculas
    clientes_por_estado['customer_state'] = clientes_por_estado['customer_state'].str.upper()
    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=clientes_por_estado,
        x='customer_state',
        y='count_customers',
        palette='Reds_r',
        ax=ax
    )
    ax.set_title(f"{'Estados con menos' if orden_invertido else 'Estados con más clientes'}")
    ax.set_xlabel('Estado')
    ax.set_ylabel('Nº clientes')

    st.pyplot(fig)




# *** GRÁFICO DEL APARTDO 2 ***

# Ampliar el ancho de la página
# st.set_page_config(layout="wide")

# Título de sección
st.header("2. Estados con mayor porcentaje de pedidos")

# Relacionar los datasets por 'customer_id'
df_merged = pd.merge(df_customers, df_orders, on='customer_id', how='inner')

# Asegurar mayúsculas en estados y formato título en ciudades
df_merged['customer_state'] = df_merged['customer_state'].astype(str).str.upper()
df_merged['customer_city'] = df_merged['customer_city'].astype(str).str.title()

# Calcular número de clientes únicos por ciudad
clientes_por_ciudad = df_merged.groupby(['customer_state', 'customer_city'])['customer_unique_id'].nunique().reset_index()
clientes_por_ciudad.columns = ['Estado', 'Ciudad', 'Número de clientes']

# Calcular número de pedidos por ciudad (conteo de order_id por ciudad)
pedidos_por_ciudad = df_merged.groupby(['customer_state', 'customer_city'])['order_id'].count().reset_index()
pedidos_por_ciudad.columns = ['Estado', 'Ciudad', 'Número de pedidos']

# Combinar ambas tablas
tabla = pedidos_por_ciudad.merge(clientes_por_ciudad, on=['Estado', 'Ciudad'])

# Calcular ratio como número entero
tabla['Ratio de Pedidos por Cliente'] = (
    tabla['Número de pedidos'] / tabla['Número de clientes']
).round(2)

# Select de estado
estados_unicos = sorted(tabla['Estado'].unique())
estado_seleccionado = st.selectbox("Selecciona un estado para ver sus ciudades", estados_unicos)

# Filtrar por estado y preparar Top 10
datos_estado = tabla[tabla['Estado'] == estado_seleccionado].copy()
datos_estado = datos_estado.sort_values(by='Número de pedidos', ascending=False)

top10_estados = datos_estado.head(10).copy()
total_top10 = top10_estados['Número de pedidos'].sum()

# Calcular porcentaje dentro del top 10
top10_estados['Porcentaje'] = (
    top10_estados['Número de pedidos'] / total_top10 * 100
).round(2)

# Gráfico circular
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.PuBu(np.linspace(0.3, 1, len(top10_estados)))

ax.pie(
    top10_estados['Porcentaje'],
    labels=top10_estados['Ciudad'],
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 9},
    wedgeprops=dict(edgecolor='white'),
    labeldistance=1.05
)

ax.set_title(f'Top 10 ciudades por pedidos en {estado_seleccionado}', fontsize=14)
ax.axis('equal')

# Mostrar gráfico y tabla
col1, col2 = st.columns([1.3, 1.7])

with col1:
    st.pyplot(fig)

with col2:
    st.subheader(f"Detalle de ciudades en {estado_seleccionado}")
    tabla_filtrada = top10_estados[
        ['Ciudad', 'Número de pedidos', 'Número de clientes', 'Porcentaje', 'Ratio de Pedidos por Cliente']
    ].reset_index(drop=True)

    st.dataframe(tabla_filtrada, use_container_width=True)



# *** GRÁFICO EXTRA ***
st.header("Pagos totales por persona")

# Cargar datos para el punto 1 (clientes y pedidos)
df_order_payments = pd.read_csv('./datasets_limpios/olist_order_payments_dataset_clean.csv')

# Unir tablas usando order_id
df_merged = pd.merge(df_order_payments, df_orders[['order_id', 'customer_id']], on='order_id', how='inner')

# Agrupar pagos por cliente
df_summary = df_merged.groupby('customer_id')['payment_value'].sum().reset_index()

# Gráfico 
# Ordenar por total pagado
top_n = st.slider("Selecciona el número de clientes a mostrar", min_value=5, max_value=50, value=20)

df_topN = df_summary.sort_values(by='payment_value', ascending=False).head(top_n)

plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

ax = sns.scatterplot(
    x='payment_value',
    y='customer_id',
    data=df_topN,
    s=100,
    color='royalblue'
)

ax.set_xlabel('Total Payment Value')
ax.set_ylabel('Customer ID')
ax.set_title(f'Top {top_n} clientes por total pagado')

# Acortar etiquetas para mejor visualización
new_labels = [c[:8] + "..." for c in df_topN['customer_id']]
ax.set_yticklabels(new_labels)

st.pyplot(plt)