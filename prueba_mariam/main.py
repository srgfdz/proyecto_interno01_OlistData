import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ampliar el ancho de la página
st.set_page_config(layout="wide")

# Título de sección
st.header("2. Estados con mayor porcentaje de pedidos")

# Cargar datos
df = pd.read_csv('C:/Users/mguadalupe/Desktop/proyecto_interno01_OlistData/datasets_limpios/olist_order_customer.csv')
df_orders = pd.read_csv('C:/Users/mguadalupe/Desktop/proyecto_interno01_OlistData/datasets_limpios/olist_orders_dataset.csv')

# Relacionar los datasets por 'customer_id'
df_merged = pd.merge(df, df_orders, on='customer_id', how='inner')

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
).round(0).astype(int)

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