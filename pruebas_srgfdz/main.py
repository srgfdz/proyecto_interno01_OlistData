import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
df_customers = pd.read_csv('./../datasets_limpios/olist_order_customer.csv')
df_orders = pd.read_csv('./../datasets_limpios/olist_orders_dataset.csv')

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