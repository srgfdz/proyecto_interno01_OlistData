import pandas as pd
df=pd.read_csv('C:/Users/mguadalupe/Desktop/proyecto_interno01_OlistData/datasets_limpios/olist_order_customer.csv')


pedidos_por_ciudad = df.groupby(['customer_state', 'customer_city'])['customer_id'].count().reset_index()
pedidos_por_ciudad.columns = ['Estado', 'Ciudad', 'Número de pedidos']
pedidos_por_ciudad = pedidos_por_ciudad.sort_values(['Estado', 'Número de pedidos'], ascending=[True, False])
pedidos_por_ciudad.head(10)