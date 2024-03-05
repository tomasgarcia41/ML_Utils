# Lista para almacenar las columnas a eliminar
columnas_a_eliminar = []

# Iteramos sobre todas las columnas del DataFrame
for columna in df.columns:
    # Si la cantidad de valores únicos en la columna es igual a 1, significa que tiene un solo valor
    if len(df[columna].unique()) == 1:
        # Agregamos la columna a la lista de columnas a eliminar
        columnas_a_eliminar.append(columna)

# Eliminamos las columnas del DataFrame
df = df.drop(columns=columnas_a_eliminar)

# Mostramos el DataFrame resultante
print("DataFrame sin las columnas con un solo valor:")
print(df)

""" esto se usa para los Decision Trees, ya que las columnas con un solo valor no nos serviran para dividir """
