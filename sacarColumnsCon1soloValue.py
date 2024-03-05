for columna in df.columns:
    # Si la cantidad de valores únicos en la columna es igual a 1, significa que tiene un solo valor
    if len(df[columna].unique()) == 1:
        print(f"La columna {columna} tiene un solo valor: {df[columna].unique()[0]}")
