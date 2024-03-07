# grafico como luce la regresion al final
plt.scatter(x_train, y_train,
            color='blue',
            label='Datos de entrenamiento')  # Datos de entrenamiento
plt.scatter(x_test, y_test, color='red',
            label='Datos de prueba')  # Datos de prueba
plt.plot(x_train,
         w_final * x_train + b_final,
         color='green', label='Línea de regresión')  # Línea de regresión
plt.xlabel('Años de experiencia')
plt.ylabel('Salario')
plt.legend()
plt.title('Ajuste de regresión lineal')
plt.show()
plt.close()
