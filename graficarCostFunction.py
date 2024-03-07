
# grafico el costo para los distintos valores de w
w_values = np.linspace(-1000000, 1000000, 10000)  # Modifica el rango según sea necesario

cost_values = [cost_function(x_train, y_train, w, b_final) for w in w_values]

plt.plot(w_values, cost_values, color='blue')
plt.xlabel('Valor de w')
plt.ylabel('Costo')
plt.title('Costo en función de w')
plt.ticklabel_format(style='plain', axis='both')
plt.show()
