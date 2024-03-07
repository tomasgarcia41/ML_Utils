# grafico el cost vs iterations
plt.plot(range(len(J_hist)), J_hist, color='blue')
plt.xlabel('Número de Iteraciones')
plt.ylabel('Costo')
plt.title('Costo vs. Iteraciones')
plt.show(block=False)
plt.close()
