# GradientDescent.py
"""
Autor: Arturo Toro
Carnet: 12-10796
Carrera: Ingenieria en computacion - USB
"""
import numpy as np

# W(n+1) = W(n) - lr*dE(W)/dW
# Descenso de gradiente: Cauchy

def LMS(X, d, f, alfa):
	""" En la actualizacion continua se presenta un dato a la vez
	y se actualizan los pesos.
	# Consideraciones: - Tasa de aprendizaje fija.
					   - La diferencia con cauchy es alfa fijo.
	# Entrada: Matriz de datos (X),
	 		   Vector de resultados esperados (d),
	 		   Funcion de activacion (f),
	 		   Tasa de aprendizaje (alfa)
	"""

	# Inicializacion de vectores.
	Ys = np.zeros(np.size(X,1))		# Vector de resultados.
	error = np.ones(np.size(X,1))	# Vector de errores.
	Ws = np.ones(np.size(X,1))		# Vector de pesos.

	for i in range(np.size(X,2)):
		Ys[i] = f(X[i]*Ws) 				# Calcular respuesta.
		error[i] = d[i] - Ys[i]			# Error de respuesta.

		# Para la actualizacion, mover a los pesos hacia el dato.
		Ws = Ws + alfa*error[i]*X[i]	# Actualizacion online.

	# Seccion para graficar error y eso.	
	# Evaluar condicion de parada.
