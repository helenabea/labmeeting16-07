import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Preparar dados de treinamento 
df = pd.read_csv("data/iris.data.csv", header=None)

# Coletar 25 amostras para Iris-setosa e 
# Iris-versicolor
versicolor = df[df[4] == "Iris-versicolor"][0:25]
setosa = df[df[4] == "Iris-setosa"][0:25]
# Novo dataframe
df = pd.concat([versicolor, setosa])

# Configurar numpy array
X = df[[0,1]].values
y = df[4].values

# Transformar nomes em valores
y = np.where(y == "Iris-versicolor", 1, -1)

# Plotar os pontos

# Encontrar mínimo e máximo
x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.title("Plotagem dos Dados")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X[y == 1, 0], X[y == 1, 1] ,
	color="blue", marker="x")
plt.scatter(X[y == -1, 0], X[y == -1, 1] ,
	color="red", marker="o")
plt.plot()
plt.savefig("figures/iris_scatter.png")

##### Perceptron Estocástico #####
# 1. Inicializar o vertor w e b com números 
# aleatorios
eta = 0.0001 # coeficiente de aprendizagem
rgen = np.random.RandomState(0)
# escolhendo 2 valores muito pequenos para w1 e w2
w = rgen.normal(loc=0, scale=0.0001, size=2) 
# escolhendo um valor muito pequeno para o viés
b = rgen.normal(loc=0, scale=0.0001) 

# 2. para cada exemplo (xi, yi)
while True: 
	has_error = False
	for (xi, yi) in zip(X,y): 
		# multiplica os pesos pelas 
		# caracteristicas e somamos o viés
		zi = np.dot(w, xi) + b
		# função de ativação
		fi = 1 if zi >= 0 else -1
		# calculo do erro
		# valor real - valor predito
		ei = yi - fi
		
		# se tem erro
		if ei != 0:
			has_error = True
			# atualização dos pesos
			w = w + eta * ei * xi
			b = b + eta * ei

	# sem erro
	if not has_error:
		break
	# 3. volte para o passo 2 até que todos ei 
	# sejam iguais a zero

# configurando as constantes a e b
# x2 = a*x1 + b
a = -(w[0]/w[1])
b = -(b/w[1])

# Gerando 100 amostras para x1
x1 = np.linspace(start=x1_min, stop=x1_max, num=100) 

# calculado as ordenadas "y"
x2 = a*x1 + b

# plotar perceptron
plt.title("Plotagem do Perceptron Estocástico")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.scatter(X[y == 1, 0], X[y == 1, 1] ,color="blue", marker="x")
plt.scatter(X[y == -1, 0], X[y == -1, 1] ,color="red", marker="o")
plt.plot(x1, x2, color="black")
plt.savefig("figures/iris_scatter_perceptron.png")