import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_decision_surface import plot_decision_regions

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


##### Perceptron from Sklearn #####
# importar classificador
from sklearn.linear_model import Perceptron
# passar parâmetros
ppn = Perceptron(max_iter=500, random_state=1)
# treinar com os dados
ppn.fit(X, y)

# plotar dados, superfície de decisão e margem
plot_decision_regions(X=X, y=y, classifier=ppn, test_idx=None)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Perceptron")
plt.savefig("figures/iris_sklearn_perceptron.png")



