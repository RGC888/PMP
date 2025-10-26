import numpy as np
import matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# generez o matrice de 5x5 care reprezinta o imagine binara
np.random.seed(0)
original = np.random.randint(0, 2, (5, 5))  # imagine binară (0 sau 1)
plt.imshow(original, cmap='gray')
plt.title("Original Image")
plt.show()

#adauf zgomot 10%
noisy = original.copy()
num_noisy = int(0.1 * original.size)
indices = np.random.choice(original.size, num_noisy, replace=False)
flat_noisy = noisy.flatten()
flat_noisy[indices] = 1 - flat_noisy[indices]  # inverseaza valoarea pixelului
noisy = flat_noisy.reshape(original.shape)

plt.imshow(noisy, cmap='gray')
plt.title("Noisy Image")
plt.show()


model = MarkovNetwork()


nodes = [(i, j) for i in range(5) for j in range(5)]
model.add_nodes_from(nodes)


edges = []
for i in range(5):
    for j in range(5):
        if i < 4:
            edges.append(((i, j), (i + 1, j)))
        if j < 4:
            edges.append(((i, j), (i, j + 1)))

model.add_edges_from(edges)

λ = 2.0  
factors = []

for i in range(5):
    for j in range(5):
        y = noisy[i, j]
        values = [np.exp(-λ * (x - y)**2) for x in [0, 1]]
        factor = DiscreteFactor(variables=[(i, j)], cardinality=[2], values=values)
        factors.append(factor)

beta = 1.0  
for (a, b) in edges:
    values = [
        np.exp(-beta * (x1 - x2)**2)
        for x1 in [0, 1]
        for x2 in [0, 1]
    ]
    factor = DiscreteFactor(variables=[a, b], cardinality=[2, 2], values=values)
    factors.append(factor)

model.add_factors(*factors)


bp = BeliefPropagation(model)
map_result = bp.map_query(variables=nodes)

denoised = np.zeros_like(original)
for (i, j), val in map_result.items():
    denoised[i, j] = val

plt.imshow(denoised, cmap='gray')
plt.title("Denoised (MAP Estimate)")
plt.show()
