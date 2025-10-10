import random
import numpy as np
import matplotlib.pyplot as plt

def simulate():
   
    urn = ['r'] * 3 + ['u'] * 4 + ['b'] * 2

    roll = random.randint(1, 6)

    if roll in [2, 3, 5]:
        urn.append('b')
    elif roll == 6:
        urn.append('r')
    else:
        urn.append('u')

    total = len(urn)
    count_r = urn.count('r')
    count_u = urn.count('u')
    count_b = urn.count('b')
    probs = [count_r / total, count_u / total, count_b / total]

    return np.random.choice(['r', 'u', 'b'], p=probs)

N = 1000
results = [simulate() for _ in range(N)]

p_red = results.count('r') / N
print(f"Estimated probability of red: {p_red:.4f}")
