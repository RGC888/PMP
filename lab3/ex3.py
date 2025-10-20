import random
import math
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# a)
def simulate_game():
    #fair coid
    starter = random.choice(["P0", "P1"])  # P0=0, P1=1

    n = random.randint(1, 6)

    if starter == "P0":
        prob_head = 4 / 7
    else:
        prob_head = 0.5

    # 2n flips
    m = sum(1 for _ in range(2 * n) if random.random() < prob_head)

    if n >= m:
        return starter 
    else:
        return "P1" if starter == "P0" else "P0"

N = 10000
wins = {"P0": 0, "P1": 0}

for _ in range(N):
    winner = simulate_game()
    wins[winner] += 1

print(f"P0 wins: {wins['P0']} ({wins['P0'] / N:.2%})")
print(f"P1 wins: {wins['P1']} ({wins['P1'] / N:.2%})\n")

# b)
model = DiscreteBayesianNetwork([
    ('Starter', 'M'),
    ('N', 'M')
])

cpd_starter = TabularCPD(
    variable='Starter',
    variable_card=2,
    values=[[0.5], [0.5]] 
)

cpd_n = TabularCPD(
    variable='N',
    variable_card=6,
    values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]]
)


def binomial_pmf(k, n, p):
    comb = math.comb(n, k)
    return comb * (p ** k) * ((1 - p) ** (n - k))

values = []
for m in range(13):  # possible number of heads
    row = []
    for starter in [0, 1]: 
        p_head = 0.5 if starter == 0 else 4/7
        for n in range(1, 7):
            prob = binomial_pmf(m, 2 * n, p_head)
            row.append(prob)
    values.append(row)

cpd_m = TabularCPD(
    variable='M',
    variable_card=13,
    values=values,
    evidence=['Starter', 'N'],
    evidence_card=[2, 6]
)

model.add_cpds(cpd_starter, cpd_n, cpd_m)
model.check_model()


# c)

infer = VariableElimination(model)
posterior = infer.query(['Starter'], evidence={'M': 1})

print("=== Inference Results ===")
print("P(Starter | M=1):")
print(posterior)
print()

if posterior.values[0] > posterior.values[1]:
    print("→ More likely that P0 (fair coin) started the game.")
else:
    print("→ More likely that P1 (rigged coin) started the game.")
