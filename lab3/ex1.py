# --- Exercise 1: Email classification Bayesian Network ---
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define network structure
model = DiscreteBayesianNetwork([
    ('S', 'O'),
    ('S', 'L'),
    ('S', 'M'),
    ('L', 'M')
])

# Define CPDs (conditional probability distributions)
cpd_S = TabularCPD('S', 2, [[0.6], [0.4]])  # P(S=0)=0.6, P(S=1)=0.4

cpd_O_given_S = TabularCPD('O', 2,
                            [[0.9, 0.3],  # O=0
                             [0.1, 0.7]],  # O=1
                            evidence=['S'], evidence_card=[2])

cpd_L_given_S = TabularCPD('L', 2,
                            [[0.7, 0.2],  # L=0
                             [0.3, 0.8]],  # L=1
                            evidence=['S'], evidence_card=[2])

cpd_M_given_S_L = TabularCPD('M', 2,
                             [[0.8, 0.4, 0.5, 0.1],  # M=0
                              [0.2, 0.6, 0.5, 0.9]],  # M=1
                             evidence=['S', 'L'], evidence_card=[2, 2])

# Add CPDs to model
model.add_cpds(cpd_S, cpd_O_given_S, cpd_L_given_S, cpd_M_given_S_L)
model.check_model()

# --- Inference ---
infer = VariableElimination(model)

# Example: P(S=1 | O=1, L=1, M=1)
posterior = infer.query(['S'], evidence={'O': 1, 'L': 1, 'M': 1})

for i in range(2):
    for j in range(2):
        for k in range(2):
            evidence = {'O': i, 'L': j, 'M': k}
            posterior = infer.query(['S'], evidence=evidence)
            print(f"P(S | O={i}, L={j}, M={k}):")
            print(posterior)  
#print("P(S | O=1, L=1, M=1):")
#print(posterior)
