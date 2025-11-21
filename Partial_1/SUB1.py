from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('O','H'),
    ('O','W'),
    ('H','R'),
    ('W','R'),
    ('H','E'),
    ('R','C'),
])

cpd_OTemp = TabularCPD(variable='O', variable_card=2,values=[[0.3], [0.7]])

cpd_H_given_O = TabularCPD(variable='H', variable_card=2,values = [[0.9, 0.2], [0.1, 0.8]], evidence=['O'], evidence_card=[2])

cpd_W_given_O = TabularCPD(variable='W', variable_card=2,values = [[0.1, 0.6], [0.9, 0.4]], evidence=['O'], evidence_card=[2])

cpd_R_given_H_W = TabularCPD(variable='R', variable_card=2,values=  [[0.6, 0.9, 0.3, 0.5],[0.4, 0.1, 0.7, 0.5]],evidence=['H', 'W'], evidence_card=[2, 2])

cpd_E_given_H = TabularCPD(variable='E', variable_card=2,values=[[0.8, 0.2], [0.2, 0.8]],evidence=['H'], evidence_card=[2])

cpd_C_given_R = TabularCPD(variable='C', variable_card=2,values=[[0.85, 0.4], [0.15, 0.6]],evidence=['R'], evidence_card=[2])

model.add_cpds(cpd_OTemp, cpd_H_given_O, cpd_W_given_O, cpd_R_given_H_W, cpd_E_given_H, cpd_C_given_R)

inference = VariableElimination(model)

query_H_given_C = inference.query(variables=['H'], evidence={'C': 0})

query_E_given_C = inference.query(variables=['E'], evidence={'C': 0})

query_H_W_given_C = inference.query(variables=['H', 'W'], evidence={'C': 0})

print(query_E_given_C.get_value(E=0))
print(query_H_given_C.get_value(H=0))
maximum = -1
for i in range (2):
    for j in range (2):
        prob = query_H_W_given_C.get_value(H=i, W=j)
        if prob > maximum:
            maximum = prob
print(maximum)


# Daca H este observat, inseamnca ca el este blocat, deci O si E sunt independente. Cum W nu depinde de nimeni
# altcineva decat de O si O nu este observat, atunci W si E sunt independente.

# Daca R e observat, atucni O si C sunt dependente, deoarece in acest caz R se deblocheaza si astfel W si H sunt dependete
# prin intermediul lui O iar C depinde de R si astfel C depinde de O.