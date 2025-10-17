
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('Die', 'Added'), ('Added', 'Draw')])

cpd_die = TabularCPD(variable='Die', variable_card=6,
                     values=[[1/6]] * 6) #cele 6 fete ale zarului

import numpy as np
added_table = np.zeros((3, 6))  #cele 3 culori adaugate x cele 6 fete ale zarului

# die = 0 -> add red
# die = 1 -> add blue
# die = 2 -> add black

added_table[1, 0] = 1.0  

added_table[2, 1] = 1.0  

added_table[2, 2] = 1.0

added_table[1, 3] = 1.0

added_table[2, 4] = 1.0

added_table[0, 5] = 1.0

#penrtu cele 6 fete ale zarului, adaugam un zazr de o culoare specifica

cpd_added = TabularCPD(variable='Added', variable_card=3,
                       values=added_table.tolist(),
                       evidence=['Die'], evidence_card=[6])


cpd_draw = TabularCPD(
    variable='Draw', variable_card=2,
    values=[
        [1 - 4/10, 1 - 3/10, 1 - 3/10],
        [4/10, 3/10, 3/10]
    ],
    evidence=['Added'],
    evidence_card=[3]
)

model.add_cpds(cpd_die, cpd_added, cpd_draw)
model.check_model()

infer = VariableElimination(model)
posterior = infer.query(['Draw'])
print(posterior)  
#extragem un rezultat al extragerii unei bile rosii
