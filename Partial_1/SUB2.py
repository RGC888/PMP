import numpy as np
from hmmlearn import hmm
import networkx as nx
import matplotlib.pyplot as plt  

n_observations = 3

activity_states = ['W', 'R', 'S']

heart_rate_states = ['L', 'M', 'H']

step_count_states = ['L', 'M', 'H']

start_prob = np.array([0.4,0.3,0.3])

trans_matrix = np.array([
    [0.6, 0.3, 0.1], 
    [0.2, 0.7, 0.1], 
    [0.3, 0.2, 0.5]  
])


emission_matrix = np.array([
    [0.1, 0.7, 0.2],   
    [0.05, 0.25, 0.7], 
    [0.8, 0.15, 0.05]    
])  

model = hmm.MultinomialHMM(n_components=len(activity_states), n_iter=100, tol=0.01, init_params="")
model.startprob_ = start_prob
model.transmat_ = trans_matrix
model.emissionprob_ = emission_matrix

print("HMM Model Defined.")
print(f"States: {activity_states}")
print(f"Start Probabilities:\n{model.startprob_}")
print(f"Transition Matrix:\n{model.transmat_}")
print(f"Emission Matrix:\n{model.emissionprob_}")

obs_sequence_str = {"M", "H", "L"}

obs_map = {"M": 0, "H": 1, "L": 2}
obs_sequence = np.array([obs_map[obs] for obs in obs_sequence_str])


obs_sequence_counts = np.zeros((len(obs_sequence), n_observations), dtype=int)
obs_sequence_counts[np.arange(len(obs_sequence)), obs_sequence] = 1

log_prob_obs = model.score(obs_sequence_counts)
prob_obs = np.exp(log_prob_obs)

print(f"Observation Sequence: {obs_sequence_str}")
print(f"Log-Probability P(O | lambda): {log_prob_obs:.3f}")
print(f"Probability P(O | lambda): {prob_obs:.4e}")


