import numpy as np
from hmmlearn import hmm
import networkx as nx  # Import networkx
import matplotlib.pyplot as plt  # Import matplotlib to show the plot

states = ["Difficult", "Medium", "Easy"]

n_components = 3

observations = ["FB", "B", "S", "NS"]
n_observations = 4

start_prob = np.array([1/3, 1/3, 1/3])

trans_matrix = np.array([
    [0.0, 0.5, 0.5], 
    [0.5, 0.25, 0.25], 
    [0.5, 0.25, 0.25]  
])

emission_matrix = np.array([
    [0.1, 0.2, 0.4, 0.3],  
    [0.15, 0.25, 0.5, 0.1], 
    [0.2, 0.3, 0.4, 0.1]   
])  

obs_sequence_str = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
obs_map = {"FB": 0, "B": 1, "S": 2, "NS": 3}
obs_sequence = np.array([obs_map[obs] for obs in obs_sequence_str])


#aici am facut o matrice cu o linie pentru fiecare timestep si o coloana pentru fiecare observatie posibila
obs_sequence_counts = np.zeros((len(obs_sequence), n_observations), dtype=int)
obs_sequence_counts[np.arange(len(obs_sequence)), obs_sequence] = 1



print("a)")
model = hmm.MultinomialHMM(n_components=n_components, n_trials=1, random_state=42)

model.startprob_ = start_prob
model.transmat_ = trans_matrix
model.emissionprob_ = emission_matrix

print("HMM Model Defined.")
print(f"States: {states}")
print(f"Start Probabilities:\n{model.startprob_}")
print(f"Transition Matrix:\n{model.transmat_}")
print(f"Emission Matrix:\n{model.emissionprob_}")


G = nx.DiGraph()
G.add_nodes_from(states)

edge_labels = {}
for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        prob = trans_matrix[i, j]
        if prob > 0:  
            G.add_edge(from_state, to_state, weight=prob)
            edge_labels[(from_state, to_state)] = f"{prob:.2f}"

pos = nx.circular_layout(G)

plt.figure(figsize=(10, 7))

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=4000,
    node_color='skyblue'
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=14,
    font_weight='bold'
)

nx.draw_networkx_edges(
    G,
    pos,
    width=2,
    arrowsize=20,
    edge_color='gray',
    connectionstyle='arc3,rad=0.1' # makes tje loops curved
)

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_color='black',
    font_size=12
)

plt.title("HMM State Transition Diagram", size=18)
plt.axis('off') 

out_path = 'lab5/ex1_graph.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight')  
print(f'Graph image saved to: {out_path}')

plt.show()   
plt.close()

print("b)")

log_prob_obs = model.score(obs_sequence_counts)
prob_obs = np.exp(log_prob_obs)

print(f"Observation Sequence: {obs_sequence_str}")
print(f"Log-Probability P(O | lambda): {log_prob_obs:.3f}")
print(f"Probability P(O | lambda): {prob_obs:.4e}")


print("c)")

log_prob_viterbi, hidden_state_indices = model.decode(obs_sequence_counts)
prob_viterbi = np.exp(log_prob_viterbi)

# pentru a putea afisa numele starilor
hidden_state_names = [states[i] for i in hidden_state_indices]

print(f"Most Probable Hidden Sequence:\n{hidden_state_names}")
print(f"Log-Probability of this path: {log_prob_viterbi:.3f}")
print(f"Probability of this path: {prob_viterbi:.4e}")

