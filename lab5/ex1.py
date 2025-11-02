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

# am adaugat codul de mai jos pentru a salva o diagrama completa a modelului HMM

print("\nVisualizing the Full HMM Model (States, Transitions, and Emissions)")

Full_HMM_Graph = nx.DiGraph()


for i, state_name in enumerate(states):
    Full_HMM_Graph.add_node(f"State: {state_name}", type='state', state_index=i)


for i, obs_name in enumerate(observations):
    Full_HMM_Graph.add_node(f"Obs: {obs_name}", type='observation', obs_index=i)


state_pos = nx.circular_layout([f"State: {s}" for s in states])


obs_pos = {}
radius = 1.5 
angle_step = 2 * np.pi / len(observations)
for i, obs_name in enumerate(observations):
    angle = i * angle_step
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    obs_pos[f"Obs: {obs_name}"] = np.array([x, y])


overall_pos = {**state_pos, **obs_pos}


transition_labels = {}
for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        prob = trans_matrix[i, j]
        if prob > 0:
            Full_HMM_Graph.add_edge(f"State: {from_state}", f"State: {to_state}", weight=prob)
            transition_labels[(f"State: {from_state}", f"State: {to_state}")] = f"{prob:.2f}"

emission_labels = {}
for i, state_name in enumerate(states):
    for j, obs_name in enumerate(observations):
        prob = emission_matrix[i, j]
        if prob > 0:
            Full_HMM_Graph.add_edge(f"State: {state_name}", f"Obs: {obs_name}", weight=prob, relation='emits')
            emission_labels[(f"State: {state_name}", f"Obs: {obs_name}")] = f"{prob:.2f}"


plt.figure(figsize=(15, 12))

nx.draw_networkx_nodes(
    Full_HMM_Graph,
    overall_pos,
    nodelist=[n for n in Full_HMM_Graph.nodes if Full_HMM_Graph.nodes[n]['type'] == 'state'],
    node_size=4000,
    node_color='skyblue',
    label='Hidden States'
)

nx.draw_networkx_nodes(
    Full_HMM_Graph,
    overall_pos,
    nodelist=[n for n in Full_HMM_Graph.nodes if Full_HMM_Graph.nodes[n]['type'] == 'observation'],
    node_size=2000,
    node_color='lightcoral',
    label='Observations'
)


nx.draw_networkx_labels(
    Full_HMM_Graph,
    overall_pos,
    font_size=10,
    font_weight='bold'
)

nx.draw_networkx_edges(
    Full_HMM_Graph,
    overall_pos,
    edgelist=[(u, v) for u, v, d in Full_HMM_Graph.edges(data=True) if d.get('relation') != 'emits'],
    width=1.5,
    arrowsize=15,
    edge_color='darkgreen',
    connectionstyle='arc3,rad=0.1' # asta i pentru ca edge urile sa fie curved
)

nx.draw_networkx_edges(
    Full_HMM_Graph,
    overall_pos,
    edgelist=[(u, v) for u, v, d in Full_HMM_Graph.edges(data=True) if d.get('relation') == 'emits'],
    width=1.0,
    arrowsize=10,
    edge_color='purple',
    style='dashed' 
)


nx.draw_networkx_edge_labels(
    Full_HMM_Graph,
    overall_pos,
    edge_labels=transition_labels,
    font_color='darkgreen',
    font_size=9
)


nx.draw_networkx_edge_labels(
    Full_HMM_Graph,
    overall_pos,
    edge_labels=emission_labels,
    font_color='purple',
    font_size=8,
    label_pos=0.7 
)

plt.title("Full Hidden Markov Model Diagram (States, Transitions, Emissions)", size=18)
plt.axis('off')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', label='Hidden States', markerfacecolor='skyblue', markersize=15),
    plt.Line2D([0], [0], marker='o', color='w', label='Observations', markerfacecolor='lightcoral', markersize=10),
    plt.Line2D([0], [0], color='darkgreen', lw=2, label='Transitions (P(s_t|s_{t-1}))', linestyle='-', marker='None'),
    plt.Line2D([0], [0], color='purple', lw=2, label='Emissions (P(o_t|s_t))', linestyle='--', marker='None')
], loc='upper left')

out_path_full_hmm = 'lab5/ex1_full_hmm_model.png'
plt.savefig(out_path_full_hmm, dpi=200, bbox_inches='tight')
print(f'Full HMM model graph saved to: {out_path_full_hmm}')

plt.show()
plt.close()


# --- BONUS: Manual Viterbi Algorithm Implementation ---
print("\n--- Bonus: Manual Viterbi Implementation ---")

def viterbi_manual(obs, states, start_p, trans_p, emit_p):
   
    N = len(states)
    T = len(obs)

    log_start_p = np.log(start_p + 1e-15)
    log_trans_p = np.log(trans_p + 1e-15)
    log_emit_p = np.log(emit_p + 1e-15)

    T1 = np.zeros((N, T))       
    T2 = np.zeros((N, T), dtype=int)  

    for i in states:
        T1[i, 0] = log_start_p[i] + log_emit_p[i, obs[0]]
        T2[i, 0] = -1 

    for t in range(1, T):
        for j in states:
            prev_probs = T1[:, t-1] + log_trans_p[:, j]
            T2[j, t] = np.argmax(prev_probs)
            T1[j, t] = prev_probs[T2[j, t]] + log_emit_p[j, obs[t]]

    best_last_state = np.argmax(T1[:, T-1])
    best_path_log_prob = T1[best_last_state, T-1]

    path = np.zeros(T, dtype=int)
    path[T-1] = best_last_state
    for t in range(T-2, -1, -1):
        path[t] = T2[path[t+1], t+1]

    return path, best_path_log_prob

manual_path_indices, manual_log_prob = viterbi_manual(
    obs=obs_sequence,
    states=[0, 1, 2],
    start_p=start_prob,
    trans_p=trans_matrix,
    emit_p=emission_matrix
)

manual_path_names = [states[i] for i in manual_path_indices]

print(f"Manual Viterbi Path:\n{manual_path_names}")
print(f"Manual Viterbi Log-Prob: {manual_log_prob:.3f}")
 
 #aici verificam daca rezultatele coincid
print("\n verificare") 
if np.all(hidden_state_indices == manual_path_indices):
    print("Manual Viterbi matches hmmlearn.decode() result!")
else:
    print(" Paths differ (possible due to equal-probability ties).")
    print("hmmlearn path:", hidden_state_names)
    print("manual path  :", manual_path_names)

if np.isclose(log_prob_viterbi, manual_log_prob):
    print(" Log-probabilities match.")
else:
    print(" Log-probabilities differ:", log_prob_viterbi, manual_log_prob)

