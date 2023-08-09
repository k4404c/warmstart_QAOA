import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
from qiskit.algorithms.optimizers import SPSA


# ---------------------------------------------------------------------------------------------------------

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()


# ---------------------------------------------------------------------------------------------------------

def maxcut_obj(x, G):
    
    obj = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            obj -= 1       
    return obj

# ---------------------------------------------------------------------------------------------------------

def compute_expectation(counts, G):

    avg = 0
    sum_count = 0
    
    for bitstring, count in counts.items():
        
        obj = maxcut_obj(bitstring, G)
        avg += obj * count
        sum_count += count   
    return avg/sum_count

# ---------------------------------------------------------------------------------------------------------

def create_qaoa_circ(G, theta):
    
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    qc = QuantumCircuit(nqubits)
    
    beta = theta[:p]
    gamma = theta[p:]
    
    # initial_state
    for i in range(0, nqubits):
        qc.h(i)
    
    for irep in range(0, p):
        
        # problem unitary
        for pair in list(G.edges()):
            #qc.cnot(pair[0], pair[1])
            qc.rzz(2 * gamma[irep], pair[0], pair[1])
            #qc.cnot(pair[0], pair[1])

        # mixer unitary
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)
     
    qc.measure_all()
    return qc  

# ---------------------------------------------------------------------------------------------------------
import random
def get_expectation(G, p, seed, shots=1024):
        
    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots
    #theta is initial beta gamma values!
    def execute_circ(theta):
        qc = create_qaoa_circ(G, theta)
        counts = backend.run(qc, seed_simulator=seed, nshots=1024).result().get_counts()
        return compute_expectation(counts, G)
    return execute_circ

# ---------------------------------------------------------------------------------------------------------
def np_solver(G, nodes):
    w = np.zeros([nodes, nodes])
    for i in range(nodes):
        for j in range(nodes):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = 1

    xbest_brute = ""
    best_cost_brute = 0
    for b in range(2**nodes):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(nodes)))]
        cost = 0
        for i in range(nodes):
            for j in range(nodes):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
        if best_cost_brute < cost:
            best_cost_brute = cost
            xbest_brute = x
    print("\nBest solution brute = " + str(xbest_brute) + " cost = " + str(best_cost_brute))

# ---------------------------------------------------------------------------------------------------------

def cost_given_string(G, nodes, sequence):
    w = np.zeros([nodes, nodes])
    for i in range(nodes):
        for j in range(nodes):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = 1
    x = []
    ''.join(sequence)
    for k in range(nodes):
        x.append(int(sequence[k]))
    cost = 0
    for i in range(nodes):
            for j in range(nodes):
                cost = cost + w[i, j] * x[i] * (1 - x[j])
    return cost
# ---------------------------------------------------------------------------------------------------------

####################complete_graph###########################################
seed = random.randint(1,1000)
print("\nseed: ", seed)

G = nx.Graph()
colors = ["green" for node in G.nodes()]
pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, "weight")

nodes = 5 #if p = (nodesMOD2 == 0)? :2 , 1   --> for complete graph
graph = "line"
elist = [(0 , 1 , 1.0) , (0 , 2 , 1.0) , (0 , 3 , 1.0) , (0 , 4 , 1.0) , (1 , 2 , 1.0) , (1 , 3 , 1.0) , (1 , 4 , 1.0) , (2 , 3 , 1.0) , (2 , 4 , 1.0) , (3 , 4 , 1.0)]
#[(0, 1, 1.0), (0,2,1.0),(0,3,1.0), (1,2,1.0),(1,3,1.0),(2,3,1.0)]

G.add_weighted_edges_from(elist)

colors = ["lightgreen" for node in G.nodes()]
pos = nx.spring_layout(G)
draw_graph(G, colors, pos)

np_solver(G,nodes)

expectation = get_expectation(G, p=1,seed= seed)
pi = 3.14159

x0 = [1.0,1.0] #starting gamma beta parameters


spsa = SPSA(maxiter=500)

res = SPSA.minimize(self = spsa, fun = expectation, x0 = x0, jac = None, bounds = [-1*pi,pi])
x0 =  res.x
print("Beta, Gamma:  ", res.x) #gamma beta values for solution



backend = Aer.get_backend('aer_simulator')
backend.shots = 512
qc_res = create_qaoa_circ(G, x0)
counts = backend.run(qc_res, seed_simulator=seed).result().get_counts()


plot_histogram(counts)  
#peaks of graph are likely all possible solutions to complete graph.
plt.show()

bestSol = ""+counts.most_frequent()
print("Best Solution QAOA: ", bestSol, "cost: ", cost_given_string(G,nodes,bestSol))
#########################################################

#############incomplete-similar-graph########################
G = nx.Graph()
colors = ["green" for node in G.nodes()]
pos = nx.spring_layout(G)
edge_labels = nx.get_edge_attributes(G, "weight")


graph = "line"

elist = [(0, 1, 1.0),(0,4,1.0), (1,2,1.0),(1,4,1.0),(2,3,1.0) ,(2,4,1.0), (3,4,1.0)]
#[(0, 1, 1.0), (0, 2, 1.0),(0, 3, 1.0),(1,2,1.0),(2,3,1.0)]

G.clear()
G.add_weighted_edges_from(elist)


np_solver(G,nodes)

#gradient-decent BFGS

expectation = get_expectation(G, p=3,seed = seed)

                                        
#COBYLA

res = SPSA.minimize(self = spsa, fun = expectation, x0 = x0, jac = None, bounds = [(-1)*pi, pi])
#res = minimize(expectation, x0 , method= "BFGS")

#i had issues implementing BFGS so opted for SPSA, non gradient though handles noise well like BFGS

x0 = res.x
print("Final Beta, Gamma: ", x0)

backend = Aer.get_backend('aer_simulator')
backend.shots = 1024
qc_res = create_qaoa_circ(G, res.x)
counts = backend.run(qc_res, seed_simulator=seed).result().get_counts()

plot_histogram(counts)
plt.show()

pos = nx.spring_layout(G)
bestSol = ""+counts.most_frequent()
print("Best Solution QAOA: ", bestSol, "cost: ", cost_given_string(G,nodes,bestSol))
colors = ["r" if digit == "0" else "b" for digit in bestSol]
draw_graph(G, colors, pos)

#############################################################

'''
SOURCES USED:

https://qiskit.org/documentation/stubs/qiskit.algorithms.optimizers.SPSA.html
https://qiskit.org/textbook/ch-applications/qaoa.html
https://arxiv.org/abs/2205.10383


'''