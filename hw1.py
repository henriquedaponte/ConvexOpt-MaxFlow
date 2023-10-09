import cvxpy as cp
import numpy as np
import time
import random

# ==================== Helper Functions ====================

# Function to compute the node-edge incidence matrix for a directed graph
def incidence_matrix(h, t, nodes):
    m = len(nodes)
    n = len(h)
    A = np.zeros((m, n))
    
    for j in range(n):
        # Finding the index of the start node (head) of edge j
        tail = nodes.index(h[j])
        # Finding the index of the end node (tail) of edge j
        head = nodes.index(t[j])
        A[tail, j] = 1
        A[head, j] = -1
        
    return A

def generate_all_connected_graph(n_nodes):
    """Generate an all-connected graph with random capacities."""
    h, t, weights = [], [], []
    for i in range(1, n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            h.append(i)
            t.append(j)
            weights.append(random.randint(1, 10))
    return h, t, weights

def computeS(h, c):
    '''
    Function to compute the column vector s 
    with a 1 for every edge leaving the source 
    and a 0 for every other edge
    '''
    s = np.zeros(len(c))

    for i in range(len(h)):
        if h[i] == 1:
            s[i] = 1

    return s

def computeL(A, source, target):
    '''
    Function to compute the matrix L 
    by removing rows corresponding to source and destination
    '''
    L = np.delete(A, [source-1, target-1], 0)

    return L

def solve_max_flow(s, L, c):
    """Solve the max flow problem for the given graph."""
    f = cp.Variable(len(c)) # Defining decision variable 

    objective = cp.Maximize(s.T @ f) # Defining the objective function

    constraints = [L @ f == 0, 0 <= f, f <= c] # Defining the constraints

    problem = cp.Problem(objective, constraints) # Defining the problem

    problem.solve() # Solving the problem

    maxFlow = round(problem.value) # Optimal value (max flow)
    maxFlowEdges = np.around(f.value) # Optimal flow on each edge


# ==================== Problem 1.a ====================

# Defining the graph (used the same graph outlined in the homework)
h = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9] # head nodes
t = [2, 4, 8, 3, 7, 4, 6, 5, 6, 8, 7, 8, 9, 3] # tail nodes
c = [10, 10, 1, 10, 1, 10, 1, 1, 12, 12, 12, 12, 6, 4] # column vector of edge capacities.
s = computeS(h, c) # column vector with a 1 for every edge leaving the source and a 0 for every other edge
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] # names of the nodes

print(s)

# Computing the node-edge incidence matrix for the graph
nodes = list(range(1, len(names) + 1))
A = incidence_matrix(h, t, nodes)


# Defining the source and destination nodes
source = 1  # 'A'
target = 9  # 'I'

# Computing L by removing rows corresponding to source and destination
L = computeL(A, source, target)

# CVXPY Max Flow Problem Formulation as linear program

maxFlow, maxFlowEdges = solve_max_flow(s, L, c)

# Printing the results
# print("Optimal value (max flow):", maxFlow)
# print("Optimal flow on each edge:", maxFlowEdges)



# ==================== Problem 1.b ====================





'''
def solve_max_flow(s, t, weights, source=1, target=None):
    """Solve the max flow problem for the given graph."""
    if not target:
        target = max(s + t)
    
    # Compute the node-edge incidence matrix
    nodes = list(range(1, len(weights) + 1))
    A_full = incidence_matrix(s, t, nodes)
    A_modified = np.delete(A_full, [source-1, target-1], 0)
    
    # CVXPY formulation for the max flow problem
    f = cp.Variable(len(weights))
    s_vector = np.zeros(len(weights))
    for i in range(len(s)):
        if s[i] == source:
            s_vector[i] = 1

    objective = cp.Maximize(s_vector.T @ f)
    constraints = [A_modified @ f == 0, 0 <= f, f <= weights]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return prob.value

# Main loop to investigate the graph size limit
n_nodes = 3
max_time = 30  # Maximum acceptable time in seconds
times = []

while True:
    s, t, weights = generate_all_connected_graph(n_nodes)
    start_time = time.time()
    solve_max_flow(s, t, weights)
    elapsed_time = time.time() - start_time
    
    times.append(elapsed_time)
    
    if elapsed_time > max_time:
        break
    n_nodes += 1

print(n_nodes, times)
'''