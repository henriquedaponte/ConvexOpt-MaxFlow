import cvxpy as cp
import numpy as np
import time
import random

# ==================== Problem 1.a ====================

# Defining the graph (used the same graph outlined in the homework)
h = [1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9] # head nodes
t = [2, 4, 8, 3, 7, 4, 6, 5, 6, 8, 7, 8, 9, 3] # tail nodes
c = [10, 10, 1, 10, 1, 10, 1, 1, 12, 12, 12, 12, 6, 4] # column vector of edge capacities.
s = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]) # column vector with a 1 for every edge leaving the source and a 0 for every other edge
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] # names of the nodes


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

# Computing the node-edge incidence matrix for the graph
nodes = list(range(1, len(names) + 1))
A = incidence_matrix(h, t, nodes)


# Defining the source and destination nodes
source = 1  # 'A'
target = 9  # 'I'

# Computing L by removing rows corresponding to source and destination
L = np.delete(A, [source-1, target-1], 0)

# CVXPY Max Flow Problem Formulation as linear program

# ASK IN OH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
f = cp.Variable(len(c)) # Defining decision variable 

objective = cp.Maximize(s.T @ f) # Defining the objective function

constraints = [L @ f == 0, 0 <= f, f <= c] # Defining the constraints

problem = cp.Problem(objective, constraints) # Defining the problem

problem.solve() # Solving the problem

# Printing the results
print("Optimal value (max flow):", round(problem.value))
print("Optimal flow on each edge:", np.around(f.value))


'''
# CVXPY Max Flow Problem Formulation

# Defining the optimization variable
f = cp.Variable(len(weights))

# Creating the s_vector
s_vector = np.zeros(len(weights))
for i in range(len(s)):
    if s[i] == source:
        s_vector[i] = 1

# Defining the objective
objective = cp.Maximize(s_vector.T @ f)

# Defining the constraints
constraints = [A_modified @ f == 0, 0 <= f, f <= weights]

# Defining the problem
prob = cp.Problem(objective, constraints)

# Solving the problem
prob.solve()

# Printing the results
print("Optimal value (max flow):", prob.value)
print("Optimal flow on each edge:", f.value)

# ==================== Problem 1.b ====================


# Function to compute the node-edge incidence matrix for a directed graph
def incidence_matrix(s, t, nodes):
    m = len(nodes)
    n = len(s)
    A = np.zeros((m, n))
    
    for j in range(n):
        # Find index of the start node (tail) of edge j
        tail = nodes.index(s[j])
        # Find index of the end node (head) of edge j
        head = nodes.index(t[j])
        A[tail, j] = 1
        A[head, j] = -1
        
    return A

def generate_all_connected_graph(n_nodes):
    """Generate an all-connected graph with random capacities."""
    s, t, weights = [], [], []
    for i in range(1, n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            s.append(i)
            t.append(j)
            weights.append(random.randint(1, 10))
    return s, t, weights

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