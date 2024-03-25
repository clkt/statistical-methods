#!/usr/bin/python3

from math import exp
from collections import defaultdict, deque
import numpy

# A program to read an adjacency matrix, and exact inference
# The example chosen here is Vertex Coloring

# Reads an n x n adjacency matrix
# For example, see below for an example with node 0 as root, and nodes 1 and 2 as leaves:
# 0 1 1
# 1 0 0 
# 1 0 0
def read_graph():
    A = [[int(_) for _ in input().split()]]
    for _ in range(len(A[0])-1):
        A.append([int(__) for __ in input().split()])
    return A

# Reads weights for coloring
def read_weights():
    return [int(_) for _ in input().split()]

# Returns single potentials for MRF
# Potential function is phi_i(w[xi]), for each xi \in vertices 
def single_potential(w, xi):
    return exp(w[xi-1])

# Returns paired potentials for MRF
# Potential function is psi_ij(xi, xj), for each (xi, xj) \in edges
def paired_potential(xi, xj):
    return 1 if xi != xj else 0

# Evaluates a complete assignment
# For an n x n Adjacency matrix, this would be a vector of dimensions n x 1
# Each entry would correspond to one of 1...k colors
def evaluate_assignment(assignment, A, w, valid_assignments):
    potentials = 1
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i][j]:
                potentials *= paired_potential(assignment[i], assignment[j])

    for color in assignment:
        potentials *= single_potential(w, color)

    if (numpy.sum(potentials) > 0):
        valid_assignments.append(([_ for _ in assignment], exp(sum(assignment))))

    return potentials 

# Recursively check valid assignments
# Backtrack if assignment is invalid
# Works for any n and any k (until stack limit is reached of course)
def enumerate_next_combination(assignment, A, w, valid_assignments):
    partial_sum = 0
    for color in range(1, len(w)+1):
        if len(assignment) == len(A):
            return evaluate_assignment(assignment, A, w, valid_assignments)
        else:
            assignment.append(color)
            partial_sum += enumerate_next_combination(assignment, A, w, valid_assignments)
            assignment.pop()

    return partial_sum

# Initializing function for kicking off backtracking
# Returns partition function/normalizing constant
def exact_inference(A, w, valid_assignments):
    Z = 0
    assignment = []
    for color in range(1, len(w)+1):
        assignment.append(color)
        Z += enumerate_next_combination(assignment, A, w, valid_assignments)
        assignment.pop()

    return Z

def main():
    A = read_graph()
    w = read_weights()
    valid_assignments = []
    Z = exact_inference(A, w, valid_assignments)
    for assignment in valid_assignments:
        print(assignment[0], '{:.8f}'.format(assignment[1] / Z)) 
    print(Z)

# Entry point for code
if __name__ == '__main__':
    main()
