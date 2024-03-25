#!/usr/bin/python3

from math import exp
import numpy

# Reads a |V| x |V| adjacency matrix, line-by-line
# Returns a list of lists [[edges of node 1], [edges of node 2], ..., [edges of node |V|]]
def read_graph():
    A = [[int(_) for _ in input().split()]]
    for _ in range(len(A[0])-1):
        A.append([int(_) for _ in input().split()])

    return A

# Reads a list of weights of size |k|, all on one line
# Returns a list of numbers [wx1, wx2, ..., wx|k|]
def read_weights():
    return [int(_) for _ in input().split()]

# Potential function over a single vertex, using list of weights
# Returns exp(weight of color assigned to vertex xi)
def single_potential(w, xi):
    return exp(w[xi-1])

# Potential function over pairs of vertices, using color assignments made to vertices
# Indicator function - 1 if both vertices as different colors, 0 otherwise
def paired_potential(xi, xj):
    return 1 if xi != xj else 0

# Returns a |k| x |k| matrix of product of potentials
def get_product_of_potentials(from_vertex, to_vertex, w):
    potentials = numpy.empty((len(w), len(w)), dtype='float64')

# CHANGING THIS
    for to_vertex_color in range(1, len(w)+1):
        for from_vertex_color in range(1, len(w)+1):
            potentials[to_vertex_color-1][from_vertex_color-1] \
                    = single_potential(w, from_vertex_color) \
                    * paired_potential(from_vertex_color, to_vertex_color)

    return potentials

def get_product_of_messages(skip_vertex, to_vertex, w, messages, A):
    product_of_messages = numpy.ones((len(w)), dtype='float64')

    for from_vertex in range(len(A)):
        if skip_vertex == from_vertex or A[from_vertex][to_vertex] == 0:
            continue

        product_of_messages *= messages[from_vertex][to_vertex]

    return product_of_messages

def sumprod(A, w, its):
    messages = numpy.ones((len(A), len(A), len(w)), dtype='float64')

    for iteration in range(1, its+1):
        for to_vertex in (range(len(A))):
            for from_vertex in range(len(A)):
                if A[to_vertex][from_vertex] == 0:
                    continue

                # phi_i(xi) * psi_ij(xi, xj)
                product_of_potentials = get_product_of_potentials(from_vertex, to_vertex, w) # k x k matrix
                # prod_k->N(i)\j_(mk->i(xi))
                product_of_messages = get_product_of_messages(to_vertex, from_vertex, w, messages, A) # k x 1 matrix
                # m_i->j(xj)
                messages[from_vertex][to_vertex] = numpy.ndarray.flatten(\
                        numpy.dot(product_of_potentials, product_of_messages)) # k x 1 matrix
                
                messages[from_vertex][to_vertex] /= numpy.sum(messages[from_vertex][to_vertex])

    single_beliefs = get_single_beliefs(messages, A, w)
    pairwise_beliefs = get_pairwise_beliefs(messages, A, w)

    '''
    for vertex in range(len(A)):
        print('single beliefs for vertex {}: {}'.format(vertex+1, single_beliefs[vertex]))
    for first in range(len(A)):
        for second in range(first+1, len(A)):
            if A[first][second] == 0:
                continue

            print('pairwise beliefs for edge {}, {}: \n{}'.format(first+1, second+1, pairwise_beliefs[first][second]))
    '''

    Z = calculate_bethe(single_beliefs, pairwise_beliefs, A, w)
        
    return Z

def calculate_bethe(single_beliefs, pairwise_beliefs, A, w):
    single_potentials = numpy.array([single_potential(w, color) for color in range(1, len(w)+1)], dtype='float64')
    logged_single_potentials = numpy.log(single_potentials, out=numpy.zeros_like(single_potentials), where=(single_potentials!=0))
    '''
    paired_potentials = numpy.array([[single_potentials[second_color] * single_potentials[first_color]\
            * paired_potential(first_color, second_color) for second_color in range(len(w))]\
            for first_color in range(len(w))], dtype='float64')
    '''
    paired_potentials = numpy.array([[paired_potential(first_color, second_color) for second_color in range(len(w))]\
            for first_color in range(len(w))], dtype='float64')
 
    logged_paired_potentials = numpy.log(paired_potentials, out=numpy.zeros_like(paired_potentials), where=(paired_potentials!=0))

    paired_term = 0
    degree = numpy.zeros((len(A)), dtype='int64')

    for first in range(len(A)):
        for second in range(first+1, len(A)):
            if A[first][second] == 0:
                continue

            degree[first] += 1
            degree[second] += 1

            for first_color in range(1, len(w)+1):
                for second_color in range(1, len(w)+1):
                    value = pairwise_beliefs[first][second][first_color-1][second_color-1]
                    
                    E = -logged_paired_potentials[first_color-1][second_color-1]\
                            -logged_single_potentials[first_color-1]\
                            -logged_single_potentials[second_color-1]

                    lnb = 0 if value == 0 else numpy.log(value)

                    value *= (E + lnb)
                    paired_term += value

    
    single_term = 0
    for vertex in range(len(A)):
        for color in range(1, len(w)+1):
            value = single_beliefs[vertex][color-1]
            
            E = -logged_single_potentials[color-1]
            
            lnb = 0 if value == 0 else numpy.log(value)
            
            value = (degree[vertex]-1) * (value * (E + lnb))
            single_term += value

    bethe_approximation = exp(-paired_term + single_term)
    return bethe_approximation

def get_pairwise_beliefs(messages, A, w):
    pairwise_beliefs = numpy.zeros((len(A), len(A), len(w), len(w)), dtype='float64')
    single_potentials = numpy.array([single_potential(w, color) for color in range(1, len(w)+1)], dtype='float64')
    for first in range(len(A)):
        for second in range(first+1, len(A)):
            if A[first][second] == 0:
                continue

            first_messages = get_product_of_messages(second, first, w, messages, A)
            second_messages = get_product_of_messages(first, second, w, messages, A)
            pairwise_belief = numpy.zeros((len(w), len(w)), dtype='float64')
            for first_color in range(1, len(w)+1):
                for second_color in range(1, len(w)+1):
                    pairwise_belief[first_color-1][second_color-1] = single_potential(w, first_color)\
                            * single_potential(w, second_color)\
                            * paired_potential(first_color, second_color)\
                            * first_messages[first_color-1]\
                            * second_messages[second_color-1]

            pairwise_beliefs[first][second] = pairwise_belief / numpy.sum(pairwise_belief)
        
    return pairwise_beliefs

def get_single_beliefs(messages, A, w):
    single_beliefs = numpy.zeros((len(A), len(w)), dtype='float64')
    single_potentials = numpy.array([single_potential(w, color) for color in range(1, len(w)+1)], dtype='float64')
    for vertex in range(len(A)):
        product_of_messages = numpy.ones(len(w), dtype='float64')

        for neighbor in range(len(A)):
            if A[vertex][neighbor] == 0:
                continue

            product_of_messages *= messages[neighbor][vertex]

        beliefs = single_potentials * product_of_messages
        single_beliefs[vertex] = beliefs / numpy.sum(beliefs)

    return single_beliefs

def product_of_messages_into_vertex(vertex, messages, A, w):
    product_of_messages = numpy.ones(len(w), dtype='float64')

    for neighbor in range(len(A)):
        if A[vertex][neighbor] == 0:
            continue

        product_of_messages *= messages[neighbor][vertex]

    return product_of_messages

def main():
    A = read_graph()
    w = read_weights()
    its = int(input())

    Z = sumprod(A, w, its)
    print('Z: {}'.format(Z))

if __name__ == '__main__':
    main()
