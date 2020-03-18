# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:42:55 2020

One element and linear problem

@author: SIM/ER
"""

import numpy as np
import matplotlib.pyplot as plt


# Parameters
x_min = 0.
x_max = 1.
nb_nodes = 2
force = 0.1
stiffness_coefficient = 100e9 # tipically for a steel

integration_points_order1 = [[0,2],]
integration_points_order3 = [[-0.577350269189625, 1], [+0.577350269189625, 1]]
integration_points_order5 = [[-0.774596669241483, 0.555555555555556],
                             [0.0, 0.888888888888889],
                             [+0.774596669241483, 0.555555555555556]]
integration_points_order7 = [[-0.861136311594052, 0.347854845137454],
                             [-0.339981043584856, 0.652145154862545],
                             [+0.861136311594052, 0.347854845137454],
                             [+0.339981043584856, 0.652145154862545]]


# ========== Shape funtions ==========

def shape_function_1d(index_node, xi):
    """
    The choosen convention is the reference element between [-1,1].
    index_node is the local index inside the element.
    xi is the local coordinate in the element
    """
    shape = [(1.+xi)/2, (1.-xi)/2]
    return shape[index_node]


def shape_function_prime_1d(index_node):
    """
    The choosen convention is the reference element between [-1,1].
    index_node is the local index inside the element.
    """
    shape = [1./2, -1./2]
    return shape[index_node]


def generate_mesh():
    """
    Generate the mesh with numpy for a 1d beam with nb_nodes nodes.
    """
    mesh = np.linspace(x_min, x_max, nb_nodes)
    element_connect = np.array([np.arange(0, nb_nodes-1, dtype=int), 
                                np.arange(1, nb_nodes,   dtype=int)]).T
    return mesh, element_connect


def hooke_material_law(epsilon):
    sigma = stiffness_coefficient*epsilon
    return sigma


def assemble(mesh, element_connect, u, integration_points):
    """
    Assemble matrix and vector using first order finite element.
    Element is 1d bar.
    Numerical integration is here degenerated to element length.
    """
    # Initialise stiffness matrix and force vector
    K = np.zeros((np.prod(mesh.shape), np.prod(mesh.shape)), dtype=float)
    F = np.zeros(np.prod(mesh.shape), dtype=float)
    # Fill them by looping on elements
    # The physics is div(\sigma) + f = 0
    # \int_\omega \sigma*\phi_prime = \int_\omega f*\phi
    # with \phi the test fonction and \omega the element.
    # We discretize u while using the Ritz method and
    # then u = \sum u_i \phi_i
    # using the same \phi as test function.
    # Let us define Ke as the element matrix contribution
    for index_int_point, int_point in enumerate(integration_points):
        int_point_position, int_point_weight = int_point
        epsilon = 0.
        for nodes_in_element in element_connect:
            index_node_1, index_node_2 = nodes_in_element
        
        # 
            epsilon += shape_function_prime_1d(index_int_point)*u[nodes_in_element[]]
            
            # call mat law to compute stress mesure, here Cauchy tensor in 1d
            sigma = hooke_material_law(eplison)
        
    
    # Add boundary conditions
    F[0] = 0.
    F[-1] = force
    return K, F


def solve_algebric():
    pass


def plot(mesh, u):
    plt.plot(mesh, u, "+--k", label="u")
    plt.show()


def run():
    print("Start program")
    # Generate the mesh and initialize fields
    mesh, element_connect = generate_mesh()
    u = np.zeros(mesh.shape, dtype=float)
    plot(mesh, u)
    # Define and fill stiffness matrix and force vector
    K, F = assemble(mesh, element_connect, u, integration_points_order3)


if __name__ == "__main__":
    run()
