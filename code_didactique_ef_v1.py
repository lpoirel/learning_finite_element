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
nb_nodes = 50
force = 0.1
#stiffness_coefficient = 100e9 # tipically for a steel
stiffness_coefficient = 1

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
    shape = [(1.-xi)/2, (1.+xi)/2]
    return shape[index_node]


def shape_function_prime_1d(index_node, xi):
    """
    The choosen convention is the reference element between [-1,1].
    index_node is the local index inside the element.
    """
    shape = [-1./2, 1./2]
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

def volumic_force(x):
    return np.sin(np.pi*x)

def assemble(mesh, element_connect, u0, integration_points):
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
    for nodes_in_element in element_connect:
        J = 0.5*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
        for i_local, i in enumerate(nodes_in_element):
            for j_local, j in enumerate(nodes_in_element):
                for int_point in integration_points:
                    xi = int_point[0]
                    weight = int_point[1]
                    Dphi_i = shape_function_prime_1d(i_local, xi)/J
                    Dphi_j = shape_function_prime_1d(j_local, xi)/J
                    K[j, i] += weight*stiffness_coefficient*Dphi_i*Dphi_j*J;

    # We find F by calculating
    # F_i = \int_\omega f*\phi_i
    for nodes_in_element in element_connect:
        J = 0.5*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
        for i_local, i in enumerate(nodes_in_element):
            for int_point in integration_points:
                xi = int_point[0]
                weight = int_point[1]
                x = mesh[nodes_in_element[0]] + 0.5*(xi+1)*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
                f = volumic_force(x)
                phi_i = shape_function_1d(i_local, xi)
                F[i] += weight*f*phi_i*J;

    # We add -\int u0 \phi_i to F_i
    for nodes_in_element in element_connect:
        J = 0.5*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
        for i_local, i in enumerate(nodes_in_element):
            for j_local, j in enumerate(nodes_in_element):
                for int_point in integration_points:
                    xi = int_point[0]
                    weight = int_point[1]
                    Dphi_i = shape_function_prime_1d(i_local, xi)/J
                    Dphi_j = shape_function_prime_1d(j_local, xi)/J
                    F[j] -= weight*stiffness_coefficient*Dphi_i*u0[i]*Dphi_j*J;

    # Add boundary conditions
    for j in range(0, nb_nodes):
        K[0, j] = 0
    K[0,0] = 1
    for j in range(0, nb_nodes):
        K[nb_nodes-1, j] = 0
    K[nb_nodes-1, nb_nodes-1] = 1

    F[0] = 0
    F[nb_nodes-1] = 0
#    F[-1] = force
    return K, F


def solve_algebric(K, F):
    du = np.linalg.solve(K, F)
    return du


def plot(mesh, u):
    plt.plot(mesh, u, "+--k", label="u")
#    plt.plot(mesh, np.sin(np.pi*mesh)/np.pi/np.pi, "+--b", label="sin(x)")
    plt.show()


def run():
    print("Start program")
    # Generate the mesh and initialize fields
    mesh, element_connect = generate_mesh()
#    u = np.zeros(mesh.shape, dtype=float)
    # Create a random u, and make sure it satisfies u(0)=u(1)=0
    u = np.random.normal(size=mesh.shape)
    u[0] = 0
    u[nb_nodes-1] = 0
    # Define and fill stiffness matrix and force vector
    K, F = assemble(mesh, element_connect, u, integration_points_order3)
    du = solve_algebric(K, F)
    u = u+du
    plot(mesh, u)

if __name__ == "__main__":
    run()
