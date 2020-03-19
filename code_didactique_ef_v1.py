# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:42:55 2020

One element and linear problem

@author: SIM/ER

solves {min 0.5*u'^2 + k*u'^3 - fu}, with u(0)=u(1)=0 

Variational formulation: u'*v' + 3k*u'^2*v' = fv , for any v

ODE: -u'' - 3k (u'^2)' = f

Linear problem:
    Find correction du such that
    du'*v'+ 6k*u0'*du'*v' = fv - u0'*v' -3k*u0'^2*v'


Verification:
    The solution is u=U0(x-x^3). u'=U0(1-3x^2). u''=-6x*U0
	f = 6*U0*x - 3k*U0^2*(1-3x^2)^2' = 6*U0*x - 6*U0^2*k*(1-3x^2)*(-6x)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import timeit

k = 0.25
U0 = 0.3

# Parameters
x_min = 0.
x_max = 1.
nb_nodes = 10002
force = 0.1
#stiffness_coefficient = 100e9 # tipically for a steel
stiffness_coefficient = 1
use_sparse = True

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

def other_non_linear_material_law(epsilon):
    return epsilon*epsilon

def volumic_force(x):
#    return np.sin(np.pi*x)
    #return -6*x - 3*k*9*4*x*x*x
    return 6*x*U0 - 6*k*(1-3*x*x)*(-6*x)*U0*U0

def assemble(K, F, mesh, element_connect, u0, integration_points,
             mat_law=hooke_material_law):
    """
    Assemble matrix and vector using first order finite element.
    Element is 1d bar.
    Numerical integration is here degenerated to element length.
    """
    K *= np.finfo(np.float32).tiny # trick for sparse matrix lil
    F *= 0.
    # Fill them by looping on elements
    # The physics is div(\sigma) + f = 0
    # \int_\omega \sigma*\phi_prime = \int_\omega f*\phi
    # with \phi the test fonction and \omega the element.
    # We discretize u while using the Ritz method and
    # then u = \sum u_i \phi_i
    # using the same \phi as test function.
    for nodes_in_element in element_connect:
        J = 0.5*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
        for int_point in integration_points:
            xi = int_point[0]
            weight = int_point[1]

            du0_dx = 0.
            for i_local, i in enumerate(nodes_in_element):
                du0_dx += shape_function_prime_1d(i_local, xi)/J*u0[i]

            for i_local, i in enumerate(nodes_in_element):
                for j_local, j in enumerate(nodes_in_element):
                    Dphi_i = shape_function_prime_1d(i_local, xi)/J
                    Dphi_j = shape_function_prime_1d(j_local, xi)/J
                    E = (stiffness_coefficient+6*k*du0_dx)
                    K[j, i] += weight*E*Dphi_i*Dphi_j*J

            # We add -\int u0 \phi_i to F_i
            for j_local, j in enumerate(nodes_in_element):
                Dphi_j = shape_function_prime_1d(j_local, xi)/J
#                sigma = hooke_material_law(du_dx)
                sigma = du0_dx + 3*k*du0_dx*du0_dx
                F[j] -= weight*sigma*Dphi_j*J

            # We find F by calculating
            # F_i = \int_\omega f*\phi_i
            x = mesh[nodes_in_element[0]] + 0.5*(xi+1)*(mesh[nodes_in_element[1]]-mesh[nodes_in_element[0]])
            for j_local, j in enumerate(nodes_in_element):
                f = volumic_force(x)
                phi_j = shape_function_1d(j_local, xi)
                F[j] += weight*f*phi_j*J


    # Add boundary conditions
    # Dirichlet on ddl 0, i.e. node 0
    K[0, :] = 0
    K[0, 0] = 1
    # Dirichlet on last ddl, i.e. last node
    K[-1, :] = 0
    K[-1, -1] = 1

    # Neumann on ddl 0 and last ddl, i.e. node 0 and last node
    F[0] = 0
    F[-1] = 0


def solve_algebric(K, F):
    check = False
    if use_sparse:
        du = spsolve(K, F)
        if check:
            ducheck = np.linalg.solve(K.toarray(), F)
            print "Error on linalg: ", np.linalg.norm(du-ducheck)
    else:
        du = np.linalg.solve(K, F)
    return du


def init_matrix(mesh_shape):
    """
    In the case of a sparse matrix, initialize approximatively the structure of the matrix.
    """
    K = diags((np.zeros(np.prod(mesh_shape)-1),
               np.zeros(np.prod(mesh_shape)),
               np.zeros(np.prod(mesh_shape)-1)), offsets=[-1, 0, 1])
    K = lil_matrix(K, dtype='d')
    return K


def solve_non_linear_problem(mesh, element_connect, u, integration_points, mat_law):
    convergence = list()
    # Initialise stiffness matrix and force vector
    if use_sparse:
        K = init_matrix(mesh.shape)
    else:
        K = np.zeros((np.prod(mesh.shape), np.prod(mesh.shape)), dtype='d')
    F = np.zeros(np.prod(mesh.shape), dtype='d')
    for iter in range(0, 50):
        # Define and fill stiffness matrix and force vector
        t0 = timeit.time.time()
        assemble(K, F, mesh, element_connect, u, integration_points_order3, mat_law)
        t1 = timeit.time.time()
        du = solve_algebric(K, F)
        t2 = timeit.time.time()
        u += du
        #plot(mesh, u)
        du_norm2 = np.linalg.norm(du, ord=2)
        du_normm = np.linalg.norm(du, ord=np.Inf)
        print (du_norm2, du_normm, t1-t0, t2-t1)
        convergence.append((du_norm2, du_normm))
        if du_normm < 1e-10:
            break
    return convergence


def plot_convergence(conv):
    fig = plt.figure()
    plt.plot(np.arange(len(conv)), conv, "+-k", label="convergence")
    plt.yscale("log")
    plt.show()
    plt.close(fig)


def plot(mesh, u):
    plt.plot(mesh, u, "+--k", label="u")
#    plt.plot(mesh, np.sin(np.pi*mesh)/np.pi/np.pi+mesh, "+--b", label="sin(x)")
    plt.plot(mesh, U0*(mesh-mesh*mesh*mesh), "+--b", label="Solution")
    plt.show()


def run():
    print("Start program")
    t0 = timeit.time.time()
    # Generate the mesh and initialize fields
    mesh, element_connect = generate_mesh()
    u = np.zeros(mesh.shape, dtype=float)
    convergence = solve_non_linear_problem(mesh, element_connect, u,
                                           integration_points_order3,
                                           hooke_material_law)
    t1 = timeit.time.time()
    plot_convergence(convergence)
    plot(mesh, u)
    print("Done in %f seconds" % (t1-t0))

if __name__ == "__main__":
    run()
