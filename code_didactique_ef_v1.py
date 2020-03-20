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
nb_nodes = 102
#stiffness = 100e9 # tipically for a steel
stiffness = 1
use_sparse = True

class UniformMesh1D:
    """
    A uniform 1D mesh with nb_nodes nodes equally spaced between
    x_min and x_max.
    """
    def __init__(self, x_min, x_max, nb_nodes):
        self.nb_dof = nb_nodes
        self.node_coordinates = np.linspace(x_min, x_max, nb_nodes)
        self.elements = np.array([np.arange(0, nb_nodes - 1, dtype=int),
                                  np.arange(1, nb_nodes,     dtype=int)]).T

    def shape_functions(self, xi):
        """
        The chosen convention is the reference element between [-1,1].
        xi is the local coordinate in the element
        """
        return [(1. - xi) / 2, # shape_function f0 -> f0(-1) = 1., f0(1) = 0.
                (1. + xi) / 2] # shape_function f1 -> f1(-1) = 0., f1(1) = 1.

    def shape_functions_prime(self, xi):
        """
        The chosen convention is the reference element between [-1,1].
        xi is the local coordinate in the element
        """
        return [-1./2, # f0'
                +1./2] # f1'

    integration_points_order1 = [[0,2],]
    integration_points_order3 = [[-0.577350269189625, 1], [+0.577350269189625, 1]]
    integration_points_order5 = [[-0.774596669241483, 0.555555555555556],
                                 [0.0, 0.888888888888889],
                                 [+0.774596669241483, 0.555555555555556]]
    integration_points_order7 = [[-0.861136311594052, 0.347854845137454],
                                 [-0.339981043584856, 0.652145154862545],
                                 [+0.861136311594052, 0.347854845137454],
                                 [+0.339981043584856, 0.652145154862545]]

class HookeMaterial:
    def __init__(self, stiffness):
        self.stiffness = stiffness

    def sigma(self, epsilon):
        return self.stiffness * epsilon

    def elasticity_tensor(self, epsilon):
        return self.stiffness

class NonLinearMaterial:
    def __init__(self, stiffness, U0, k):
        self.stiffness = stiffness
        self.U0, self.k = U0, k

    def sigma(self, epsilon):
        return self.stiffness*(epsilon + 3*k*epsilon**2)

    def elasticity_tensor(self, epsilon):
        return stiffness*(1 + 6*k*epsilon)

class NonLinearTestProblem:
    def __init__(self, stiffness, U0, k):
        self.material = NonLinearMaterial(stiffness, U0, k)
        self.U0, self.k = U0, k

    def volumic_force(self, x):
        U0, k = self.U0, self.k
        return 6*x*U0 - 6*k*(1 - 3*x**2)*(-6*x)*U0**2

    def true_solution(self, x):
        return self.U0 * (x - x**3)

def sinusoidal_volumic_force(x):
    return np.sin(np.pi*x)

def assemble(K, F, mesh, u0, integration_points,
             material, volumic_force):
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
    # We discretize u using the Ritz method:
    # u = \sum u_i \phi_i
    # using the same \phi as test function.
    for nodes_in_element in mesh.elements:
        x_left, x_right = (mesh.node_coordinates[node] for node in nodes_in_element)
        J = 0.5*(x_right - x_left)
        for xi, weight in integration_points:
            phi = mesh.shape_functions(xi)
            dphi_dx = [df_dx/J for df_dx in mesh.shape_functions_prime(xi)]
            nodes_and_shape_functions = zip(nodes_in_element, phi, dphi_dx)

            du0_dx = sum(u0[node_i] * dphi_i_dx
                         for node_i, phi_i, dphi_i_dx in nodes_and_shape_functions)
            E = material.elasticity_tensor(du0_dx)
            for node_i, phi_i, dphi_i_dx in nodes_and_shape_functions:
                for node_j, phi_j, dphi_j_dx in nodes_and_shape_functions:
                    K[node_j, node_i] += weight * E * dphi_i_dx * dphi_j_dx * J

            # We add -\int u0 \phi_i to F_i
            sigma = material.sigma(du0_dx)
            for node_j, phi_j, dphi_j_dx in nodes_and_shape_functions:
                F[node_j] -= weight * sigma * dphi_j_dx * J

            # We find F by calculating
            # F_i = \int_\omega f*\phi_i
            x = x_left + 0.5*(xi+1)*(x_right - x_left)
            f = volumic_force(x)
            for node_j, phi_j, dphi_j_dx in nodes_and_shape_functions:
                F[node_j] += weight * f * phi_j * J

    # Add boundary conditions
    # Dirichlet on ddl 0, i.e. node 0
    K[0, :] = 0
    K[0, 0] = 1
    F[0] = 0

    # Dirichlet on last ddl, i.e. last node
    K[-1, :] = 0
    K[-1, -1] = 1
    F[-1] = 0

def solve_algebraic(K, F):
    check = False
    if use_sparse:
        du = spsolve(K, F)
        if check:
            ducheck = np.linalg.solve(K.toarray(), F)
            print("Error on linalg: {}".format(np.linalg.norm(du-ducheck)))
    else:
        du = np.linalg.solve(K, F)
    return du

def init_matrix(nb_dof):
    """
    In the case of a sparse matrix, initialize approximatively the structure of the matrix.
    """
    K = diags((np.zeros(nb_dof-1),
               np.zeros(nb_dof),
               np.zeros(nb_dof-1)), offsets=[-1, 0, 1])
    K = lil_matrix(K, dtype='d')
    return K

def solve_non_linear_problem(mesh, u, integration_points,
                             material, volumic_force):
    convergence = list()
    # Initialise stiffness matrix and force vector
    if use_sparse:
        K = init_matrix(mesh.nb_dof)
    else:
        K = np.zeros((mesh.nb_dof, mesh.nb_dof), dtype='d')
    F = np.zeros(mesh.nb_dof, dtype='d')
    # Print title of output log columns
    print("Iter  N_2(u)       N_inf(u)     T_ass      T_solv")
    # Loop on Newton iterations
    for newton_iter in range(0, 50):
        # Define and fill stiffness matrix and force vector
        t0 = timeit.time.time()
        assemble(K, F, mesh, u, integration_points,
                 material, volumic_force)
        t1 = timeit.time.time()
        du = solve_algebraic(K, F)
        t2 = timeit.time.time()
        # Add correction
        u += du
        # Compute norms of correction and convergence criteria
        du_norm2 = np.linalg.norm(du, ord=2)
        du_normm = np.linalg.norm(du, ord=np.Inf)
        print("%4i  %3.5e  %3.5e  %2.3e, %2.3e" % (newton_iter, du_norm2, du_normm, t1-t0, t2-t1))
        convergence.append((du_norm2, du_normm))
        if du_normm < 1e-10:
            break
    return np.array(convergence)


def plot_convergence(conv):
    fig = plt.figure()
    plt.plot(np.arange(len(conv)), conv[:,0], "+-k", label="norm_2(u)")
    plt.plot(np.arange(len(conv)), conv[:,1], "+--k", label="norm_inf(u)")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close(fig)

def plot(mesh, u, u_compare=None):
    x = mesh.node_coordinates
    plt.plot(x, u, "+--k", label="u")
    #plt.plot(node_coordinates, np.sin(np.pi*node_coordinates)/np.pi/np.pi, "+--b", label="sin(x)")
    if u_compare is not None:
        plt.plot(x, u_compare(x), "-b", label="Solution")
    plt.legend()
    plt.show()

def run():
    print("Start program...")
    t0 = timeit.time.time()
    # Generate the mesh and initialize fields
    mesh = UniformMesh1D(x_min, x_max, nb_nodes)
    u = np.zeros(mesh.nb_dof, dtype=float)
    problem = NonLinearTestProblem(stiffness, U0, k)
    convergence = solve_non_linear_problem(mesh, u, mesh.integration_points_order3,
#                                          HookeMaterial, sinusoidal_volumic_force)
                                           problem.material, problem.volumic_force)
    t1 = timeit.time.time()
    plot_convergence(convergence)
    plot(mesh, u, u_compare=problem.true_solution)
    print("Done in %f seconds" % (t1-t0))

if __name__ == "__main__":
    run()
