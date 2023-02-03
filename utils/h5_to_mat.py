"""
.. _IMPORTING_dataset_form_FEniCS_solution:

IMPORTING dataset form FEniCS solution - .h5 to .mat

This script is used to import a FEniCS solution in .h5 format and convert it to a .mat file.

Dependencies

h5py
numpy
scipy

Functions
    extract_edges(triangulation)
    Given a triangulation, this function returns a list of edges in the triangulation.
"""

import h5py
import numpy as np
from scipy.io import savemat

def extract_edges(triangulation):
    edges = set()
    for triangle in triangulation:
        for i in range(3):
            edge = (triangle[i], triangle[(i + 1) % 3])
            edges.add(tuple(sorted(edge)))
    return list(edges)

# These lines define and open the path of the .h5 file
h5_file = "../data/solution_hf_advection.h5"
f = h5py.File(h5_file)

# This line calculates the number of degrees of freedom
dof = f["/Mesh/0/mesh/geometry"].shape[0]

# This for loop iterates through all the mesh elements in the .h5 file
for i in range(len(f["Mesh"])):

    # These lines extract the mesh information, x and y coordinates, and the solution vector
    mesh = f["Mesh/"+str(i)+"/mesh/geometry"]
    triang = np.array(f["/Mesh/"+str(i)+"/mesh/topology"]) + 1
    x = mesh[:, 0:1]
    y = mesh[:, 1:2]
    u = np.array(f["VisualisationVector/"+str(i)])

    try:
        xx = np.concatenate([xx, x], axis=1)
        yy = np.concatenate([yy, y], axis=1)
        solution = np.concatenate([solution, u], axis=1)
    except:
        xx = x
        yy = y
        solution = u

# These lines extract the edges from the triangulation
edges = np.array(extract_edges(triang))
edges = edges[edges[:, 1].argsort()]
edges = edges[edges[:, 0].argsort(kind='mergesort')]

# These lines create a dictionary that stores the triangulation, edges, number of degrees of freedom, solution, x and y coordinates
mat_file = "../data/file.mat"
dataset = dict()
dataset['T'] = triang.astype('float')
dataset['E'] = edges.astype('float')
dataset['dof'] = float(dof)
dataset['U'] = solution
dataset['xx'] = xx
dataset['yy'] = yy

# This line saves the dataset as a .mat file
savemat(mat_file, dataset)