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
    extract_edges_3d(tetrahedron)
    Given a triangulation in 3d, this function returns a list of edges in the triangulation.
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

def extract_edges_3d(tetrahedron):
    edges = set()
    for tetra in tetrahedron:
        for i in range(4):
            for j in range(i+1, 4):
                edge = (tetra[i], tetra[j])
                edges.add(tuple(sorted(edge)))
    return list(edges)

problem_names = ["poisson", "advection", "graetz", "diffusion", "elasticity", "poiseuille", "stokes_u"]
pb = 0

# These lines define and open the path of the .h5 file
h5_file = "../dataset/h5_files/" + problem_names[pb] + ".h5"
f = h5py.File(h5_file)
mesh_str = "/Mesh"
geometry_str = "mesh/geometry"
topology_str = "mesh/topology"

keys = [key for key in f.keys()]

if "VisualisationVector" in keys:
    print("Dataset generated with FEniCS")
    function_string = "VisualisationVector/"
    dof_string =  mesh_str+"/0/"+geometry_str
    iter_str = "/Mesh"
elif "Function" in keys:
    print("Dataset generated with FEniCSx")
    function_string = "/Function/uh/"
    dof_string = mesh_str+"/"+geometry_str
    iter_str = function_string

# This line calculates the number of degrees of freedom
dof = f[dof_string].shape[0]

# This for loop iterates through all the mesh elements in the .h5 file
for i in range(len(f[iter_str])):
    if "VisualisationVector" in keys:
        # Dataset generated with FEniCS
        mesh_string = mesh_str+"/"+str(i)+"/"+geometry_str
        top_string = mesh_str+"/"+str(i)+"/"+topology_str
    elif "Function" in keys:
        # Dataset generated with FEniCSx
        mesh_string = mesh_str+"/"+geometry_str
        top_string = mesh_str+"/"+topology_str
    # These lines extract the mesh information, x and y coordinates, and the solution vector
    mesh = f[mesh_string]
    dim = mesh.shape[1]
    if dim == 2:
        triang = np.array(f[top_string]) + 1
        x = mesh[:, 0:1]
        y = mesh[:, 1:2]
        u = np.array(f[function_string+str(i)])
        if u.shape[1] > 1:
            u = np.sqrt(np.sum(np.square(u), axis=1)).reshape((-1, 1))
        try:
            xx = np.concatenate([xx, x], axis=1)
            yy = np.concatenate([yy, y], axis=1)
            solution = np.concatenate([solution, u], axis=1)
        except:
            xx = x
            yy = y
            solution = u
    elif dim == 3:
        triang = np.array(f[top_string]) + 1
        x = mesh[:, 0:1]
        y = mesh[:, 1:2]
        z = mesh[:, 2:3]
        u = np.array(f[function_string+str(i)])
        if u.shape[1] > 1:
            u = np.sqrt(np.sum(np.square(u), axis=1)).reshape((-1, 1))
        try:
            xx = np.concatenate([xx, x], axis=1)
            yy = np.concatenate([yy, y], axis=1)
            zz = np.concatenate([zz, z], axis=1)
            solution = np.concatenate([solution, u], axis=1)
        except:
            xx = x
            yy = y
            zz = z
            solution = u

# These lines extract the edges from the triangulation
if dim == 2:
    edges = np.array(extract_edges(triang))
elif dim == 3:
    edges = np.array(extract_edges_3d(triang))
edges = edges[edges[:, 1].argsort()]
edges = edges[edges[:, 0].argsort(kind='mergesort')]

# These lines create a dictionary that stores the triangulation, edges, number of degrees of freedom, solution, x and y coordinates
mat_file = "../dataset/"+problem_names[pb]+"_unstructured.mat"
dataset = dict()
dataset['T'] = triang.astype('float')
dataset['E'] = edges.astype('float')
dataset['dof'] = float(dof)
dataset['U'] = solution
dataset['xx'] = xx
dataset['yy'] = yy
if dim == 3:
    dataset['zz'] = zz

# This line saves the dataset as a .mat file
savemat(mat_file, dataset)