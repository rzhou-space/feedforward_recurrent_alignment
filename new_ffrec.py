import numpy as np
import matplotlib.pyplot as plt
from NN_1D import Networks


def total_ffrec_align(h_det, interaction): # TODO: further step: distribution with eigenvalues.
    """
    Calculate the ffrec with (Jh)^* (Jh) where (Jh)^* the complex conjugate of the vector.
    :param h_det: n x 1 dimensional array.
    :param interaction: nxn asymmetric matrix.
    """
    h_norm = np.linalg.norm(h_det)
    transform_vec = interaction @ h_det
    ffrec = np.conjugate(transform_vec).T @ transform_vec/ h_norm**2
    return ffrec


def eigval_ffrec(interaction):
    #interaction = network.interaction
    eigval, eigvec = np.linalg.eig(interaction)
    ffrec = []
    for i in range(len(eigval)):
        h = eigvec[:, i].reshape(-1, 1) # turn to (n, 1) dimensional array.
        ffrec.append(total_ffrec_align(h, interaction))
    return np.array(ffrec), eigval


def plot_ffrec_eigval(ffrec, eigval, mode):
    plt.figure()
    if mode == "real":
        plt.scatter(eigval, ffrec)
    plt.show()

########################################################################################################################
n = 200
R = 0.85

sym_full_rank = Networks.LinearRecurrentNetwork(n, R).interaction

ffrec, eigval = eigval_ffrec(sym_full_rank)
plot_ffrec_eigval(ffrec, eigval, "real")

