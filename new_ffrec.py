import numpy as np
import matplotlib.pyplot as plt
from NN_1D import Networks
from Asym_1D import AsymNetworks


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


def eigval_ffrec(network):
    interaction = network.interaction
    eigval, eigvec = np.linalg.eig(interaction)
    ffrec = []
    for i in range(len(eigval)):
        h = eigvec[:, i].reshape(-1, 1) # turn to (n, 1) dimensional array.
        ffrec.append(total_ffrec_align(h, interaction))
    return np.array(ffrec), eigval

# TODO. try with asymmetric interaction

def plot_ffrec_eigval(ffrec, eigval, mode):
    plt.figure()
    if mode == "real":
        plt.scatter(eigval, ffrec)
        plt.xlabel("eigval")
        plt.ylabel("ffrec")
    elif mode == "complex":
        x = eigval.real
        y = eigval.imag
        plt.xlabel("Re($\lambda$)")
        plt.ylabel("Im($\lambda$)")
        plt.scatter(x, y, c=ffrec, cmap = "PuBuGn")
        cbar = plt.colorbar()
        cbar.set_label("ffrec")
    plt.show()

'''
class ttc_sym:
    def __init__(self, network):
        self.interaction = 
'''

########################################################################################################################

if __name__ == "__main__":
    n = 200
    R = 0.85

    # symmetric full rank matrix.
    sym_full_rank = Networks.LinearRecurrentNetwork(n, R)
    ffrec, eigval = eigval_ffrec(sym_full_rank)
    plot_ffrec_eigval(ffrec, eigval, "real")

    # Asymmetric full rank matrix.
    asym_full_rank = AsymNetworks.AsymLinearRecurrentNet(n,R)
    ffrec, eigval = eigval_ffrec(asym_full_rank)
    plot_ffrec_eigval(ffrec, eigval, "complex")
