import numpy as np
import matplotlib.pyplot as plt


class AsymLinearRecurrentNet:

    def __init__(self, n, R):
        """
        :param n: the number of neurons.
        :param R: the eigenvalue radius.
        """
        self.neuron = n
        self.R = R
        # TODO: build in a check point to garantie the interaction matrix has determinant != 0 for
        ## the existence of fixpoint?
        self.interaction = self.asym_random_interaction_matrix()


    def asym_random_interaction_matrix(self):
        """
        Generating the asymmetric random interaction matrix (nxn dimensional) through normal distribution.
        Its eigenvalue distribution would be limited by the circle with radius R.
        :return: nxn dimensional asymmetric interaction matrix.
        """
        rng = np.random.default_rng(seed=42)
        J = rng.normal(0, 1, size = (self.neuron, self.neuron))
        eigenvals = np.linalg.eigvals(J) # Eigenvalues of the interaction matrix, including complex numbers.
        # Get the maximal magnitude of eigenvalues.
        max_mag = max(map(abs, eigenvals))
        # Normalize the eigenvalues by the maximal magnitude and rescale through the given radius.
        J = J * self.R / max_mag
        return J


    def eigval_distribution(self):
        """
        Plot the eigenvalue distribution of the rescaled interaction matrix to check if the radius is
        correctly reset.
        """
        new_eigenvals = np.linalg.eigvals(self.interaction)
        # Plot the new eigenvalues in 2D space.
        # Extract real part.
        real = new_eigenvals.real
        # Extract imaginary part.
        imag = new_eigenvals.imag

        plt.figure()
        plt.scatter(real, imag)
        plt.axvline(x = self.R, c = "red", label = "R="+str(self.R))
        plt.axvline(x = - self.R, c = "red")
        plt.axhline(y = self.R, c = "red")
        plt.axhline(y = -self.R, c = "red")
        plt.title("Eigenvalue distribution")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.legend()
        plt.show()

        return new_eigenvals



class CombiAsymNet:
    '''
    Construct the asymmetrical interaction matrix J = a*J_sym + (1-a)*J_asym.
    '''
    def __init__(self, n, a, R):
        self.neuron = n
        self.a = a  # The rate for J_sym.
        self.R = R  # For both matrices the radius limit.
        self.asymnet = AsymLinearRecurrentNet(self.neuron, self.R)
        self.interaction = self.combi_inter()


    def combi_inter(self):
        # Generate a symmetrical matrix.
        rng = np.random.default_rng(seed = 42)
        J_sym = rng.normal(0, 1, size = (self.neuron,self.neuron))
        J_sym = (J_sym + J_sym.T)/2
        # Normalized by largest eigenvalue
        sym_eigvals = np.linalg.eigvals(J_sym)
        J_sym = J_sym*self.R/np.max(sym_eigvals)

        # Generate an asymmetrical matrix.
        J_asym = self.asymnet.interaction

        # Combine the matrices through a*J_sym + (1-a)*J_asym.
        J = self.a * J_sym + (1-self.a) * J_asym

        J_eigvals = np.linalg.eigvals(J)
        max_mag = max(map(abs, J_eigvals))
        J = J * self.R/max_mag
        return J


    def eigval_dist(self):
        new_eigenvals = np.linalg.eigvals(self.interaction)
        # Plot the new eigenvalues in 2D space.
        # Extract real part.
        real = new_eigenvals.real
        # Extract imaginary part.
        imag = new_eigenvals.imag

        plt.figure(figsize=(4, 4))
        plt.scatter(real, imag, alpha=0.5)
        plt.axvline(x = self.R, c = "grey")#, label = "R="+str(self.R))
        plt.axvline(x = - self.R, c = "grey")
        plt.axhline(y = self.R, c = "grey")
        plt.axhline(y = -self.R, c = "grey")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        #ax = plt.gca()
        #ax.set_aspect('equal', adjustable='box')
        plt.xticks([-1, 0, 1], fontsize=15)
        plt.yticks([-1, 0, 1], fontsize=15)
        #plt.title("Eigenvalue distribution, a="+str(self.a))
        #plt.xlabel("Real Part", fontsize=15)
        #plt.ylabel("Imaginary Part", fontsize=15)
        #plt.legend()
        plt.axis('scaled')
        plt.show()

        return new_eigenvals



# Construct the low rank interaction matrix.
class LowRank:

    def __init__(self, n, D, R, sigma_1, sigma_2):
        self.neuron = n
        self.rank = D  # The rank/dimension of the interaction matrix.
        self.R = R  # The normalization factor of the interaction matrix.
        self.sigma_1 = sigma_1  # Determine the importance of the first mode.
        self.sigma_2 = sigma_2  # Detetmine the importance of the second mode.
        # The interaction matrix is in general not symmetrical.
        self.interaction = self.low_rank_inter()


    def low_rank_inter(self):
        # sigma_1 and sigma_2 determines the weight of the modes.
        # The first mode set. The columns are mode vectors.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, self.sigma_1, size = (self.neuron, self.rank))
        # Orthogonalization of m.
        q1, r1 = np.linalg.qr(m)
        m = q1 # Matrix with orthonormal columns.
        # The second mode set. The columns are mode vectors.
        n = rng.normal(0, self.sigma_2, size = (self.neuron, self.rank))
        q2, r2 = np.linalg.qr(n)
        n = q2
        inter_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1)  # Turn the (n_neuron,) dimension into (n_neuron, 1) dimension.
            n_i = n[:, i].reshape(-1, 1)
            inter_J += (1/self.neuron)*(m_i @ n_i.T)
        # Normalize the interaction matrix (generally not symmetrical).
        J_eigvals = np.linalg.eigvals(inter_J)
        max_mag = max(map(abs, J_eigvals))
        inter_J = inter_J * self.R/max_mag
        return inter_J


    def eigval_distribution(self):
        eigvals = np.linalg.eigvals(self.interaction)

        real = eigvals.real
        imag = eigvals.imag

        plt.figure(figsize=(6,5))
        plt.scatter(real, imag, alpha=0.5)
        #plt.axvline(x = self.R, c = "red", label = "R="+str(self.R))
        #plt.axvline(x = -self.R, c = "red")
        #plt.axhline(y = self.R, c = "red")
        #plt.axhline(y = -self.R, c = "red")
        #plt.title("Eigenvalue distribution")
        #plt.xlabel("Real Part", fontsize=20)
        plt.xlim(-self.R-0.1, self.R+0.1)
        plt.ylim(-self.R-0.1, self.R+0.1)
        #plt.xlabel("Real Part", fontsize=20, labelpad=15)
        plt.xticks([-self.R, 0, self.R], fontsize=15)
        #plt.ylabel("Imaginary Part", fontsize=20)
        plt.yticks([-0.5, 0, 0.5], fontsize=15)
        #plt.legend()
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()

        return eigvals



class NoisedLowRank:

    def __init__(self, n, D, R):
        self.neuron = n
        self.rank = D
        self.R = R
        self.interaction = self.noise_low_rank_inter()

    def noise_low_rank_inter(self):
        # The first mode set. The columns are mode vectors.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, 1, size = (self.neuron, self.rank))
        # Orthogonalization of m.
        q1, r1 = np.linalg.qr(m)
        m = q1 # Matrix with orthonormal columns.
        # The second mode set. The columns are mode vectors. Differ generally from m.
        n = rng.normal(0, 1, size = (self.neuron, self.rank))
        q2, r2 = np.linalg.qr(n)
        n = q2
        low_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1)  # Turn the (n_neuron,) dimension into (n_neuron, 1) dimension.
            n_i = n[:, i].reshape(-1, 1)
            low_J += (1/self.neuron)*(m_i @ n_i.T)
        # Construct the random part.
        rand_J = rng.normal(0, 1, size=(self.neuron, self.neuron))
        # Final formulation of interaction matrix.
        inter_J = low_J + rand_J
        # Normalization of the inter_J.
        J_eigvals = np.linalg.eigvals(inter_J)
        max_mag = max(map(abs, J_eigvals))
        inter_J = inter_J * self.R/max_mag
        return inter_J

    def eigval_distribution(self):
        eigvals = np.linalg.eigvals(self.interaction)
        real = eigvals.real
        imag = eigvals.imag
        plt.figure(figsize=(6,5))
        plt.scatter(real, imag, alpha=0.5)
        #plt.axvline(x = self.R, c = "red", label = "R="+str(self.R))
        #plt.axvline(x = -self.R, c = "red")
        #plt.axhline(y = self.R, c = "red")
        #plt.axhline(y = -self.R, c = "red")
        #plt.title("Eigenvalue distribution")
        #plt.xlabel("Real Part", fontsize=18)
        plt.xticks([-self.R, 0, self.R], fontsize=18)
        plt.ylabel("Imaginary Part", fontsize=15)
        plt.yticks([-0.5, 0, 0.5], fontsize=15)
        #plt.legend()
        plt.show()



# Construct the asymmetrical interaction matrix J = J_sym + b*J_asym.
'''
class CombiAsymNet2:
    
    #Construct the asymmetrical interaction matrix J = J_sym + b*J_asym.
    
    def __init__(self, n, b, R):
        self.neuron = n
        self.b = b  # The rate for J_sym.
        self.R = R  # For both matrices the radius limit.
        self.asymnet = AsymLinearRecurrentNet(self.neuron, self.R)
        self.interaction = self.combi_inter()


    def combi_inter(self):
        # Generate a symmetrical matrix.
        rng = np.random.default_rng(seed = 42)
        J_sym = rng.normal(0, 1, size = (self.neuron,self.neuron))
        J_sym = (J_sym + J_sym.T)/2
        #normalize by largest eigenvalue
        eigvals = np.linalg.eigvals(J_sym)
        J_sym = J_sym*self.R/np.max(eigvals)

        # Generate an asymmetrical matrix.
        J_asym = self.asymnet.interaction

        # Combine the matrices through a*J_sym + (1-a)*J_asym.
        J = J_sym + self.b * J_asym
        return J


    def eigval_dist(self):
        new_eigenvals = np.linalg.eigvals(self.interaction)
        # Plot the new eigenvalues in 2D space.
        # Extract real part.
        real = new_eigenvals.real
        # Extract imaginary part.
        imag = new_eigenvals.imag

        plt.figure()
        plt.scatter(real, imag, alpha=0.5)
        plt.axvline(x = self.R, c = "red", label = "R="+str(self.R))
        plt.axvline(x = - self.R, c = "red")
        plt.axhline(y = self.R, c = "red")
        plt.axhline(y = -self.R, c = "red")
        plt.title("Eigenvalue distribution")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.legend()
        plt.show()

        return new_eigenvals
'''



#############################################################################################################
if __name__ == "__main__":
    n_neuron = 200 # With 1000 for better effect of eigval distribution picture.
    R = 0.85
    '''
    #network = AsymLinearRecurrentNet(n_neuron, R)
    #J = network.interaction
    #print(J)
    #eigval = network.eigval_distribution()
    #print(network.eigval)

    # J = a*J_sym + (1-a)*J_asym
    combi_net = CombiAsymNet(n_neuron, 1, R)
    J = combi_net.interaction
    combi_net.eigval_dist()
    
    lowrank = LowRank(n_neuron, 1, R, 1, 1)
    print(lowrank.eigval_distribution())
    
    # J = J_sym + (1-b)*J_asym
    combi_net2 = CombiAsymNet2(n_neuron, 0.5, R)
    J2 = combi_net2.interaction
    print(J2.shape)
    combi_net2.eigval_dist() # Hier has the problem that the rescaling of redius would not be hold.
    '''
    lowrank = LowRank(n_neuron, 10, R, 1, 1)
    lowrank.eigval_distribution()

    #noise_low_rank = NoisedLowRank(n_neuron, 1, R)
    #noise_low_rank.eigval_distribution()






