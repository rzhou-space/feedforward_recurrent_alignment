import numpy as np
import matplotlib.pyplot as plt

class LinearRecurrentNetwork:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param R: The eigenvalue radius.
        """
        self.neuron = n
        self.R = R
        # Applying random interaction matrix.
        self.interaction = self.random_interaction_matrix()


    def random_interaction_matrix(self):
        """
        Generating the random interaction matrix (nxn dimensional) with normal distribution.
        The matrix is real, full rank and symmetric.
        :return: The interaction matrix.
        """
        #mu = 0
        #sigma = self.R/np.sqrt(self.neuron)
        rng = np.random.default_rng(seed = 42)
        J = rng.normal(0, 1, size = (self.neuron,self.neuron))
        # J should be a symmetric matrix.
        J = (J + J.T)/2
        #normalize by largest eigenvalue
        eigvals = np.linalg.eigvals(J)
        J = J*self.R/np.max(eigvals)
        return J

    def eigval_distribution(self):
        J = self.interaction
        eigval = np.linalg.eigvals(J)
        plt.scatter(eigval, np.zeros(len(eigval)), s=1)
        plt.show()

    '''
    def trial_response(self, h_det, sigma_trial, N_trial):
        """
        The mean response vector given the trial input vector. Multivariate normal distributed.
        :param h_det: The deterministic part of the input (1xn dimensional).
        :param sigma_trial: The stength of the covariance matrix.
        :param N_trial: The number of trials.
        :return: The response vector matrix (Nxn dimensional) depending on the input for N trials.
        """
        new_inter = np.linalg.inv(np.identity(self.neuron) - self.interaction) # (1-J)^(-1).
        mean = np.matmul(new_inter, h_det)  # 1xn dimensional array.
        cov = (sigma_trial**2)*(np.matmul(new_inter, new_inter.T))
        # Response vector normally distributed.
        rng = np.random.default_rng(seed = 42)
        r_trial = rng.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial

    
    def trial_response_alter(self, eigval, eigvec, h_det, sigma_trial, N_trial):
        # Calculations of mean and covariance matrix.
        n = len(eigval)
        mean = 0
        cov = 0
        for i in range(n):
            mean += (np.dot(eigvec[i], h_det)/(1-eigval[i])) * eigvec[i]
            vec = np.reshape(eigvec[i], (n,1))
            vecT = np.reshape(eigvec[i], (1,n))
            cov += np.matmul(vec, vecT)/(1-eigval[i])**2
        cov = sigma_trial**2*cov

        # Response vector normally distributed.
        r_trial = np.random.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial
    '''


class LowRank:

    def __init__(self, n, D, R):
        self.neuron = n
        self.rank = D  # The rank of the interaction matrix.
        self.R = R  # The eigenvalue normalization factor.
        self.interaction = self.low_rank_inter() # (n,n) dimenaional symmatrical.

    def low_rank_inter(self):
        # Generate the basis vector that construct the matrix.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, 1, size = (self.neuron, self.rank)) # The columns are basis vectors.
        # Orthogonalization of m.
        m,r = np.linalg.qr(m) # m is a Matrix with orthonormal columns.
        inter_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1) # get (n_neuron, 1) dimensional vector.
            # m_i @ m_i^T symmetrical. inter_J is sum of symmetrical matrices.
            inter_J += (1/self.neuron)*(m_i @ m_i.T)
        #Normalize the interaction matrix by R.
        eigvals = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J*self.R/np.max(eigvals)
        return inter_J

    def eigval_distribution(self):
        eigvals = np.linalg.eigvalsh(self.interaction) # Eigenvalues are real numbers.
        plt.figure()
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.yticks([])
        plt.show()
        return eigvals



class NoisedLowRank_1D:

    def __init__(self, n, R):
        self.neuron = n
        self.R = R
        self.interaction = self.noise_low_rank_inter()

    def noise_low_rank_inter(self):
        # Generate the low rank matrix basis.
        rng = np.random.default_rng(seed = 42)
        m = rng.normal(0, 1, size = self.neuron) # (self.neuron, ) dimensional.
        # Normalization of m and turn it into (self.neuron, 1) dimensional.
        m = m/np.linalg.norm(m)
        m = m.reshape((self.neuron, 1))
        # The noise/random symmetircal part of the matrix.
        rand_J = rng.normal(0, 1, size = (self.neuron, self.neuron))
        rand_J = (rand_J + rand_J.T)/2
        # Construct the low rank interaction as low rank part + random part.
        inter_J = (m @ m.T)/self.neuron + rand_J # Symmetrical (n_neuron, n_neuron) matrix.
        # Normalization of the matrix.
        eigval = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J * self.R / np.max(eigval)
        return inter_J

    def eigval_distriution(self):
        eigvals = np.linalg.eigvalsh(self.interaction)
        plt.figure()
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.show()
        return eigvals



class NoisedLowRank_nD:

    def __init__(self, n, R, D):
        self.neuron = n
        self.R = R
        self.rank = D
        self.interaction = self.noise_low_rank_inter()

    def noise_low_rank_inter(self):
        # Generate the random basis vector set.
        rng = np.random.default_rng(seed=42)
        m = rng.normal(0, 1, size = (self.neuron, self.rank))
        # Orthogonalization of m.
        q, r = np.linalg.qr(m)
        m = q # Matrix with orthonormal columns.
        low_J = np.zeros((self.neuron, self.neuron))
        for i in range(self.rank):
            m_i = m[:, i].reshape(-1, 1)  # Turn the (n_neuron,) dimension into (n_neuron, 1) dimension.
            low_J += (1/self.neuron)*(m_i @ m_i.T)
        # The noise/random symmetircal part of the matrix.
        rand_J = rng.normal(0, 1, size = (self.neuron, self.neuron))
        rand_J = (rand_J + rand_J.T)/2
        # Construct the low rank interaction as low rank part + random part.
        inter_J = low_J + rand_J
        # Normalization of the matrix.
        eigval = np.linalg.eigvalsh(inter_J)
        inter_J = inter_J * self.R / np.max(eigval)
        return inter_J

    def eigval_distriution(self):
        eigvals = np.linalg.eigvalsh(self.interaction)
        plt.figure()
        plt.scatter(eigvals, np.zeros(len(eigvals)))
        plt.show()
        return eigvals


######################################################################################################
if __name__ == "__main__":
    # Global setting.
    n_neuron = 200
    R = 0.85
    sigma_trial = 0.02
    N_trial = 100

    sym_low_rank = LowRank(200, 1, R)
    print(sym_low_rank.eigval_distribution())












