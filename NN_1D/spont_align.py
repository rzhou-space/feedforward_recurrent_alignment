import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from Networks import LinearRecurrentNetwork
from sklearn.decomposition import PCA



class Dimensionality:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param R: The radius of eigenvalue distribution.
        """
        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        # The i-th column in the eigvec is the eigenvector corresponded to the i-th eigenvalue in eigval.
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)



    def evok_activity(self, kappa, beta, L, basis_vectors, num_sample):
        """
        :param basis_vectors: An array containing basis vectors. nxn dimensional array.
        :param num_sample: the number of response samples.
        """
        # Determine the upper limit factor M.
        M = kappa * beta

        # Calculate the input variance Sigma^Dim.
        sigma_dim = np.zeros((self.neurons, self.neurons))
        for i in range(L-1, L-1+M): # Given L starts with 1.
            v_i = np.exp(-2*(i-L)/beta)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = basis_vectors[:, i].reshape(-1,1)
            sigma_dim += v_i * (e_i @ e_i.T) # A nxn dimensional numpy matrix.

        # Calculate the response variance Sigma^Act.
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_dim @ new_interact.T # (1-J)^(-1)*sigma_dim*(1-J)^(-T)
        # Samples from multivariate Gaussian distribution generate the response vectors.
        rng = np.random.default_rng(seed = 42)
        act_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.



########################################################################################################################

class AlignmentSpontaneousAct:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param R: The radius of eigenvalue distribution.
        """
        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        # The i-th column in the eigvec is the eigenvector corresponded to the i-th eigenvalue in eigval.
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        #self.rng = np.random.default_rng(seed=42)



    def spont_input(self, kappa, beta, L, num_sample):
        # Determine the upper limit factor M.
        M = kappa * beta

        # Calculate the input variance Sigma^spont.
        sigma_spont = np.zeros((self.neurons, self.neurons))
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = self.eigvec[:, i].reshape(-1,1)
            sigma_spont += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.

        #input_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_spont, size = num_sample)
        rng = np.random.default_rng(seed=42)
        input_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_spont, size = num_sample)
        return input_vec, sigma_spont  # input_vec is a num_sample x n_neuron dimensional array.
        # sigma_spont will be used to generate spont. act in the method "spont_act".



    def spont_act(self, sigma_spont, num_sample):
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_spont @ new_interact.T # (1-J)^(-1)*sigma_spont*(1-J)^(-T)
        #act_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        rng = np.random.default_rng(seed=42)
        act_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.



    def align_A_to_B_alter(self, act_patternA, act_patternB):
        """
        :param act_patternA: n_sample x n_neuron dimensional numpy array.
        :param act_patternB: n_sample x n_neuron dimensional numpy array.
        """
        # Calculate the covariance matrix of pattern B.
        cov_B = np.cov(act_patternB.T) # n_neuron x n_neuron dimensional covariance between neurons.
        # Normalization of the act_patternA.
        row_norms = np.linalg.norm(act_patternA, axis=1)
        normalized_patternA = act_patternA / row_norms[:, np.newaxis]
        # Calculate the whole score matrix with pattern A aligned to pattern B.
        all_align = normalized_patternA @ cov_B @ normalized_patternA.T # all_align a n_sample x n_sample dimensional array.
        # Extract the digonal elements for the defined aligned score.
        align_scores = all_align.diagonal()
        # Take the mean value and divided by the trace of cov(B).
        final_score = np.mean(align_scores)/np.trace(cov_B)

        return final_score



    def all_ffrec(self):
        # Sort eigenvectors in ascending order of eigenvalues.
        sorted_indices = np.argsort(self.eigval)
        #sorted_eigval = self.eigval[sorted_indices]
        sorted_eigvec = self.eigvec[:, sorted_indices]

        # Calculate all pairwise alignment scores resulting a matrix.
        pair_ffrec = sorted_eigvec.T @ self.interaction @ sorted_eigvec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()

        return ffrec_align # (n_neuron,) dimensional array.



    def align_to_ffrec_alter(self, kappa, beta_spont, num_sample, beta_dim):
        # Access the spontaneous activity. L = 1 already inserted.
        sigma_spont = self.spont_input(kappa, beta_spont, 1, num_sample)[1]
        spont_act = self.spont_act(sigma_spont, num_sample)

        # Order the eigenvectors in descending order of eigenvalues. The sorted eigenvectors are applied to generate
        # the evoked activity below.
        sorted_indices = np.argsort(self.eigval)[::-1]
        sorted_eigvec = self.eigvec[:, sorted_indices]

        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!

        # Storing the pattern to pattern alignment scores.
        pattern_align = np.zeros(len(L))

        for L_current in L: # L starts with 0.
            # Access the evoked activity under the given L_current using sorted eigenvectors.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, sorted_eigvec, num_sample)
            pattern_align[L_current] = self.align_A_to_B_alter(current_act, spont_act)

        return pattern_align # (len(L) < n_neuron,) dimensional array.



    def plot_align_to_ffrec(self, pattern_align):
        ffrec_align = self.all_ffrec()
        plt.figure()
        plt.title("Spontaneous Alignment against Feedforward Recurrent Alignment")
        plt.xlabel("Feedforward Recurrent Alignment")
        plt.ylabel("Alignment with spont. Activity")
        plt.scatter(ffrec_align[-len(pattern_align):], np.flip(pattern_align), alpha=0.5)
        plt.show()





########################################################################################################################
if __name__ == "__main__":
    # Global parameters:
    sigma_trial = 0.05
    sigma_time = 0.3
    N_trial = 500
    n_neuron = 200
    R = 0.85
    T = 120
    dt_intra = 20
    dt_euler = 0.1
    kappa = 5
    beta_dim = 10
    beta_spont = 20
    num_sample = 500


    SpontObj = AlignmentSpontaneousAct(n_neuron, R)
    align_scores = SpontObj.align_to_ffrec_alter(kappa, beta_spont, num_sample, beta_dim)
    SpontObj.plot_align_to_ffrec(align_scores)
