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
        # self.rng = np.random.default_rng(seed = 42)



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
            sigma_dim += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.

        # Calculate the response variance Sigma^Act.
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_dim @ new_interact.T # (1-J)^(-1)*sigma_dim*(1-J)^(-T)
        # Samples from multivariate Gaussian distribution generate the response vectors.

        # act_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        rng = np.random.default_rng(seed = 10)
        act_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.



    def variance_ratio(self, dataset):
        """
        :param dataset: numpy array containing evoked activity vectors. num_sample x n_neuron dimensional.
        """
        # Create a PCA object with the desired number of components. Here take all the samples.
        pca = PCA(n_components=self.neurons)
        # Fit the data to the PCA model.
        pca.fit(dataset)
        # Transform the data to the specified number of components.
        data_trans = pca.transform(dataset)
        # Get the explained variance ratio.
        explained_variance = pca.explained_variance_ratio_
        return explained_variance



    def align_eigvec(self, kappa, beta, L, num_sample):
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(self.eigval)[::-1]
        #sorted_eigenvalues = self.eigval[sorted_indices]
        # Columns are the sorted eigenvectors.
        sorted_eigenvectors = self.eigvec[:, sorted_indices]
        # Calculate the evoked activity with sorted eigenvectors.
        activity = self.evok_activity(kappa, beta, L, sorted_eigenvectors, num_sample)
        # Get the variance ratio with PCA on evoked activity pattern.
        var_ratio = self.variance_ratio(activity)
        return var_ratio



    def align_random(self, kappa, beta, L, num_sample):
        # Generate orthonormal random vectors with Gram-Schmidt orthogonalization.
        # Generate firstly a random n x n dimensional matrix.
        # random_matrix = self.rng.normal(0, 1, size = (self.neurons, self.neurons))
        rng = np.random.default_rng(seed = 10)
        random_matrix = rng.normal(0, 1, size = (self.neurons, self.neurons))
        # Perform Gram-Schmidt orthogonalization. The columns of matrix q form a an orthonormal set.
        q, r = np.linalg.qr(random_matrix)
        # Calculate the evoked activity with sorted eigenvectors.
        activity = self.evok_activity(kappa, beta, L, q, num_sample)
        # Get the variance ratio with PCA on evoked activity pattern.
        var_ratio = self.variance_ratio(activity)
        return var_ratio



    def plot_align(self, kappa, beta, num_sample):
        """
        :return: A plot containing the variance ratio of num_sampes of PC in both cases of aligned with eigenvectors or
        random orthonormal vectors.
        """
        # L = 1.
        var_aligned = self.align_eigvec(kappa, beta, 1, num_sample)[:20]
        var_random = self.align_random(kappa, beta, 1, num_sample)[:20]
        plt.figure()
        plt.title("Variance Ratio of Aligned and Random Inputs")
        plt.xlabel("PC Index")
        plt.ylabel("Variance ratio")
        plt.plot([i for i in range(20)], var_aligned, c="blue", label = "Aligned")
        plt.plot([i for i in range(20)], var_random, c="green", label = "Random")
        plt.legend()
        plt.show()



    def dim_to_ffrec(self, kappa, beta):
        """
        Calculation dimensionality analytically.
        """
        # Sorted eigenvalues in descending order.
        sorted_indices = np.argsort(self.eigval)[::-1]
        sorted_eigeval = self.eigval[sorted_indices]

        # Define the repeated function in the d_eff.
        inner_function = lambda k, L, beta, lambda_k, factor : np.exp(factor * (k-L)/beta) * (1-lambda_k)**factor

        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!
        # Calculate the numerator of d_eff.
        dim_eff_above = np.zeros(len(L))
        # Calculate the denominator of d_eff.
        dim_eff_below = np.zeros(len(L))

        for L_current in L: # L_current begins with 0!
            above = []
            below = []
            for k in range(L_current, L_current + M):
                eigval = sorted_eigeval[k]
                above.append(inner_function(k, L_current+1, beta, eigval, -2))
                below.append(inner_function(k, L_current+1, beta, eigval, -4))
            dim_eff_above[L_current] = sum(above)**2
            dim_eff_below[L_current] = sum(below)

        # Divide dim_eff_above and dim_eff_below elementwise to get the vector of final d_eff.
        d_eff = dim_eff_above/dim_eff_below

        return L, d_eff
        #return dim_eff_above, dim_eff_below



    def dim_to_ffrec_empir(self, kappa, beta, num_sample):
        """
        Calculate dimensionality empirically.
        :param var_ratio: the variance ratio vector. (num_neuron, ) dimensional numpy array.
        """
        # Sorted eigenvectors in descending order of eigenvalues.
        sorted_indices = np.argsort(self.eigval)[::-1]
        sorted_eigvec = self.eigvec[:, sorted_indices]

        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!
        # Calculate the dimentionality for a certain variance ratio.
        d_eff = lambda var_ratio_vec : sum(var_ratio_vec)**2/sum(var_ratio_vec**2)

        dim = []
        for L_current in L:
            var_vec = self.align_eigvec(kappa, beta, L_current+1, num_sample)
            dim_current = d_eff(var_vec[:M])
            dim.append(dim_current)
        return L, np.array(dim)



    def plot_dim_to_ffrec(self, kappa, beta, num_sample):
        dim_ffrec = self.dim_to_ffrec(kappa, beta)
        dim_ffrec_empir = self.dim_to_ffrec_empir(kappa, beta, num_sample)

        # x-axis with L.
        '''
        plt.figure()
        plt.title("L values against effective dimensionality")
        plt.xlabel("L values")
        plt.ylabel("effective dimensionality")
        plt.plot(dim_ffrec[0], dim_ffrec[1], c = "green", label = "analytical")
        plt.plot(dim_ffrec_empir[0], dim_ffrec_empir[1], c = "blue", label = "empirical")
        plt.legend()
        plt.show()
        '''

        # If the relationship between L and feedforward alignment is linear.
        plt.figure()
        plt.title("feedforward alignment against dimensionality")
        plt.xlabel("feedforward alignment")
        plt.ylabel("effective dimensionality")
        plt.plot(np.linspace(0,1,int(n_neuron/2)), np.flip(dim_ffrec[1]), c = "green", label = "analytical")
        plt.scatter(np.linspace(0,1,int(n_neuron/2)), np.flip(dim_ffrec_empir[1]), c = "blue", label = "empirical", alpha=0.5)
        plt.legend()
        plt.show()


########################################################################################################################
if __name__ == "__main__":
    # Global parameters:
    # sigma_trial = 0.02
    sigma_trial = 0.05
    sigma_time = 0.3
    N_trial = 100
    n_neuron = 200
    R = 0.85
    T = 120
    dt_intra = 20
    dt_euler = 0.1
    kappa = 5
    beta_dim = 10
    beta_spont = 20
    num_sample = 3000

    # Results from dimensionality.

    dim_obj = Dimensionality(n_neuron, R)
    # fig 4d above.
    #dim_obj.plot_align(kappa, beta_dim, num_sample)

    # fig 4d below.
    dim_obj.plot_dim_to_ffrec(kappa, beta_dim, num_sample)


    '''
    # Minimal example of two different implementations. Run in the console to test.
    import numpy as np
    # random1 is the correct implementation (in methods).
    class random1:
        def generate(self):
            rng = np.random.default_rng(seed = 10)
            return rng.normal(0, 1, size = 3)

        def multiple_generate(self):
            l = []
            for i in range(3):
                l.append(self.generate())
            return l

    class random2:
        def __init__(self):
            self.rng = np.random.default_rng(seed = 10)

        def generate(self):
            return self.rng.normal(0, 1, size = 3)

        def multiple_generate(self):
            l = []
            for i in range(3):
                l.append(self.generate())
            return l

    result1 = random1()
    print(result1.generate())
    print(result1.generate()) # same as the first output.
    print(result1.multiple_generate()) # Each time calling method has the same results.
    result2 = random2()
    print(result2.generate()) # same as the result1.
    print(result2.generate()) # different from the first output of result2.
    print(result2.multiple_generate()) # Each time calling has different set of numbers!
    '''




