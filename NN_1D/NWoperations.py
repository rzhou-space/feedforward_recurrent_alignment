import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from NN_1D.Networks import LinearRecurrentNetwork, LowRank, NoisedLowRank_1D, NoisedLowRank_nD
from sklearn.decomposition import PCA


class TrialToTrialCor:
    #TODO: make sigma_trial depend on neuron numbers so that each neuron contribute to it.
    def __init__(self, n, R, network):
        """
        :param n: The number of neurons.
        :param mode: There are three modes: max_align, random_align, and sort_align.
        """
        self.neurons = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)


    def trial_response(self, h_det, sigma_trial, N_trial):
        """
        The response vector given the trial input vector. Multivariate normal distributed.
        :param h_det: The deterministic part of the input (1xn dimensional).
        :param sigma_trial: The strength of the covariance matrix.
        :param N_trial: The number of trials.
        :return: The response vector matrix (Nxn dimensional) depending on the input for N trials.
        """
        new_inter = np.linalg.inv(np.identity(self.neurons) - self.interaction) # (1-J)^(-1).
        mean = np.matmul(new_inter, h_det)  # 1xn dimensional array.
        cov = (sigma_trial**2)*(np.matmul(new_inter, new_inter.T))
        # Response vector normally distributed.
        rng = np.random.default_rng(seed = 42)
        r_trial = rng.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial


    def single_cor(self, r_trial):
        """
        Correlations of pairwise trials.
        :param r_trial: N_trial x n_neuron dimensional array.
        :return: The correlations between trials.
        """
        # Get the covariance matrix for trials.
        trial_cor = np.corrcoef(r_trial) # N_trial x N_trial dimensional
        # Take the upper right triangle without diagonal elements.
        upr_triangle = trial_cor[np.triu_indices(trial_cor.shape[0], k=1)]
        return upr_triangle


    def hist_cor_distribution(self, sigma_trial, N_trial):
        # Get the correlation values under random input and maximal aligned input.

        # Random input.
        rng = np.random.default_rng(seed = 3)
        h_rand = rng.normal(0, 1, size = n_neuron)
        h_rand = h_rand/np.linalg.norm(h_rand)
        # Response by random input.
        r_rand = self.trial_response(h_rand, sigma_trial, N_trial)
        # Correlation of random response.
        cor_rand = self.single_cor(r_rand)

        # Maximal aligned input.
        h_max = self.eigvec[:, np.argmax(self.eigval)]
        h_max = h_max/np.linalg.norm(h_max)
        # Response by maximal aligned input.
        r_max = self.trial_response(h_max, sigma_trial, N_trial)
        # Correlation of maximal aligned response.
        cor_max = self.single_cor(r_max)

        # plot the distribution of the correlations as histogram.
        plt.figure()
        bins = np.linspace(-1, 1, 100)
        plt.title("Trial to Trial correlation distribution")
        plt.xlabel("correlation")
        plt.ylabel("frequency")
        plt.hist(cor_rand, bins, color = "green", alpha = 0.5, label = "random")
        plt.hist(cor_max, bins, color = "blue", alpha = 0.5, label = "real max")
        plt.legend()
        plt.show()


    def trial_to_trial_correlation(self, all_response, N_trial):
        '''
        Calculation of trial to trial correlation.
        :param all_response: Nxn dimensional numpy array containing all responses for N-trials with n neurons.
        :param N_trial: The number of trials.
        :return: The trial to trial correlation of all N trials.
        '''
        # The correlation matrix over all response vectors.
        cor_matrix = np.corrcoef(all_response)
        # Take the sum of upper triangular matrix to get the sum of all pairwise correlations.
        sum_cor = (cor_matrix.sum() - np.diag(cor_matrix).sum())/2
        # The trial to trial correlation as mean over sum of correlations.
        beta = 2*sum_cor/(N_trial*(N_trial-1))
        return beta


    def ttc_max_align(self, sigma_trial, N_trial, repeats):
        """
        Calculation of the mean trial to trial correlations under the maximal alignment of interaction
        matrix.
        :return: An array containing the number of repeats of trial to trial correlation.
        """
        result = []
        # Choose the eigenvector corresponded to maximal eigenvalue as h_det.
        h_det = self.eigvec[:, np.argmax(self.eigval)]
        for i in range(repeats):
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)


    def ttc_random_align(self, sigma_trial, N_trial, repeats):
        """
        Calculation of the mean trial to trial correlation under random alignment of the interaction
        matrix.
        :return: An array containing the number of repeats of trial to trial correlation.
        """
        result = []
        rng = np.random.default_rng(seed = 42)
        rand_h = rng.normal(0, 1, size = self.neurons)
        h_det = rand_h/np.linalg.norm(rand_h)
        for i in range(repeats):
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)


    def hist_plot(self, ttc_max, ttc_rand):
        """
        Histogramm plot tool for ttc max and ttc rand distribution.
        :param ttc_max: ttc with maximal alignment.
        :param ttc_rand: ttc with random alignment.
        :return: A histogramm of the frequency of certain ttc value.
        """
        plt.figure()
        bins = np.linspace(-1, 1, 100)
        plt.title("Trial to Trial Correlations")
        plt.xlabel("Correlation Values")
        plt.ylabel("Frequency")
        plt.hist(ttc_max, bins, alpha = 0.5, label = "maximal alignment")
        plt.hist(ttc_rand, bins, alpha = 0.5, label = "random alignment")
        plt.legend(loc = "upper right")
        plt.show()


    def ttc_sort_align(self, sigma_trial, N_trial):
        """
        Calculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        :return: An array containing a trial to trial correlation for each h_det.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        result = []
        result_ffrec_align = []
        for i in sort_index:
            h_det = self.eigvec[:, i]
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            result_ffrec_align.append(h_det @ self.interaction @ h_det)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result), np.asarray(result_ffrec_align)  # ttc values, ffrec-align values.


    def scatter_plot(self, align_vec, ttc_vec):
        """
        Scatter plot tool for ttc_vec of sorted ttc values.
        :param ttc_vec: the numpy array containing ttc values from sorted alignment.
        :param align_vec: the numpy array containing align values from sorted alignment.
        :return: a scatter plot with x-axis the ordered alignment score = eigenvalues and y-axis the ttc values.
        """
        plt.figure(figsize=(6,5))
        plt.scatter(align_vec, ttc_vec, alpha=0.5)
        #plt.axvline(x = 0, color = "black")
        #plt.axvline(x = self.R, color = "red", label = "align = R = "+str(self.R))
        plt.xlabel("Feedforward recurrent alignment", fontsize=18)
        plt.ylabel("Trial-to-trial correlation", fontsize = 18)
        plt.xticks([-0.5, 0, 0.5], fontsize=15)
        plt.yticks([0, 0.5, 1], fontsize=15)
        #plt.legend()
        plt.savefig("F:/Downloads/fig_ttc.pdf", bbox_inches='tight')
        plt.show()


########################################################################################################################

class IntraTrialStab:

    def __init__(self, n, R, network):

        self.neurons = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)
        # Sort eigenvectors in the order of ascending eigenvalues.
        sorted_indices = np.argsort(self.eigval)
        self.sorted_eigvec = self.eigvec[:, sorted_indices]


    def euler_maruyama(self, dt_euler, T, h_det, sigma_time):
        """
        Calculate the response vector with Euler Maruyama scheme.
        :param dt_euler: Time step distance by scheme.
        :param T: The total length of response time.
        :param h_det: Deterministic part of input.
        :param sigma_time: Variance parameter at euler scheme.
        :return: The response vector within time range T.
        """
        start_act = np.linalg.inv(np.eye(self.neurons) - self.interaction) @ h_det
        num_steps = int(T/dt_euler)
        sqrtdt=np.sqrt(dt_euler)
        rng = np.random.default_rng(seed = 42)

        res = []
        res.append(start_act)

        # Euler-Maruyama Scheme
        for istep in range(num_steps):
            act_t= np.copy(res[-1])

            dW = sqrtdt * rng.normal(size=self.neurons)
            K1 =  dt_euler * (-1*act_t + h_det + self.interaction @ act_t )  + sigma_time*(dW)

            act_new = act_t + K1
            res.append(act_new)

        return np.asarray(res)[1:]


    def rand_align(self, dt_euler, T, sigma_time):
        """
        Access response under alignment with random vector.
        """
        rng = np.random.default_rng(seed=42)
        h_det = rng.normal(0, 1, size = self.neurons)
        # Normalization.
        h_det /= np.linalg.norm(h_det)

        r = self.euler_maruyama(dt_euler, T, h_det, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)


    def max_align(self, dt_euler, T, sigma_time):
        """
        Access response under alignment with maximal eigenvector.
        """
        # The eigenvector corresponding with the largest eigenvalue.
        h_det = self.sorted_eigvec[:, -1]

        r = self.euler_maruyama(dt_euler, T, h_det, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)


    def plot_max_rand_align(self, r_rand, r_max):
        '''
        r_rand and r_max are (num_neuron,) dimensional array, each for response of the same neuron
        under the random and maximal alignment.
        '''
        # Normalization of neuron response so that they have the same mean value after normalization.
        r_rand_normal = r_rand/np.mean(r_rand)
        r_max_normal = r_max/np.mean(r_max)  # If the given response vector is under maximal alignment.

        plt.figure()
        plt.title("Intra Trial Neuron Activity")
        plt.xlabel("time (ms)")
        plt.ylabel("Normalized activity")
        plt.plot(np.linspace(0,120,1200), r_rand_normal, c = "green", label = "random align", alpha = 0.7)
        plt.plot(np.linspace(0,120,1200), r_max_normal, c = "blue", label = "max align", alpha = 0.7)
        plt.legend()
        plt.show()


    def stab_hdet(self, dt_euler, dt_intra, h_det, T, sigma_time):
        """
        Calculate the intra trial stability under the alignment with a given h_det.
        :param dt_euler: The time step distance in euler scheme.
        :param dt_intra: The time step distance in correlation calculation.
        :param h_det: The deterministic input.
        :param T: The total time range for response.
        :param sigma_time: The variance parameter needed at euler scheme.
        :return: The mean intra trial correalation for one dt_intra.
        """
        # Access the response vector under the given h_det through Euler_Maruyama
        # with steady state as initial condition.
        r0 = np.linalg.inv(np.eye(self.neurons) - self.interaction) @ h_det # (1-J)^-1 * h_det
        r_vec = self.euler_maruyama(dt_euler, T, h_det, sigma_time) # n_steps x n_neuron dimensional

        dt = int(dt_intra/dt_euler)
        cor = np.corrcoef(r_vec)

        return np.mean(np.diag(cor, k=dt))


    def sort_stab(self, dt_euler, dt_intra, T, sigma_time):
        """
        Calculate the intra trial stability under the alignment with eigenvectors in descending
        order of eigenvalues.
        For each h_det = eigvec the stability is calculated.
        """
        ffrec_align = []
        mean_stab = []
        # Range over sorted eigenvectors to calculate the mean stability for each of them.
        for i in range(len(self.eigval)):
            h_det = self.sorted_eigvec[:, i]
            mean_stab.append(self.stab_hdet(dt_euler, dt_intra, h_det, T, sigma_time))
            ffrec_align.append(h_det @ self.interaction @ h_det)

        return ffrec_align, mean_stab


    def plot_stab(self, ffrec_align, mean_stab):
        """
        Plot the intra trial stability against the feedforward recurrent alignment.
        """
        plt.figure(figsize=(6,5))
        #plt.title("Intra Trial Stability")
        plt.xlabel("Feedforward recurrent alignment", fontsize = 18)
        plt.ylabel("Intra-trial Stability", fontsize = 18)
        #plt.yscale("log")
        #plt.xlim(-self.R-0.1, self.R+0.1)
        plt.scatter(ffrec_align, mean_stab, alpha=0.5)
        plt.xticks([-0.5, 0, 0.5], fontsize=15)
        plt.yticks([0, 0.5, 1], fontsize=15)
        plt.savefig("F:/Downloads/fig_its.pdf", bbox_inches='tight')
        plt.show()


##############################################################################################################

class Dimensionality:

    def __init__(self, n, R, network):
        """
        :param n: The number of neurons.
        :param R: The radius of eigenvalue distribution.
        """
        self.neurons = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        # The i-th column in the eigvec is the eigenvector corresponded to the i-th eigenvalue in eigval.
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)

        sorted_indices = np.argsort(self.eigval)[::-1]
        self.sorted_eigeval = self.eigval[sorted_indices]
        self.sorted_eigvec = self.eigvec[:, sorted_indices]


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
        rng = np.random.default_rng(seed = 42)
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
        rng = np.random.default_rng(seed = 42)
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
        plt.figure(figsize=(6,5))
        #plt.title("Variance Ratio of Aligned and Random Inputs")
        plt.xlabel("PC Index", fontsize=18)
        plt.ylabel("Variance ratio", fontsize=18)
        plt.plot([i for i in range(1, 21)], var_aligned, c="red", label = "Maximal aligned")
        plt.plot([i for i in range(1, 21)], var_random, c="green", label = "Random aligned")
        plt.xticks([1, 5, 10, 15, 20], fontsize=15)
        plt.yticks([0, 0.1, 0.2, 0.3], fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig("F:/Downloads/fig_dim.pdf", bbox_inches='tight')
        plt.show()


    def dim_to_ffrec(self, kappa, beta):
        """
        Calculation dimensionality analytically.
        """
        # Sorted eigenvalues in descending order.
        #sorted_indices = np.argsort(self.eigval)[::-1]
        #sorted_eigeval = self.eigval[sorted_indices]
        #sorted_eigvect = self.eigvec[:, sorted_indices]

        # Define the repeated function in the d_eff.
        inner_function = lambda k, L, beta, lambda_k, factor : np.exp(factor * (k-L)/beta) * (1-lambda_k)**factor

        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!
        # Calculate the numerator of d_eff.
        dim_eff_above = np.zeros(len(L))
        # Calculate the denominator of d_eff.
        dim_eff_below = np.zeros(len(L))

        ffrec = []
        for L_current in L: # L_current begins with 0!
            above = []
            below = []
            h = self.sorted_eigvec[:, L_current]
            ffrec.append(h @ self.interaction @ h)
            for k in range(L_current, L_current + M):
                eigval = self.sorted_eigeval[k]
                above.append(inner_function(k, L_current+1, beta, eigval, -2))
                below.append(inner_function(k, L_current+1, beta, eigval, -4))
            dim_eff_above[L_current] = sum(above)**2
            dim_eff_below[L_current] = sum(below)

        # Divide dim_eff_above and dim_eff_below elementwise to get the vector of final d_eff.
        d_eff = dim_eff_above/dim_eff_below

        return np.array(ffrec), d_eff
        #return dim_eff_above, dim_eff_below


    def dim_to_ffrec_empir(self, kappa, beta, num_sample):
        """
        Calculate dimensionality empirically.
        :param var_ratio: the variance ratio vector. (num_neuron, ) dimensional numpy array.
        """
        # Sorted eigenvectors in descending order of eigenvalues.
        #sorted_indices = np.argsort(self.eigval)[::-1]
        #sorted_eigvec = self.eigvec[:, sorted_indices]

        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!
        # Calculate the dimentionality for a certain variance ratio.
        d_eff = lambda var_ratio_vec : sum(var_ratio_vec)**2/sum(var_ratio_vec**2)

        dim = []
        ffrec = []
        for L_current in L:
            h = self.sorted_eigvec[:, L_current]
            ffrec.append(h @ self.interaction @ h)
            var_vec = self.align_eigvec(kappa, beta, L_current+1, num_sample)
            dim_current = d_eff(var_vec[:M])
            dim.append(dim_current)
        return np.array(ffrec), np.array(dim)


    def plot_dim_to_ffrec(self, kappa, beta, num_sample):
        ffrec_ana, dim_ffrec = self.dim_to_ffrec(kappa, beta)
        ffrec_empir, dim_ffrec_empir = self.dim_to_ffrec_empir(kappa, beta, num_sample)

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
        plt.figure(figsize=(6,5))
        #plt.title("feedforward alignment against dimensionality")
        plt.xlabel("Feedforward recurrent alignment", fontsize = 18)
        plt.ylabel("Dimensionality", fontsize = 18)

        #plt.plot(np.linspace(0,self.R,int(n_neuron/2)), np.flip(dim_ffrec[1]), c = "green", label = "Analytical")
        plt.scatter(np.flip(ffrec_ana), np.flip(dim_ffrec), c = "green", label = "Analytical")
        plt.scatter(np.flip(ffrec_empir), np.flip(dim_ffrec_empir), label = "Empirical", alpha=0.5)
        plt.xticks([0, self.R], fontsize=15)
        plt.yticks([2,4,6,8,10], fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig("F:/Downloads/fig_dim.pdf", bbox_inches='tight')
        plt.show()


########################################################################################################################

class AlignmentSpontaneousAct:

    def __init__(self, n, R, network):
        """
        :param n: The number of neurons.
        :param R: The radius of eigenvalue distribution.
        """
        self.neurons = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        # The i-th column in the eigvec is the eigenvector corresponded to the i-th eigenvalue in eigval.
        #self.eigval, self.eigvec = np.linalg.eigh(self.interaction)

        #rng = np.random.default_rng(seed = 42)
        #J = rng.normal(0, 1, size = (self.neurons,self.neurons))
        # J should be a symmetric matrix.
        #J = (J + J.T)/2
        #normalize by largest eigenvalue
        #eigvals = np.linalg.eigvals(J)
        #self.interaction = J*self.R/np.max(eigvals)

        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)

        # Sort eigenvectors in descending order.
        sorted_indices = np.argsort(self.eigval)[::-1]
        self.sorted_eigval = self.eigval[sorted_indices]
        self.sorted_eigenvec = self.eigvec[:, sorted_indices]


    def spont_input(self, kappa, beta, L, num_sample):
        """
        The input for spontaneous activity.
        """
        # Determine the upper limit factor M.
        M = kappa * beta

        # Calculate the input variance Sigma^spont.
        sigma_spont = np.zeros((self.neurons, self.neurons))
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            #e_i = self.eigvec[:, i].reshape(self.neurons,1)
            e_i = self.sorted_eigenvec[:, i].reshape(self.neurons, 1)
            sigma_spont += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.

        rng = np.random.default_rng(seed=42)
        input_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_spont, size = num_sample)
        return input_vec, sigma_spont  # input_vec is a num_sample x n_neuron dimensional array.
        # sigma_spont will be used to generate spont. act in the method "spont_act".


    def spont_act(self, sigma_spont, num_sample):
        """
        The spontaneous activity evoked by the broad input "spont_input".
        """
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_spont @ new_interact.T # (1-J)^(-1)*sigma_spont*(1-J)^(-T)
        rng = np.random.default_rng(seed=42)
        act_vec = rng.multivariate_normal(np.zeros(self.neurons), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.


    def variance_spont_input_act(self, input_vec, act_vec):
        """
        Variance ratio through PCA of spontaneous input and its spontaneous activity.
        """
        # Apply the PCA on input and response vectors.
        input_var = Dimensionality.variance_ratio(self, input_vec)
        act_var = Dimensionality.variance_ratio(self, act_vec)

        # Plot the variances for input and response.
        plt.figure()
        plt.title("Variance Ratio of Spontaneous Input and Activity")
        plt.xlabel("Spont.PC")
        plt.ylabel("Variance Ratio")
        x_axis = [i for i in range(20)]  # [str(i) for i in range(20)] could show x axis with integers.
        plt.plot(x_axis, input_var[:20], c = "grey", label = "Broad Inputs")
        plt.plot(x_axis, act_var[:20], c = "red", label = "Spont. Activity")
        plt.legend()
        plt.show()


    def var_explain_A_by_B(self, act_patternA, act_patternB):
        """
        Calculate the variance ratio where pattern A is explained by the PCs of pattern B.
        :param act_patterA: n_sample x n_neuron dimensional
        :param act_patternB: n_sample x neuron dimensional
        """
        # Extract the principal components of pattern B.
        pca = PCA(n_components=self.neurons)
        pca.fit_transform(act_patternB)
        pc_B = pca.components_ # Rows are principal components.

        # Calculate the covariance matrix of pattern A.
        cov_A = np.cov(act_patternA.T)

        # Calculate the projected variacnce of A in B.
        var_ratio = np.zeros(self.neurons)
        for i in range(self.neurons):
            var_ratio[i] = pc_B[i,:] @ cov_A @ pc_B[i, :] / np.trace(cov_A)

        return var_ratio


    def compare_var_explain_align_rand_spont(self, kappa, beta_spont, beta_dim, L, num_sample):
        # Access the variance ratio of spontaneous activity.
        sigma_spont = self.spont_input(kappa, beta_spont, L, num_sample)[1]
        spont_act = self.spont_act(sigma_spont, num_sample)
        spont_var = Dimensionality.variance_ratio(self, spont_act)

        # Access the explained variance ratio of aligned input evoked activity by spont. act.
        align_act = Dimensionality.evok_activity(self, kappa, beta_dim, L, self.sorted_eigenvec, num_sample)
        align_expl_var = self.var_explain_A_by_B(align_act, spont_act)

        # Access the explained variance ratio of random input evoked activity by spont. act.
        # When applying multiple situmuli empirically -- without seed.
        #rng = np.random.default_rng(seed=42)
        #random_matrix = rng.normal(0, 1, size = (self.neurons, self.neurons))
        random_matrix = np.random.normal(0, 1, size = (self.neurons, self.neurons))
        q, r = np.linalg.qr(random_matrix)
        rand_act = Dimensionality.evok_activity(self, kappa, beta_dim, L, q, num_sample)
        rand_expl_var = self.var_explain_A_by_B(rand_act, spont_act)

        '''
        # Plot the three variance ratios together. If applying with more networks, turn the plot-operations here off.
        plt.figure()
        plt.title("Alignment of evoked to spontaneous activity")
        plt.xlabel("spont. PC")
        plt.ylabel("variance ratio")
        plt.plot(np.arange(20), spont_var[:20], c = "red", label = "spont. act")
        plt.plot(np.arange(20), align_expl_var[:20], c = "blue", label = "aligned act.")
        plt.plot(np.arange(20), rand_expl_var[:20], c = "green", label = "rand act.")
        plt.legend()
        plt.show()
        '''

        return spont_var, align_expl_var, rand_expl_var


    def multiple_stimuli_compare_align_rand_spont(self, num_stimuli, kappa, beta_spont, beta_dim, L, num_sample):
        """
        For num_stimuli times (for each stimuli different) generate the variance of spontanous act,
        aligned act and random aligned act explained by spontaneous activity PC. Plot their mean and confidence interval
        for the explained variance with the first 20 PCs.
        """
        all_spont = []
        all_align = []
        all_rand = []
        for i in range(num_stimuli):
            # Calculate the variance explained for three aligned activity variance expalained by spont. PC.
            #spont_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[0]
            #align_expl_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[1]
            #rand_expl_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[2]
            spont_var, align_expl_var, rand_expl_var = self.compare_var_explain_align_rand_spont(kappa,
                                                                                                 beta_spont, beta_dim, L, num_sample)
            # Consider the first 20 PCs for illustration.
            all_spont += spont_var[:20].tolist()
            all_align += align_expl_var[:20].tolist()
            all_rand += rand_expl_var[:20].tolist()

        # X-axis for the line plot.
        pc_index = [i for rep in range(num_stimuli) for i in range(1,21)] # A list with 1,...,20 repeat num_stimuli times.

        # Lineplot with confidence interval.
        plt.figure(figsize=(6, 5))
        #plt.title("Alignment of evoked to spontaneous activity")
        plt.xlabel("Spontaneous Activity PC index ", fontsize=18)
        plt.ylabel("Variance ratio", fontsize=18)
        sns.lineplot(x=pc_index, y=all_spont, color="red", label="Spontaneous")
        sns.lineplot(x=pc_index, y=all_align, color="blue", label="Maximal aligned")
        sns.lineplot(x=pc_index, y=all_rand, color="green", label="Random aligned")
        plt.xticks(range(1, 21, 4))
        plt.legend(fontsize=15)
        plt.xticks(fontsize=18)
        plt.yticks([0.0, 0.1, 0.2, 0.3], fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()


    #One way to calculate the alignment and the correlation with ffrec-align.
    '''
    def align_A_to_B(self, act_patternA, act_patternB, num_sample):
        """
        Calculate the alignment of pattern A to pattern B as fraction of total pattern B's variance
        explained by A.
        :param act_patternA: n_sample x n_neuron dimensional numpy array.
        :param act_patternB: n_sample x n_neuron dimensional numpy array.
        """
        align_scores = np.zeros(num_sample)

        # Calculate the covariance matrix of pattern B.
        cov_B = np.cov(act_patternB.T)

        for i in range(num_sample):
            r_iA = act_patternA[i, :]
            align_scores[i] = r_iA @ cov_B @ r_iA / ((np.linalg.norm(r_iA)**2)*np.trace(cov_B))

        return np.mean(align_scores)



    def align_to_ffrec(self, kappa, beta_spont, beta_dim, num_sample):
        """
        Plot the correlation between the alignment of evoked activity pattern aligned to spontaneous activity
        and the feedforward recurrent alignment.
        Or the dependency of alignment on feedforward recurrent alignment through the choice of eigenvector subset.
        """
        # Access the spontaneous activity. L = 1 already inserted.
        sigma_spont = self.spont_input(kappa, beta_spont, 1, num_sample)[1]
        spont_act = self.spont_act(sigma_spont, num_sample)

        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!

        # Sorted eigenvalues in descending order.
        sorted_indices = np.argsort(self.eigval)[::-1]
        sorted_eigeval = self.eigval[sorted_indices]
        sorted_eigvec = self.eigvec[:, sorted_indices]

        align = np.zeros(len(L))
        ffrec = np.zeros(len(L))
        # Take L from 1 to n/2 to formulate the evoked activities.
        for L_current in L: # L starts with 0.
            # Accesee the evoked activity under the given L_current.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, sorted_eigvec, num_sample)
            # Calculate the alignment score for this current activity.
            align[L_current] = self.align_A_to_B(current_act, spont_act, num_sample)
            ffrec[L_current] = sorted_eigvec[:, L_current] @ self.interaction @ sorted_eigvec[:, L_current]

        # Plot the alignment with spontaneous activity against ff-rec.alignment.
        # The larger the L, the smaller the ff-rec. aligment. Therefore apply the flipped "align".
        plt.figure()
        plt.title("Alignment Spont.Activity against FF-Rec. Alignment")
        plt.xlabel("ff-rec. Alignment")
        plt.ylabel("Alignment to spont. activity")
        plt.scatter(np.flip(ffrec), np.flip(align))
        plt.show()
    '''


    def align_A_to_B_alter(self, act_patternA, act_patternB):
        """
        Alternative calculation of alignment between two activity patterns using vectorised operations.
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

    '''
    def all_ffrec(self):
        """
        Calculate the feedforward recurrent alignment with eigenvectors sorted in ascending order of
        eigenvalues.
        """
        # Sort eigenvectors in ascending order of eigenvalues.
        #sorted_indices = np.argsort(self.eigval)
        #sorted_eigval = self.eigval[sorted_indices]
        #sorted_eigvec = self.eigvec[:, sorted_indices]

        # Calculate all pairwise alignment scores resulting a matrix.
        pair_ffrec = self.sorted_eigenvec.T @ self.interaction @ self.sorted_eigenvec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()

        return ffrec_align # (n_neuron,) dimensional array.
    '''


    def align_to_ffrec_alter(self, kappa, beta_spont, num_sample, beta_dim):
        '''
        Calculate the aligenment scores of evoked activity depending on L aligned to the spontaneous activity.
        '''
        # Access the spontaneous activity. L = 1 already inserted.
        sigma_spont = self.spont_input(kappa, beta_spont, 1, num_sample)[1]
        spont_act = self.spont_act(sigma_spont, num_sample)

        # Order the eigenvectors in descending order of eigenvalues. The sorted eigenvectors are applied to generate
        # the evoked activity below.
        #sorted_indices = np.argsort(self.eigval)[::-1]
        #sorted_eigvec = self.eigvec[:, sorted_indices]

        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!

        # Storing the pattern to pattern alignment scores.
        pattern_align = np.zeros(len(L))
        ffrec = np.zeros(len(L))
        for L_current in L: # L starts with 0.
            # Access the evoked activity under the given L_current using sorted eigenvectors.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, self.sorted_eigenvec, num_sample)
            pattern_align[L_current] = self.align_A_to_B_alter(current_act, spont_act)
            h = self.sorted_eigenvec[:, L_current]
            ffrec[L_current] = h @ self.interaction @ h

        return ffrec, pattern_align # (len(L) < n_neuron,) dimensional array.


    def plot_align_to_ffrec(self, ffrec, pattern_align):
        #ffrec_align = self.all_ffrec()
        plt.figure(figsize=(6,5))
        #plt.title("Spontaneous Alignment against Feedforward Recurrent Alignment")
        plt.xlabel("Feedforward recurrent alignment", fontsize=18)
        plt.ylabel("Alignment to spont. Activity", fontsize = 18)
        plt.scatter(np.flip(ffrec), np.flip(pattern_align), alpha=0.5)
        #plt.scatter(np.linspace(0,self.R,int(n_neuron/2)), np.flip(pattern_align), alpha=0.5)
        #plt.xlim(-self.R, self.R+0.1) # For the case of rescaled x-axis.
        plt.xticks([0, 0.4, 0.8], fontsize=15)
        plt.yticks([0, 0.05, 0.1, 0.15], fontsize=15)
        plt.savefig("F:/Downloads/fig_align.pdf", bbox_inches='tight')
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
    num_sample = 500

    network_rand_sym = LinearRecurrentNetwork(n_neuron, R)
    network_low_rank_sym = LowRank(n_neuron, 1, R)
    network_noise_low_rand_sym = NoisedLowRank_1D(n_neuron, R)
    network_noise_low_rand_sym_nD = NoisedLowRank_nD(n_neuron, R, 100)


    # Results from trial to trial correlation.

    ttc_Obj = TrialToTrialCor(n_neuron, R, network_noise_low_rand_sym)
    
    # fig 4b upper.

    #ttc_Obj.hist_cor_distribution(sigma_trial, N_trial)
    
    #ttc_max = ttc_Obj.ttc_max_align(sigma_trial, N_trial, 100)
    #ttc_rand = ttc_Obj.ttc_random_align(sigma_trial, N_trial, 100)
    #ttc_Obj.hist_plot(ttc_max, ttc_rand)
    
    
    # fig 4b below.
    sort_align = ttc_Obj.ttc_sort_align(sigma_trial, N_trial)
    ttc_vec = sort_align[0]
    ffrec_align = sort_align[1]
    ttc_Obj.scatter_plot(ffrec_align, ttc_vec)

    '''

    # Results from intra trial stability.

    its_Obj = IntraTrialStab(n_neuron, R, network_noise_low_rand_sym)
    
    # fig 4c upper. 
    #r_rand = its_Obj.rand_align(dt_euler, T, sigma_time)[:,12]
    #r_max = its_Obj.max_align(dt_euler, T, sigma_time)[:, 12]
    #its_Obj.plot_max_rand_align(r_rand, r_max)
    
    # fig 4c below.
    sort_its = its_Obj.sort_stab(dt_euler, dt_intra, T, sigma_time)
    ffrec_align = sort_its[0]
    mean_stab = sort_its[1]
    print(ffrec_align)
    its_Obj.plot_stab(ffrec_align, mean_stab)


    
    # Results from dimensionality.
    
    dim_obj = Dimensionality(n_neuron, R, network_noise_low_rand_sym)

    # fig 4d above.
    #dim_obj.plot_align(kappa, beta_dim, num_sample)

    # fig 4d below.
    dim_obj.plot_dim_to_ffrec(kappa, beta_dim, num_sample)


    # Results from Alignment to spontaneous activity.
    
    align_spont_obj = AlignmentSpontaneousAct(n_neuron, R, network_noise_low_rand_sym)
    # fig 4e above. (Has conflict with 4f above)!
    #input_vec = align_spont_obj.spont_input(kappa, beta_spont, 1, num_sample)
    #act_vec = align_spont_obj.spont_act(input_vec[1], num_sample)
    #align_spont_obj.variance_spont_input_act(input_vec[0], act_vec)
    
    # fig 4f above
    #align_spont_obj.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, 1, num_sample)
    
    # When running multiple_stimuli function, deactivate the fig 4e above!!! Otherwise random generator conflict...
    #align_spont_obj.multiple_stimuli_compare_align_rand_spont(50, kappa, beta_spont, beta_dim, 1, num_sample)

    
    # fig 4f below.
    # The frist way to implement:
    #align_spont_obj.align_to_ffrec(kappa, beta_spont, beta_dim, num_sample)
    # The alternative to implement:
    ffrec, align_scores = align_spont_obj.align_to_ffrec_alter(kappa, beta_spont, num_sample, beta_dim)
    align_spont_obj.plot_align_to_ffrec(ffrec, align_scores)
    '''






































