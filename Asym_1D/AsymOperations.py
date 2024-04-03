import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
import AsymNetworks as AN
import scipy as sp
from sklearn.decomposition import PCA
from NN_1D import NWoperations
import FfrecAlign




class FFRec_Alignment:

    def real_ffrec_align(self, h_det):
        """
        :param h_det: 1 x n_neuron numpy array. Deterministic part of input. Generally a complex vector.
        :return: feedforward recurrent alignment defined with the real part of h_det elementwise.
        """
        real_h = np.real(h_det)
        # Normalization of real_h.
        h = real_h/np.linalg.norm(real_h)
        # Insert into the general feedforward recurrent formular.
        ffrec = h @ self.interaction @ h
        return ffrec


    def mag_ffrec_align(self, h_det):
        """
        :param h_det: 1 x n_neuron numpy array. Deterministic part of input. Generally a complex vector.
        :return: feedforward recurrent alignment defined with the magnitude of h_det elementwise.
        """
        mag_h = np.abs(h_det)
        # Normalization of mag_h.
        h = mag_h/np.linalg.norm(mag_h)
        # Insert into the general formular.
        ffrec = h @ self.interaction @ h
        return ffrec


    def ffrec_align(self, h_det, interaction):
        # h_det real vector.
        h = h_det/np.linalg.norm(h_det)
        ffrec = h @ interaction @ h
        return ffrec




###########################################################################################################
# Most Operations introduced from symmetric network model NWoperations.py in folder 1D_NN.
# The main difference is the initial interaction matrix -- if symmetric or not. Here is asymmetrical.
###########################################################################################################

class TrialtoTrialCor:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        # TODO: right eigenvectors?
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        # TODO: numpy compares the complex numbers depending on the real part. When the real parts are
        ## the same, the complex part will be then taken into account.
        ## Is this the way we want to have? Or better with magnitude?
        self.maxeigvec = self.eigvec[:, np.argmax(self.eigval)]


    def trial_response(self, h, sigma_trial, N_trial):
        """
        Generate N_trial number of responses.
        :param h: 1 x n_neuron dimensional vector. Should be normalized real vector.
        :param sigma_trial: the strength of covariance in response.
        :param N_trial: number of trials.
        :return: N_trial x n_neuron dimensional array. N_trial responses rowwise returned.
        """
        # Normalization of h.
        h = h/np.linalg.norm(h)
        new_inter = np.linalg.inv(np.identity(self.neuron) - self.interaction) # (1-J)^(-1).
        mean = new_inter @ h
        cov = (sigma_trial**2)*(np.matmul(new_inter, new_inter.T))
        rng = np.random.default_rng(seed = 42)
        r_trial = rng.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial # N_trial x n_neuron dimensional array.


    def single_cor(self, r_trial):
        """
        :param r_trial: N_trial x n_neuron dimensional array.
        :return: The correlations between trials.
        """
        # Get the covariance matrix for trials.
        trial_cor = np.corrcoef(r_trial) # N_trial x N_trial dimensional
        # Take the upper right triangle without diagonal elements.
        upr_triangle = trial_cor[np.triu_indices(trial_cor.shape[0], k=1)]
        return upr_triangle


    def hist_real_cor_distribution(self, sigma_trial, N_trial):
        # Get the correlation values under random input and maximal aligned input.

        # Random input.
        rng = np.random.default_rng(seed = 3)
        h_rand = rng.normal(0, 1, size = n_neuron)
        # Response by random input.
        r_rand = self.trial_response(h_rand, sigma_trial, N_trial)
        # Correlation of random response.
        cor_rand = self.single_cor(r_rand)

        # Maximal aligned input.
        h_max = np.real(self.maxeigvec)
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


    def real_ttc_sort_align(self, sigma_trial, N_trial):
        """
        Calculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        :return: An array containing a trial to trial correlation for each h_det.
        """
        # Sort the eigenvalues based on their real part (when real parts are same, then imaginary part)
        # in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        ttc = []
        ffrec_align = []
        for i in sort_index:
            h = np.real(self.eigvec[:, i])  # real part of the eigenvectors.
            all_response = self.trial_response(h, sigma_trial, N_trial) # h normalized in the function.
            # Apply the feedforward recurrent alignment with real part. h normalized in the function.
            ffrec_align.append(FFRec_Alignment.real_ffrec_align(self, h))
            ttc.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(ffrec_align), np.array(ttc)


    def sym_ttc_sort_align(self, sigma_trial, N_trial):
        # Symmetrisize the original asymmetrical interaction matrix.
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec.
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)

        sort_index = np.argsort(sym_eigval)  # Sort eigenvalues of symmetrized J in ascending order.
        ttc = np.zeros(len(sym_eigval))
        ffrec = np.zeros(len(sym_eigval))
        for i in sort_index:
            # h_det is the i-th eigenvector of the symmtrisized matrix. They are therefore real.
            h_det = sym_eigvec[:, i]
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            # Calculate the ffrec still with the original asymmetrical interaction matrix.
            # As long as the corresponding values are on the same position, the plot should be correct.
            #ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)
            ffrec[i] = h_det @ self.interaction @ h_det
            ttc[i] = self.trial_to_trial_correlation(all_response, N_trial)
        return ffrec, ttc, sym_eigval, sym_eigvec


    def noise_ttc_sort_align(self, sigma_trial, N_trial):
        # PCs of the spontaneous activity evoked by white noise is the eigenvectors
        # of the covariance matrix.
        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1
        act_cov = np.matmul(steady_inter, steady_inter.T)
        cov_eigval, cov_eigvec = np.linalg.eigh(act_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        variance_ratio = cov_eigval[sort_index]
        pc = cov_eigvec[:, sort_index]

        ffrec = np.zeros(self.neuron)
        ttc = np.zeros(self.neuron)
        for i in range(self.neuron):
            # h_det is the i-th PC from the spontaneous activity evoked by white noise.
            h_det = pc[:, i]
            # Calcualte the ffrec-align with PCs.
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)
            # Calcualte the mean trial to trial correlation.
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            ttc[i] = self.trial_to_trial_correlation(all_response, N_trial)
        return ffrec, ttc, variance_ratio, pc


    '''
    def plot_sort_align(self, ffrec_align, ttc):
        # Scattering the result against the feedforward recurrent alignment.
        plt.figure()
        plt.scatter(ffrec_align, ttc, alpha = 0.5)
        #plt.axvline(x = 0, color = "black")
        #plt.axvline(x = self.R, color = "red", label = "align = R = "+str(self.R))
        plt.title("Trial to Trial Correlation against ffrec-Alignment")
        plt.xlabel("feedforward alignment")
        plt.ylabel("trial to trial correlation")
        #plt.legend()
        plt.show()
    '''

############################################################################################################
class IntraTrialStab:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        # TODO: right eigenvector and numpy sort.
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        # The eigenvector with the maximal eigenvalue.
        self.maxeigvec = self.eigvec[:, np.argmax(self.eigval)]
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
        # Steady state as intitial condition.
        start_act = np.linalg.inv(np.eye(self.neuron) - self.interaction) @ h_det
        num_steps = int(T/dt_euler)
        sqrtdt = np.sqrt(dt_euler)
        rng = np.random.default_rng(seed = 42)

        res = []
        res.append(start_act)
        # Euler-Maruyama Scheme
        for istep in range(num_steps):
            act_t= np.copy(res[-1])
            dW = sqrtdt * rng.normal(size=self.neuron)
            K1 =  dt_euler * (-1*act_t + h_det + self.interaction @ act_t )  + sigma_time*(dW)
            act_new = act_t + K1
            res.append(act_new)

        return np.asarray(res)[1:]


    def rand_align(self, dt_euler, T, sigma_time):
        """
        Access response under alignment with random vector.
        """
        rng = np.random.default_rng(seed=42)
        h_det = rng.normal(0, 1, size = self.neuron)
        # Normalization.
        h_det /= np.linalg.norm(h_det)
        r = self.euler_maruyama(dt_euler, T, h_det, sigma_time)
        # return r
        return h_det, np.linalg.inv(np.eye(self.neuron) - self.interaction) @ h_det


    def real_max_align(self, dt_euler, T, sigma_time):
        """
        Access response under alignment with the real part of the maximal eigenvector.
        """
        # The eigenvector corresponding with the largest eigenvalue.
        h_det = self.sorted_eigvec[:, -1]
        # Apply the real part of the complex eigenvector.
        h_real = np.real(h_det)
        # Normalization.
        h_real = h_real/np.linalg.norm(h_real)
        r = self.euler_maruyama(dt_euler, T, h_real, sigma_time)
        # return r
        return h_det, np.linalg.inv(np.eye(self.neuron) - self.interaction) @ h_det


    def plot_max_rand_align(self, dt_euler, T, sigma_time, neuron_index):
        # Get the responses under random align and real maximal align.
        r_rand = self.rand_align(dt_euler, T, sigma_time)[:, neuron_index]
        r_max = self.real_max_align(dt_euler, T, sigma_time)[:, neuron_index]
        # Normalization of neuron response so that they have the same mean value after normalization.
        r_rand_normal = r_rand/np.mean(r_rand)
        r_max_normal = r_max/np.mean(r_max)  # Given response vector is under maximal alignment.

        plt.figure()
        plt.title("Intra Trial Neuron Activity")
        plt.xlabel("time (ms)")
        plt.ylabel("Normalized activity")
        plt.plot(np.linspace(0,120,1200), r_rand_normal, c = "green", label = "random align", alpha = 0.7)
        plt.plot(np.linspace(0,120,1200), r_max_normal, c = "blue", label = "max align", alpha = 0.7)
        plt.legend()
        plt.show()


    def stab_h(self, dt_euler, dt_intra, h, T, sigma_time):
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
        r0 = np.linalg.inv(np.eye(self.neuron) - self.interaction) @ h # (1-J)^-1 * h_det
        r_vec = self.euler_maruyama(dt_euler, T, h, sigma_time) # n_steps x n_neuron dimensional
        dt = int(dt_intra/dt_euler)
        # Correlation of z-scored resposnes (at each t) with step width 200.
        cor = np.corrcoef(r_vec)

        return np.mean(np.diag(cor, k=dt))


    def plot_real_sort_stab(self, dt_euler, dt_intra, T, sigma_time):
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
            # Consider the real part of eigenvectors.
            real_h = np.real(h_det)
            mean_stab.append(self.stab_h(dt_euler, dt_intra, real_h, T, sigma_time))
            ffrec_align.append(FFRec_Alignment.real_ffrec_align(self, h_det))
        '''
        # Plot the intra trial stability against the feedforward recurrent alignment.
        plt.figure()
        plt.title("Intra Trial Stability")
        plt.xlabel("feedforward recurrent alignment")
        plt.ylabel("Intra Trial Stability")
        plt.scatter(ffrec_align, mean_stab, alpha = 0.5)
        plt.show()
        '''
        return ffrec_align, mean_stab


    def sym_sort_stab(self, dt_euler, dt_intra, T, sigma_time):
        # Symmetrisize the original asymmetrical interaction matrix.
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec.
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)

        # Sort eigenvalues of symmetrized J in ascending order.
        sort_index = np.argsort(sym_eigval)
        its = np.zeros(len(sym_eigval))
        ffrec = np.zeros(len(sym_eigval))
        for i in sort_index:
            # h is the corresponding eigenvector of i-th sorted eigenvalue.
            h_det = sym_eigvec[:, i]
            # Calculate ffrec with the original interaction matrix.
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)
            # Calculate the mean intra trial correlation.
            its[i] = self.stab_h(dt_euler, dt_intra, h_det, T, sigma_time)
        return ffrec, its, sym_eigval, sym_eigvec


    def noise_sort_stab(self, dt_euler, dt_intra, T, sigma_time):
        # PCs of the spontaneous activity evoked by white noise are the eigenvectors
        # of the covariance matrix.
        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1
        act_cov = np.matmul(steady_inter, steady_inter.T)
        cov_eigval, cov_eigvec = np.linalg.eigh(act_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        variance_ratio = cov_eigval[sort_index]
        PCs = cov_eigvec[:, sort_index]
        # Calculate the ffrec with PCs (i.e., eigvectors of cov) of spontaneous activity.
        ffrec = np.zeros(self.neuron)
        its = np.zeros(self.neuron)
        for i in range(self.neuron):
            # h_det is the i-th PC.
            h_det = PCs[:, i]
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)
            # Calculate the mean intra trial correlation.
            its[i] = self.stab_h(dt_euler, dt_intra, h_det, T, sigma_time)
        return ffrec, its, variance_ratio, PCs



##############################################################################################################
class Dimensionality:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        # TODO: right eigenvector and numpy sort.
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        # The eigenvector with the maximal eigenvalue.
        self.maxeigvec = self.eigvec[:, np.argmax(self.eigval)]
        # Sort eigenvectors in the order of descending eigenvalues.
        sorted_indices = np.argsort(self.eigval)[::-1]
        self.sorted_eigvec = self.eigvec[:, sorted_indices]
        self.sorted_eigval = self.eigval[sorted_indices]

        # For the case of symmetrized interaction matrix and
        # further operations on it.
        # Symmetrize the original asymmetrical interaction matrix.
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec. They are both real.
        self.sym_eigval, self.sym_eigvec = np.linalg.eigh(sym_inter)
        #sym_sort_index = np.argsort(self.sym_eigvec)[::-1]
        self.sort_sym_eigvec = np.sort(self.sym_eigvec)[::-1]

        # For the case of applying the response evoked by white noise.
        # Construct the covariance matrix of white noise activity.
        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (1-J)^-1
        act_cov = np.matmul(steady_inter, steady_inter.T)
        # Get the eigenvalues and eigenvectors of the covariance matrix --> PCs of the response.
        cov_eigval, cov_eigvec = np.linalg.eigh(act_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        self.var_ratio = cov_eigval[sort_index]
        self.pc = cov_eigvec[:, sort_index]


    def evok_activity(self, kappa, beta, L, basis_vectors, num_sample):
        """
        :param basis_vectors: An array containing basis vectors. nxn dimensional array.
        :param num_sample: the number of response samples.
        """
        # Determine the upper limit factor M.
        M = kappa * beta

        # Calculate the input variance Sigma^Dim.
        sigma_dim = np.zeros((self.neuron, self.neuron))
        for i in range(L-1, L-1+M): # Given L starts with 1.
            v_i = np.exp(-2*(i-L)/beta)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = basis_vectors[:, i].reshape(-1,1)
            # Should be a nxn dimensional numpy matrix.
            sigma_dim += v_i * (e_i @ e_i.T)

        # Calculate the response variance Sigma^Act.
        new_interact = np.linalg.inv(np.identity(self.neuron)-self.interaction) # (1-J)^(-1)
        # Since sigma_dim real, sigma_act also real.
        sigma_act = new_interact @ sigma_dim @ new_interact.T # (1-J)^(-1)*sigma_dim*(1-J)^(-T)
        # Samples from multivariate Gaussian distribution generate the response vectors.
        rng = np.random.default_rng(seed = 42)
        act_vec = rng.multivariate_normal(np.full(self.neuron, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.


    def variance_ratio(self, dataset):
        """
        :param dataset: numpy array containing evoked activity vectors. num_sample x n_neuron dimensional.
        """
        # Create a PCA object with the desired number of components. Here take all the samples.
        pca = PCA(n_components=self.neuron)
        # Fit the data to the PCA model.
        pca.fit(dataset)
        # Transform the data to the specified number of components.
        data_trans = pca.transform(dataset)
        # Get the explained variance ratio.
        explained_variance = pca.explained_variance_ratio_
        return explained_variance


    def real_align_eigvec(self, kappa, beta, L, num_sample):
        sorted_eigvec = np.real(self.sorted_eigvec)
        # Calculate the evoked activity with sorted eigenvectors.
        activity = self.evok_activity(kappa, beta, L, sorted_eigvec, num_sample)
        # Get the variance ratio with PCA on evoked activity pattern.
        var_ratio = self.variance_ratio(activity)
        return var_ratio


    def sym_align_eigvec(self, kappa, beta, L, num_sample):
        # Sort eigenvectors in the order of eigenvectors in descending order.
        sorted_indices = np.argsort(self.sym_eigval)[::-1]
        # Eigenvectors are columnwised corresponded to eigenvalues.
        sorted_eigvec = self.sym_eigvec[:, sorted_indices]
        # Calculate the evoked activity.
        activity = self.evok_activity(kappa, beta, L, sorted_eigvec, num_sample)
        # Get the variance ratio from activity.
        var_ratio = self.variance_ratio(activity)
        return var_ratio


    def noise_align_eigvec(self, kappa, beta, L, num_sample):
        # Calculate the evoked activity.
        activity = self.evok_activity(kappa, beta, L, self.pc, num_sample)
        # Get the variance ratio from activity.
        var_ratio = self.variance_ratio(activity)
        return var_ratio


    def align_random(self, kappa, beta, L, num_sample):
        # Generate orthonormal random vectors with Gram-Schmidt orthogonalization.
        # Generate firstly a random n x n dimensional matrix.
        rng = np.random.default_rng(seed = 42)
        random_matrix = rng.normal(0, 1, size = (self.neuron, self.neuron))
        # Perform Gram-Schmidt orthogonalization. The columns of matrix q form a an orthonormal set.
        q, r = np.linalg.qr(random_matrix)
        # Calculate the evoked activity with sorted eigenvectors.
        activity = self.evok_activity(kappa, beta, L, q, num_sample)
        # Get the variance ratio with PCA on evoked activity pattern.
        var_ratio = self.variance_ratio(activity)
        return var_ratio


    def plot_real_align(self, kappa, beta, num_sample):
        """
        :return: A plot containing the variance ratio of num_sampes of PC in both cases of aligned with eigenvectors or
        random orthonormal vectors.
        """
        # L = 1. Take the first 20 PCs for illustration.
        var_aligned = self.real_align_eigvec(kappa, beta, 1, num_sample)[:20]
        var_random = self.align_random(kappa, beta, 1, num_sample)[:20]
        plt.figure()
        plt.title("Variance Ratio of Aligned and Random Inputs")
        plt.xlabel("PC Index")
        plt.ylabel("Variance ratio")
        plt.plot([i for i in range(20)], var_aligned, c="blue", label = "Aligned")
        plt.plot([i for i in range(20)], var_random, c="green", label = "Random")
        plt.legend()
        plt.show()


    def analytical_dim(self, kappa, beta, sorted_eigval):
        # Define the repeated function in the d_eff.
        inner_function = lambda k, L, beta, lambda_k, factor : np.exp(factor * (k-L)/beta) * (1-lambda_k)**factor

        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!

        # Calculate the numerator of d_eff.
        #dim_eff_above = np.zeros(len(L))
        dim_eff_above = []
        # Calculate the denominator of d_eff.
        #dim_eff_below = np.zeros(len(L))
        dim_eff_below = []

        for L_current in L: # L_current begins with 0!
            above = []
            below = []
            for k in range(L_current, L_current + M):
                eigval = sorted_eigval[k]
                above.append(inner_function(k, L_current+1, beta, eigval, -2))
                below.append(inner_function(k, L_current+1, beta, eigval, -4))
            #dim_eff_above[L_current] = sum(above)**2
            #dim_eff_below[L_current] = sum(below)
            dim_eff_above.append(sum(above)**2)
            dim_eff_below.append(sum(below))

        # Divide dim_eff_above and dim_eff_below elementwise to get the vector of final d_eff.
        d_eff = np.array(dim_eff_above)/np.array(dim_eff_below)
        return d_eff


    def real_dim_to_ffrec(self, kappa, beta):
        """
        Calculation dimensionality analytically.
        """
        # Apply the real part of eigenvalues (descending order) to calculate the dimensionality.
        sorted_eigval = np.real(self.sorted_eigval)
        return self.analytical_dim(kappa, beta, sorted_eigval)


    def real_dim_to_ffrec_empir(self, kappa, beta, num_sample):
        """
        Calculate dimensionality empirically.
        :param var_ratio: the variance ratio vector. (num_neuron, ) dimensional numpy array.
        """
        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!
        # Calculate the dimentionality for a certain variance ratio.
        d_eff = lambda var_ratio_vec : sum(var_ratio_vec)**2/sum(var_ratio_vec**2)

        dim = []
        for L_current in L:
            var_vec = self.real_align_eigvec(kappa, beta, L_current+1, num_sample)
            dim_current = d_eff(var_vec[:M])
            dim.append(dim_current)
        return np.array(dim)

    def real_ffrec_dim(self):
        L = np.array([i for i in range(int(self.neuron/2))])
        basis_vec = np.real(self.sorted_eigvec)
        # Calculate the ffrec for basis_vec.
        pair_ffrec = basis_vec.T @ self.interaction @ basis_vec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()
        return np.flip(ffrec_align[L])


    def real_plot_dim_to_ffrec(self, kappa, beta, num_sample):
        dim_ffrec = self.real_dim_to_ffrec(kappa, beta)
        dim_ffrec_empir = self.real_dim_to_ffrec_empir(kappa, beta, num_sample)
        # If the relationship between L and feedforward alignment is linear.
        plt.figure()
        plt.title("feedforward alignment against dimensionality")
        plt.xlabel("feedforward alignment")
        plt.ylabel("effective dimensionality")
        plt.plot(np.linspace(0,1,int(n_neuron/2)), np.flip(dim_ffrec[1]), c = "green", label = "analytical")
        plt.scatter(np.linspace(0,1,int(n_neuron/2)), np.flip(dim_ffrec_empir[1]), c = "blue", label = "empirical", alpha=0.5)
        plt.legend()
        plt.show()

    def sym_ffrec(self):
        L = np.array([i for i in range(int(self.neuron/2))])
        sym_ffrec = []
        for i in L:
            h = self.sort_sym_eigvec[:, i]
            sym_ffrec.append(h @ self.interaction @ h)
        return np.array(sym_ffrec)


    def sym_dim_analytical(self, kappa, beta):
        #L = np.array([i for i in range(int(self.neuron/2))])
        #L = [i for i in range(0, 56)] + [i for i in range(57, int(self.neuron/2))]
        # Apply the eigenvalues of the symmetrized matrix. Sort them in descending order.
        sorted_eigval = np.sort(self.sym_eigval)[::-1]
        return self.analytical_dim(kappa, beta, sorted_eigval)


    def sym_dim_empir(self, kappa, beta, num_sample):
        """
        Analogous to the real case. Use the variance ratio to approximate the analytical
        dimensionality.
        """
        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!
        # Calculate the dimentionality for a certain variance ratio.
        d_eff = lambda var_ratio_vec : sum(var_ratio_vec)**2/sum(var_ratio_vec**2)

        dim = []
        for L_current in L:
            var_vec = self.sym_align_eigvec(kappa, beta, L_current+1, num_sample)
            dim_current = d_eff(var_vec[:M])
            dim.append(dim_current)
        return np.array(dim)



    def noise_dim_analytical(self, kappa, beta):
        #L_list = [i for i in range(0, 92)] + [i for i in range(93, int(self.neuron/2))]
        # Apply the variance ratios of the response evoked by white noise. Those are already sorted
        # in descending order.
        return self.analytical_dim(kappa, beta, self.var_ratio)


    # Case of using PCs of spontaneous activity evoked by white noise.
    def noise_dim_empir(self, kappa, beta, num_sample):
        M = kappa * beta
        # The range of L (the feedforward-alignment).
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!
        # Calculate the dimentionality for a certain variance ratio.
        d_eff = lambda var_ratio_vec : sum(var_ratio_vec)**2/sum(var_ratio_vec**2)

        dim = []
        for L_current in L:
            var_vec = self.noise_align_eigvec(kappa, beta, L_current+1, num_sample)
            dim_current = d_eff(var_vec[:M])
            dim.append(dim_current)
        return np.array(dim)



#############################################################################################################
# Alignment to the spontaneous activity.
class AlignmentSpontaneousAct:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.network = network
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eig(self.interaction) # Get normalized eigenvectors.
        # Sort eigenvectors in the order of descending eigenvalues.
        sorted_indices = np.argsort(self.eigval)[::-1]
        self.sorted_eigvec = self.eigvec[:, sorted_indices]
        self.sorted_eigval = self.eigval[sorted_indices]

        # For the case of symmetrizing the original interaction matrix J.
        # Symmetrisize the original asymmetrical interaction matrix.
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec. They are both real.
        self.sym_eigval, self.sym_eigvec = np.linalg.eigh(sym_inter)

        # For the case of applying the response evoked by white noise.
        # Construct the covariance matrix of white noise activity.
        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction)
        act_cov = np.matmul(steady_inter, steady_inter.T)
        # Get the eigenvalues and eigenvectors of the covariance matrix --> PCs of the response.
        cov_eigval, cov_eigvec = np.linalg.eigh(act_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        self.var_ratio = cov_eigval[sort_index]
        self.pc = cov_eigvec[:, sort_index]


    def spont_act(self, kappa, beta_spont, basis_vec, L, num_sample):
        # Determine the upper limit factor M.
        M = kappa * beta_spont

        # Calculate the input variance Sigma^spont.
        sigma_spont = np.zeros((self.neuron, self.neuron))
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta_spont)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = basis_vec[:, i].reshape(-1,1)
            sigma_spont += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.

        # Generate the spontaneous activity based on the input.
        new_interact = np.linalg.inv(np.identity(self.neuron)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_spont @ new_interact.T # (1-J)^(-1)*sigma_spont*(1-J)^(-T)
        rng = np.random.default_rng(seed=42)
        act_vec = rng.multivariate_normal(np.full(self.neuron, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.


    def align_A_to_B(self, act_patternA, act_patternB):
        """
        Calculation of alignment between two activity patterns using vectorised operations.
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


    def pattern_align_score(self, kappa, spont_act, num_sample, beta_dim, basis_vec):
        # Basic vectors should be sorted (as descending sorted eigenvectors).
        # The range of L (the feedforward-alignment).

        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!

        # Storing the pattern to pattern alignment scores.
        pattern_align = []

        for L_current in L: # L starts with 0.
            # Access the evoked activity under the given L_current using sorted eigenvectors.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, basis_vec, num_sample)
            pattern_align.append(self.align_A_to_B(current_act, spont_act))

        return np.array(pattern_align) # (len(L) < n_neuron,) dimensional array.


    def real_pattern_align(self, kappa, beta_spont, beta_dim, num_sample):
        # Generate the spontaneous activity with real part of eigenvalues.
        basis_vec = np.real(self.sorted_eigvec)

        # Calculate the ffrec for basis_vec.
        pair_ffrec = basis_vec.T @ self.interaction @ basis_vec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()

        spont_act = self.spont_act(kappa, beta_spont, basis_vec, 1, num_sample)
        pattern_align = self.pattern_align_score(kappa, spont_act, num_sample, beta_dim, basis_vec)
        return np.flip(ffrec_align[:len(pattern_align)]), np.flip(pattern_align) # Flip in the ascending order.


    def sym_pattern_align(self, kappa, beta_spont, beta_dim, num_sample):
        #L_list = [i for i in range(0, 56)] + [i for i in range(57, int(self.neuron/2))]
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!
        # Sort the eigenvectors of symmetrized J in descending order corresponds eigenvalues.
        sorted_indices = np.argsort(self.sym_eigval)[::-1]
        # Apply the sorted eigenvectors of J_sym to generate spont.act and evok act.
        basis_vec = self.sym_eigvec[:, sorted_indices]
        # Spontaneous defined aligned with L = 1.
        spont_act = self.spont_act(kappa, beta_spont, basis_vec, 1, num_sample)
        pattern_align = self.pattern_align_score(kappa, spont_act, num_sample, beta_dim, basis_vec)

        # Calculate the ffrec with basis_vec.
        pair_ffrec = basis_vec.T @ self.interaction @ basis_vec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()

        #return np.flip(ffrec_align[L]), np.flip(pattern_align) # Flip in the ascending order.
        return np.linspace(0, 1, len(L)), np.flip(pattern_align) # Flip in the ascending order.


    def noise_pattern_align(self, kappa, beta_spont, beta_dim, num_sample):
        L = np.array([i for i in range(int(self.neuron/2))]) # L starts with 0 here!
        # Apply the PCs of response evoked by white noise to construct the spont_act and evoked activity.
        # The PCs are already ordered in descending order.
        basis_vec = self.pc
        # Spontaneous activity defined by L = 1.
        spont_act = self.spont_act(kappa, beta_spont, basis_vec, 1, num_sample)
        pattern_align = self.pattern_align_score(kappa, spont_act, num_sample, beta_dim, basis_vec)

        # Calculate the ffrec with basis_vec.
        pair_ffrec = basis_vec.T @ self.interaction @ basis_vec
        # Extract the diagonal elements for the defined ffrec alignment score for one eigenvector.
        ffrec_align = pair_ffrec.diagonal()

        return np.flip(ffrec_align[L]), np.flip(pattern_align) # Flip in the ascending order.




#############################################################################################################
if __name__ == "__main__":
    n_neuron = 200
    R = 0.85
    sigma_trial = 0.05 # determines the curve. 0.02 will lead to a "diagonal line".
    sigma_time = 0.3
    N_trial = 100
    dt_euler = 0.1
    dt_intra = 20
    T = 120
    kappa = 5
    beta_dim = 10
    beta_spont = 20
    num_sample = 500
    # For the case of low rank interaction.
    sigma_1 = 1
    sigma_2 = 1

    # TODO: find out reason for low ffrec but high ttc and its with symmetrized J and white noise.

    # Symmetrized J.
    network = AN.LowRank(n_neuron, 1, R, sigma_1, sigma_2) # Try with small network. Also have the effect with low ffrec but high ttc.
    # ttc

    ttc = TrialtoTrialCor(n_neuron, R, network)
    sym_ttc = ttc.sym_ttc_sort_align(sigma_trial, N_trial)
    ttc_ffrec = sym_ttc[0]
    ttc_ttc = sym_ttc[1]
    #print(ttc_ffrec)
    #print(ttc_ttc)
    plt.figure(figsize=(6,5))
    plt.scatter(ttc_ffrec, ttc_ttc, alpha=0.5)
    #plt.xlabel("Feedforward recurrent alignment", fontsize=20, labelpad=15)
    plt.ylabel("Trial-to-trial correlation", fontsize=20)
    plt.xticks([-5, 0, 5], fontsize=15)
    plt.yticks([0.5, 1], fontsize=15)
    plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
    plt.show()

    '''
    # Find out the eigenvalues and eigenvectors that have ffrec < 0  but ttc > 0.8 -- depends on the observation of plots.
    find_index = []
    for i in range(len(ttc_ttc)):
        if ttc_ffrec[i] < 0 and ttc_ttc[i] > 0.8:
            find_index.append(i)
    print(sym_eigval[find_index])
    print(ttc_ffrec[find_index])
    print(ttc_ttc[find_index])
    print(sym_eigvec[find_index])
    '''

    '''
    # its
    its = IntraTrialStab(20, R, network)
    sym_its = its.sym_sort_stab(dt_euler, dt_intra, T, sigma_time)
    its_ffrec = sym_its[0]
    its_its = sym_its[1]
    sym_eigval = sym_its[2]
    sym_eigvec = sym_its[3]
    # Find out the eigenvalues and eigenvectors that have ffrec < 0  but its > 0.5 -- depends on the observation of plots.
    find_index = []
    for i in range(len(its_its)):
        if its_ffrec[i] < 0 and its_its[i] > 0.5:
            find_index.append(i)
    print(sym_eigval[find_index])
    print(its_ffrec[find_index])
    print(its_its[find_index])
    print(sym_eigvec[find_index])


    # White noise
    # ttc
    
    ttc = TrialtoTrialCor(20, R, network)
    noise_ttc = ttc.noise_ttc_sort_align(sigma_trial, N_trial)
    ttc_ffrec = noise_ttc[0]
    ttc_ttc = noise_ttc[1]
    var_ratio = noise_ttc[2]
    pc = noise_ttc[3]
    # Find out the eigenvalues and eigenvectors that have ffrec < 0  but ttc > 0.8 -- depends on the observation of plots.
    find_index = []
    for i in range(len(ttc_ttc)):
        if ttc_ffrec[i] > 0 and ttc_ttc[i] > 0.8:
            find_index.append(i)
    print(var_ratio[find_index])
    print(ttc_ffrec[find_index])
    print(ttc_ttc[find_index])
    print(pc[find_index])
    
    #its
    its = IntraTrialStab(20, R, network)
    noise_its = its.noise_sort_stab(dt_euler, dt_intra, T, sigma_time)
    its_ffrec = noise_its[0]
    its_ttc = noise_its[1]
    var_ratio = noise_its[2]
    pc = noise_its[3]
    # Find out the eigenvalues and eigenvectors that have ffrec < 0  but its > 0.6 -- depends on the observation of plots.
    find_index = []
    for i in range(len(its_ttc)):
        if its_ffrec[i] > 0 and its_ttc[i] > 0.6:
            find_index.append(i)
    print(var_ratio[find_index])
    print(its_ffrec[find_index])
    print(its_ttc[find_index])
    print(pc[find_index])
    '''

