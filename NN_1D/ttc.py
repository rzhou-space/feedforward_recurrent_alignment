import matplotlib.pyplot as plt
import numpy as np
from Networks import LinearRecurrentNetwork

class TrialToTrialCor:

    def __init__(self, n, R):

        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)


    # TODO: error here?
    def trial_response(self, h_det, sigma_trial, N_trial):
        """
        The response vector given the trial input vector. Multivariate normal distributed.
        """
        new_inter = np.linalg.inv(np.identity(self.neurons) - self.interaction) # (1-J)^(-1).
        # mean = np.matmul(new_inter, h_det)  # 1xn dimensional array.
        mean = new_inter @ h_det
        cov = (sigma_trial**2)*(np.matmul(new_inter, new_inter.T))
        # Response vector normally distributed.
        rng = np.random.default_rng(seed = 42)
        r_trial = rng.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial



    def ttc_cor(self, r_trial):
        """
        Correlations of pairwise trials.
        """
        # Get the covariance matrix for trials.
        trial_cor = np.corrcoef(r_trial) # N_trial x N_trial dimensional
        # Take the upper right triangle without diagonal elements.
        upr_triangle = trial_cor[np.triu_indices(trial_cor.shape[0], k=1)]
        return upr_triangle, np.mean(upr_triangle) # single trial correlation, beta/trial to trial correlation.



    def hist_cor_distribution(self, sigma_trial, N_trial):
        # Get the correlation values under random input and maximal aligned input.

        # Random input.
        rng = np.random.default_rng(seed = 42)
        h_rand = rng.normal(0, 1, size = n_neuron)
        h_rand = h_rand/np.linalg.norm(h_rand)
        # Response by random input.
        r_rand = self.trial_response(h_rand, sigma_trial, N_trial)
        # Correlation of random response.
        cor_rand = self.ttc_cor(r_rand)[0]

        # Maximal aligned input.
        h_max = self.eigvec[:, np.argmax(self.eigval)]
        h_max = h_max/np.linalg.norm(h_max)
        # Response by maximal aligned input.
        r_max = self.trial_response(h_max, sigma_trial, N_trial)
        # Correlation of maximal aligned response.
        cor_max = self.ttc_cor(r_max)[0]

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



    def rand_cor_ffrec(self, sigma_trial, N_trial):
        rand_inputs = np.random.normal(0, 1, size = (N_trial, self.neurons)) # N_trial x n_neurons array. 100 random input samples.
        mean_ttc = np.zeros(N_trial)
        ffrec = np.zeros(N_trial)
        for i in range(N_trial):
            h = rand_inputs[i]/np.linalg.norm(rand_inputs[i])
            all_resp = self.trial_response(h, sigma_trial, N_trial)
            mean_ttc[i] = self.ttc_cor(all_resp)[1]
            ffrec[i] = h @ self.interaction @ h

        plt.scatter(ffrec, mean_ttc)
        plt.show()



    def ttc_sort_align(self, sigma_trial, N_trial):
        """
        Calculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        result = []
        result_ffrec_align = []
        for i in sort_index:
            h_det = self.eigvec[:, i]
            all_response = self.trial_response(h_det, sigma_trial, N_trial)
            result_ffrec_align.append(h_det @ self.interaction @ h_det)
            result.append(self.ttc_cor(all_response)[1])
        return np.asarray(result_ffrec_align), np.array(result)



    def scatter_plot(self, align_vec, ttc_vec):
        """
        Scatter plot tool for ttc_vec of sorted ttc values.
        """
        plt.figure()
        plt.scatter(align_vec, ttc_vec, alpha = 0.5)
        plt.axvline(x = 0, color = "black")
        # plt.axvline(x = self.R, color = "red", label = "align = R = "+str(self.R))
        plt.xlabel("feedforward alignment")
        plt.ylabel("trial to trial correlation")
        # plt.legend()
        plt.show()



########################################################################################################################
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


    ttc = TrialToTrialCor(n_neuron, R)
    # Distribution of single trial to trial correlation.
    ttc.hist_cor_distribution(sigma_trial, N_trial)
    # Distribution of ffrec under random alignment with 200 trials.
    ttc.rand_cor_ffrec(sigma_trial, N_trial= 200)
    # Correlation of ffrec with sorted eigenvectors.
    sort_align = ttc.ttc_sort_align(sigma_trial, N_trial)
    ttc.scatter_plot(sort_align[0], sort_align[1])
    '''
    sort_align = ttc.ttc_sort_align(sigma_trial, N_trial)
    ttc_vec = sort_align[0]
    ffrec_align = sort_align[1]
    ttc.scatter_plot(ffrec_align, ttc_vec)
    '''


