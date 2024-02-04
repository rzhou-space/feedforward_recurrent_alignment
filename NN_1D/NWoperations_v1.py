import matplotlib.pyplot as plt
import numpy as np
from Networks import LinearRecurrentNetwork

class TrialToTrialCor:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param mode: There are three modes: max_align, random_align, and sort_align.
        """
        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        #self.eigval = self.R*self.eigval/np.amax(self.eigval)

        #self.eigval, self.eigvec = np.linalg.eig(self.network.interaction)
        #self.interaction = self.R*self.network.interaction/np.amax(self.eigval)


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
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)


    def ttc_random_align(self, sigma_trial, N_trial, repeats):
        """
        Calculation of the mean trial to trial correlation under random alignment of the interaction
        matrix.
        :return: An array containing the number of repeats of trial to trial correlation.
        """
        result = []
        # TODO: random vector instead of random eigenvector. Set the seed for random.

        # random vectors with elements normal distributed with mean 0 and variance 1.
        np.random.seed(42)
        rand_h = np.random.normal(0, 1, size = self.neurons)
        h_det = rand_h/np.linalg.norm(rand_h)

        #index = np.random.randint(0, len(self.eigval), size = repeats)
        #for i in index:
        for i in range(repeats):
            # Choose a random eigenvector as h_det.
            #h_det = self.eigvec[:, i]
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)


    def ttc_sort_align(self, sigma_trial, N_trial):
        """
        Caculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        :return: An array containing a trial to trial correlation for each h_det.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        result = []
        for i in sort_index:
            h_det = self.eigvec[:, i]
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
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


    def scatter_plot(self, ttc_vec):
        """
        Scatter plot tool for ttc_vec of sorted ttc values.
        :param ttc_vec: the numpy array containing ttc values from sorted alignment.
        :return: a scatter plot with x-axis the ordered alignment score = eigenvalues and y-axis the ttc values.
        """
        sort_eigval = np.sort(self.eigval)
        print(sort_eigval)
        plt.figure()
        for i in range(len(ttc_vec)):
                plt.scatter(sort_eigval[i], ttc_vec[i], c="blue", alpha = 0.5)
        plt.axvline(x = 0, color = "black")
        plt.axvline(x = self.R, color = "red", label = "align = R = "+str(self.R))
        plt.xlabel("alignment")
        plt.ylabel("trial to trial correlation")
        plt.legend()
        plt.show()



class IntraTrialStab:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param mode: There are three modes: max_align, random_align, and sort_align.
        """
        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)


    def EulerMaruyama(self, sigma_trial, Delta_t_euler, h_det, T):
        """
        Euler-Maruyama Scheme for iteratively solve SDE.
        :param sigma_trial:
        :param Delta_t:
        :param h_det:
        :return:
        """
        r0 = np.dot(np.linalg.inv(np.identity(self.neurons) - self.interaction), h_det) # (1-J)^(-1)*h_det
        steps = int(T/Delta_t_euler)
        r_values = [r0]

        for i in range(int(steps-1)):
            Delta_W = np.random.normal(0, np.sqrt(Delta_t_euler), size = self.neurons)
            # Euler-Maruyama Scheme.
            r_values.append(r_values[i] + (-r_values[i]+np.dot(self.interaction, h_det))*Delta_t_euler
                            + sigma_trial*Delta_W)

        return np.array(r_values) # Should be kxn dimensional.


    def intra_trial_stability(self, Delta_t_intra, T, Delta_t_euler, sigma_trial, h_det):
        """
        Calculate the intra trial stability.
        :param Delta_t_intra:
        :param T:
        :param Delta_t_euler:
        :param sigma_trial:
        :param h_det:
        :return:
        """
        intra_c = []
        # The steps that goes in euler approximated solution.
        step_width = int(Delta_t_intra/Delta_t_euler)
        # The total steps that iterate in euler appeoximated solution.
        steps = int((T-Delta_t_intra)/Delta_t_euler)

        r_values = self.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T)

        for t in range(int(steps)):
            # z-scored response.
            r_t = (r_values[t] - r_values[t].mean())/r_values[t].std()
            r_tdt = (r_values[t+step_width] - r_values[t+step_width].mean())/r_values[t+step_width].std()
            intra_c.append(np.dot(r_t, r_tdt))

        return (intra_c, np.average(intra_c))











if __name__ == "__main__":

    # Global parameters:
    sigma_trial = 0.02
    N_trial = 100
    n_neuron = 200
    R = 0.85
    Delta_t_euler = 0.1
    Delta_t_intra = 20
    T = 120

    ########################################################################################################
    # Trial to trial correlation.

    obj = TrialToTrialCor(n_neuron, R)
    max_align = obj.ttc_max_align(sigma_trial, N_trial, 100)
    print(max_align)
    print(np.shape(max_align))

    rand_align = obj.ttc_random_align(sigma_trial, N_trial, 100)
    print(rand_align)
    print(np.shape(rand_align))

    obj.hist_plot(max_align, rand_align)

    sort_align = obj.ttc_sort_align(sigma_trial, N_trial)
    print(sort_align)
    print(np.shape(sort_align))

    obj.scatter_plot(sort_align)

    '''
    #########################################################################################################
    # Intra trial stability.

    obj_stab = IntraTrialStab(n_neuron, R)
    # Take random h_det as an example.
    h_det = np.random.normal(0,1,size = n_neuron)
    r_value = obj_stab.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T)
    print(r_value)
    print(np.shape(r_value))

    c_value = obj_stab.intra_trial_stability(Delta_t_intra, T, Delta_t_euler, sigma_trial, h_det)
    print(c_value[0])
    print(np.shape(c_value[0]))
    print(c_value[1])
    '''













