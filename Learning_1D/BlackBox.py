import matplotlib.pyplot as plt
import matplotlib.colors as cm
import numpy as np
import seaborn as sns
#from Asym_1D import AsymStillWork
from sklearn.decomposition import PCA

class BlackBox:

    def __init__(self, n, R, network):
        self.neuron = n
        self.R = R
        self.interaction = network.interaction

        rng = np.random.default_rng(seed = 42)


        # Generate the random interaction matrix (black box).
        '''
        rand_inter = rng.random((self.neuron, self.neuron)) # n x n dimensional interaction matrix.
        eigval = np.linalg.eigvals(rand_inter)
        # Normalize the eigenvalues to limit the size. Use the magnitude of maximal eigenvalue.
        max_mag = max(map(abs, eigval))
        self.interaction = rand_inter*self.R / max_mag
        # Get the new rescaled eigenvalues and eigenvectors.
        self.eigval = np.linalg.eigvals(self.interaction)
        '''

        self.steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1

        # Apply the symmetrized interaction matrix to get orthogonal bases that could generate the
        # low dimensional input and simulate the steady state output.
        '''
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec. They are both real.
        self.sym_eigval, self.sym_eigvec = np.linalg.eigh(sym_inter)
        # Sort the eigenvectors in descending order corresponding to the eigenvalues.
        sorted_indices = np.argsort(self.sym_eigval)[::-1]
        self.sym_sort_eigvec = self.sym_eigvec[:, sorted_indices]
        '''

        # Construct random orthonomal basis.
        # Firstly generate a random square matrix.
        rand_matrix = rng.normal(0, 1, size=(self.neuron, self.neuron)) # --> have faster learning.
        # rand_matrix = rng.random(size = (self.neuron, self.neuron)) # uniform distributed --> slower learning.
        # Apply the gram-schmit to orthogonalize the random matrix.
        Q,R = np.linalg.qr(rand_matrix)
        # The columns are the basis vectors.
        self.orthobasis = Q


    def sigma_spont(self, kappa, beta_spont, basis_vec, L=1):
        """
        Generate the covariance matrix with low dimensional input (M = kappa * beta)
        dimensional. The contruction was applied in the alignment to spontaneous activity.
        """
        # Determine the upper limit factor M.
        M = kappa * beta_spont
        # Calculate the input variance Sigma^spont.
        sigma_spont = np.zeros((self.neuron, self.neuron))
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta_spont)
            # Getting vectors in columns as basis vector and transform it into nx1 dimensional numpy array.
            e_i = basis_vec[:, i].reshape(-1,1)
            sigma_spont += v_i * (e_i @ e_i.T) # Should be a nxn dimensional numpy matrix.
        return sigma_spont

    # Also one possibility to construct the low dimensional input, however without scaling and
    # with knowing the interaction matrix (which is not the case here).
    '''
    def sigma_low_dim(self, dim, basis_vec):
        # Choose randomly dim number of eigenvectors from symmetrized J as dominant modes.
        rng = np.random.default_rng(seed = 42)
        # Upper limit of randint is excluded.
        dominant_index = rng.integers(1, self.neuron+1, size = dim)
        # The chosen dominant eigenvectors/eigenmodes.
        dominant_modes = basis_vec[:, dominant_index]
        sigma_low_dim = np.zeros((self.neuron, self.neuron))
        for i in range(dim):
            e_i = dominant_modes[:, i].reshape(-1,1)
            sigma_low_dim += e_i @ e_i.T
        return sigma_low_dim
    '''

    '''
    def plot_eigval(self):
        # plot the eigenvalue distribution.
        plt.figure()
        plt.scatter(self.eigval.real, self.eigval.imag)
        plt.axhline(y=self.R)
        plt.axhline(y=-self.R)
        plt.axvline(x=self.R)
        plt.axvline(x=-self.R)
        plt.show()
    '''

    '''
    def white_noise_response(self, num_sample):
        # TODO: Is the matrix (1-J) ganranteed to be invertible? Seems not! since np.linalg.inv
        ## gives complex number matrix back!
        ## Could I just ask if all imaginary parts = 0 and only take the real part?
        if np.all(np.imag(self.steady_inter) == 0) == True:  # all imaginary parts equals 0.
            real_steady_inter = self.steady_inter.real
            rng = np.random.default_rng(seed = 42)
            spont_act = rng.multivariate_normal(mean = np.zeros(self.neuron),
                                                    cov = np.matmul(real_steady_inter, real_steady_inter.T), # (I-J)^-1 * (I-J)^-T
                                                    size = num_sample)
        else:
            print("Not inversible")
        return spont_act
    '''

    '''
    def n_th_noise_response(self, n_potential, num_sample):
        # Take the spontaneous activity evoked by white noise as input. The steady state responses could
        # be again applied as input. Run this circle n-times could get the n-th steady state response.
        n_cov = np.matmul(np.linalg.matrix_power(self.steady_inter, n_potential+1),
                              np.linalg.matrix_power(self.steady_inter.T, n_potential+1))  # ((1-J)^-1)^n+1 * ((1-J)^-T)^n+1
        rng = np.random.default_rng(seed = 42)
        n_response = rng.multivariate_normal(mean = np.zeros(self.neuron),
                                             cov = n_cov, size = num_sample)
        # Normalize the n_responses row-wise, i.e., each response sample would be normalized.
        row_norms = np.linalg.norm(n_response, axis=1)
        n_response = n_response / row_norms[:, np.newaxis]
        return n_response # (n_sample, n_neuron) dimensional array.
    


    def noise_align_ffrec(self, n_turn, num_sample):
        # Calculate the ffrec for responses after applying n times the response evoked by white noise.
        n_response = self.n_th_noise_response(n_turn, num_sample)  # With normalized rows.
        # For each row, calculate the ffrec. n_row = num_sample.
        ffrec = np.zeros(num_sample)
        for i in range(num_sample):
            input = n_response[i]
            ffrec[i] = input @ self.interaction @ input
        return ffrec


    def noise_histogram_ffrec(self, n_list, num_sample):
        # n_list is a list contains the number of turns that should be applied to generate the n_responses.
        fig, axs = plt.subplots(len(n_list), sharex=True, sharey=True)
        fig.suptitle("Distribution of ffrec")
        for i in range(len(n_list)):
            n = n_list[i]
            ffrec = self.noise_align_ffrec(n, num_sample)
            axs[i].hist(ffrec)
            axs[i].set_title("n = "+ str(n))

        fig.supxlabel("ffrec")
        fig.supylabel("frequency")
        plt.show()
    
    '''

    def n_th_response(self, n_turn, cov_sigma, num_sample):
        # The covariance matrix for the n-th response after n-turns of applying prior repsonses as inputs.
        # ((1-J)^-1)^n+1 * sigma_spont * ((1-J)^-T)^n+1
        n_cov = np.linalg.matrix_power(self.steady_inter, n_turn+1) @ cov_sigma \
                @ np.linalg.matrix_power(self.steady_inter.T, n_turn+1)
        # Generate the responses.
        rng = np.random.default_rng(seed = 42)
        n_response = rng.multivariate_normal(mean = np.zeros(self.neuron),
                                             cov = n_cov, size = num_sample)
        # Normalize the n_responses row-wise, i.e., each response sample would be normalized. -- Think not necessary.
        row_norms = np.linalg.norm(n_response, axis=1)
        n_response = n_response / row_norms[:, np.newaxis]
        return n_response, n_cov # (n_sample, n_neuron) dimensional array.


    def align_ffrec(self, n_turn, cov_sigma, num_sample):
        # Calculate the ffrec for responses after applying n-1 times the prior responses.
        n_input = self.n_th_response(n_turn-1, cov_sigma, num_sample)[0]  # With normalized rows.
        # For each row, calculate the ffrec. n_row = num_sample.
        ffrec = np.zeros(num_sample)
        for i in range(num_sample):
            input = n_input[i]
            ffrec[i] = input @ self.interaction @ input
        return ffrec  # (num_sample, ) dimensional


    def align_ffrec_PCA(self, n_turn, cov_sigma, num_sample):
        '''
        # Calculate the ffrec for responses after applying n times the prior responses with the PC.
        n_responses = self.n_th_response(n_turn, cov_sigma, num_sample)[0].T # (n_neuron, n_sample) dimenaional.
        # Do PCA on the n_resposes.
        pca = PCA(n_components=self.neuron)
        PCs = pca.fit_transform(n_responses)
        variance_ratio = pca.explained_variance_ratio_
        '''
        n_cov = self.n_th_response(n_turn, cov_sigma, num_sample)[1]
        cov_eigval, cov_eigvec = np.linalg.eigh(n_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        variance_ratio = cov_eigval[sort_index]
        variance_ratio = variance_ratio/np.linalg.norm(variance_ratio)
        PCs = cov_eigvec[:, sort_index]

        ffrec = np.zeros(len(variance_ratio))
        for i in range(len(variance_ratio)):
            h = PCs[:, i]/np.linalg.norm(PCs[:, i])
            ffrec[i] = h @ self.interaction @ h
        return variance_ratio, ffrec


    def PCA_n_th_response(self, n_list, beta_list, num_sample, kappa):
        colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
        fig, axs = plt.subplots(len(n_list), sharex=True)#, sharey=True)
        fig.suptitle("Variance Ratio of Responses")
        for j in range(len(beta_list)):
            beta_spont = beta_list[j]
            cov_sigma = self.sigma_spont(kappa, beta_spont, self.orthobasis)
            for i in range(len(n_list)):
                n = n_list[i]
                var_ratio = self.align_ffrec_PCA(n, cov_sigma, num_sample)[0]
                axs[i].plot(var_ratio[:5], color = colors[j], label = "dim="+str(beta_spont) if i==0 else "")
                axs[i].set_title("n = "+ str(n))
        fig.supxlabel("PC")
        fig.supylabel("variance ratio")
        fig.legend()
        plt.show()


    def ffrec_PC_multi_dim(self, n_list, beta_list, num_sample, kappa):
        colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
        fig, axs = plt.subplots(len(n_list), sharex=True, sharey=True)
        fig.suptitle("ffrec of Responses based on PC")
        for j in range(len(beta_list)):
            beta_spont = beta_list[j]
            cov_sigma = self.sigma_spont(kappa, beta_spont, self.orthobasis)
            for i in range(len(n_list)):
                n = n_list[i]
                pca_result = self.align_ffrec_PCA(n, cov_sigma, num_sample)
                var_ratio = pca_result[0]
                ffrec = pca_result[1]
                axs[i].plot(var_ratio[:5], ffrec[:5], color = colors[j], label = "dim="+str(beta_spont) if i==0 else "")
                axs[i].set_title("n = "+ str(n))
        fig.supxlabel("Variance ratio")
        fig.supylabel("ffrec")
        fig.legend()
        plt.show()


    def histogram_ffrec(self, n_list, cov_sigma, num_sample):
        # n_list is a list contains the number of turns that should be applied to generate the n_responses.
        fig, axs = plt.subplots(len(n_list), sharex=True, sharey=True)
        fig.suptitle("Distribution of ffrec")
        for i in range(len(n_list)):
            n = n_list[i]
            ffrec = self.align_ffrec(n, cov_sigma, num_sample)
            axs[i].hist(ffrec)
            axs[i].set_title("n = "+ str(n))

        fig.supxlabel("ffrec")
        fig.supylabel("frequency")
        plt.show()


    def histogram_ffrec_multiple_dim(self, n_list, beta_list, num_sample, kappa):
        # n_list is a list contains the number of turns that should be applied to generate the n_resposnes.
        # beta_list contains the dimensionalities that will be compared.
        #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
        colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
        fig, axs = plt.subplots(len(n_list), sharex=True, sharey=True)
        fig.suptitle("Distribution of ffrec")
        for j in range(len(beta_list)):
            beta_spont = beta_list[j]
            cov_sigma = self.sigma_spont(kappa, beta_spont, self.orthobasis)
            for i in range(len(n_list)):
                n = n_list[i]
                ffrec = self.align_ffrec(n, cov_sigma, num_sample)
                # For each beta_spont a color in histogram for each step n.
                axs[i].hist(ffrec, color=colors[j], alpha=0.5, label="dim="+str(beta_spont) if i==0 else "")
                axs[i].set_title("n = "+ str(n))
        fig.supxlabel("ffrec")
        fig.supylabel("frequency")
        fig.legend()
        plt.show()


    # Switch i and j brings the same consistant results as above.
    '''
    def histogram_ffrec_multiple_dim(self, n_list, beta_list, num_sample, kappa):
        # n_list is a list contains the number of turns that should be applied to generate the n_resposnes.
        # beta_list contains the dimensionalities that will be compared.
        #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
        colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
        fig, axs = plt.subplots(len(n_list), sharex=True, sharey=True)
        fig.suptitle("Distribution of ffrec")
        for i in range(len(n_list)):
            n = n_list[i]
            for j in range(len(beta_list)):
                beta_spont = beta_list[j]
                cov_sigma = self.sigma_spont(kappa, beta_spont, self.orthobasis)
                ffrec = self.align_ffrec(n, cov_sigma, num_sample)
                # For each beta_spont a color in histogram for each step n.
                axs[i].hist(ffrec, color=colors[j], alpha=0.5, label="dim="+str(beta_spont) if i==0 else "")
                axs[i].set_title("n = "+ str(n))
        fig.supxlabel("ffrec")
        fig.supylabel("frequency")
        fig.legend()
        plt.show()
    '''


    # Functions for mean_ffrec plot against steps for multiple dimensionas.
    def mean_ffrec_step(self, num_step, kappa, beta_dim, num_sample):
        """
        Calculate the mean ffrec for each step under a certain dimensionality.
        """
        mean_ffrec = []
        cov_sigma = self.sigma_spont(kappa, beta_dim, self.orthobasis)
        for n_turn in range(num_step):
            mean_ffrec.append(np.mean(self.align_ffrec(n_turn, cov_sigma, num_sample)))
        return mean_ffrec # (num_step, ) dimensional.

    def mean_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        '''
        Calcualte for multiple dimensions the mean ffrec over num_samples.
        '''
        mean_ffrec_multi_dim = []
        for dim in beta_list:
            mean_ffrec_multi_dim.append(self.mean_ffrec_step(num_step, kappa, dim, num_sample))
        return np.array(mean_ffrec_multi_dim) # (num_dim, num_step) dimensional.

    def plot_mean_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        """
        Plot the mean_ffrec curves against steps for different dimensions.
        """
        mean_ffrecs = self.mean_ffrec_step_multi_dim(num_step, kappa, beta_list, num_sample) # Get (num_dim, num_step) dimensional array.
        plt.figure()
        for i in range(len(beta_list)):
            plt.plot(mean_ffrecs[i], label="dim = " + str(beta_list[i]))
            plt.xticks(range(0,num_step))
        plt.xlabel("step")
        plt.ylabel("mean ffrec")
        plt.legend()
        plt.show()

    def plot_ffrec_step_multi_dim(self, num_step, kappa, beta_list, num_sample):
        plt.figure(figsize=(6,5))
        plt.xlabel("Step", fontsize=18)
        plt.ylabel("Feedforward recurrent alignment", fontsize=18)
        for beta_dim in beta_list:
            cov_sigma = self.sigma_spont(kappa, beta_dim, self.orthobasis)
            all_ffrec = [] # Stores at each step num_sample of ffrec value.
            x_step = [] # x-axis value for each value in all_ffrec.
            for n_turn in range(1, num_step):
                # Final length of all_ffrec, x_step would be num_step*num_sample.
                all_ffrec += self.align_ffrec(n_turn, cov_sigma, num_sample).tolist()
                x_step += [n_turn] * num_sample # Repeat n_turn for num_step times.
            # Lineplot with error interval.
            sns.lineplot(x=x_step, y=all_ffrec) #, label="dim = "+ str(beta_dim))
            plt.xticks([1, 20, 40, 60, 80, 100], fontsize=15)
            plt.yticks([0.7, 0.76, 0.82], fontsize=15)
        #plt.legend()
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()

# TODO: Only firstly for real recurrent interaction network. Test and plot.
    def variance_ratio(self, n_list, kappa, beta_spont, n_sample):
        cov_sigma = self.sigma_spont(kappa, beta_spont, self.orthobasis)
        # Eigenvectors in sorted descending order corresponds to eigenvalues.
        eigval_sym, eigvec_sym = np.linalg.eigh(self.interaction)
        sort_index = np.argsort(eigval_sym)[::-1]
        sorted_eigval = eigval_sym[sort_index]
        sorted_eigvec = eigvec_sym[sort_index]

        var_ratio = []
        for n in n_list:
            # The covariance matrix of the n-th responses.
            n_th_cov = self.n_th_response(n, cov_sigma, n_sample)[1]
            eigvec_proj = []
            for i in range(self.neuron):
                eigvec = sorted_eigvec[:,i].reshape(self.neuron, 1)
                eigvec_proj.append(eigvec.T @ n_th_cov @ eigvec)
            var_ratio.append(eigvec_proj)
        return var_ratio




############################################################################################################
'''
def noise_response(n, R, kappa, beta_dim, num_sample):
    network = BlackBox(n, R)
    # Apply the principal components of responses to generate activity.
    # L = 1
    dim_class = AsymStillWork.Dimensionality(n, R, network)
    return dim_class.evok_activity(kappa, beta_dim, 1, dim_class.pc, num_sample)
'''
############################################################################################################
if __name__ == "__main__":
    n_neuron = 200
    R = 0.85
    kappa = 5
    beta_spont = 20
    beta_dim = 10
    num_sample = 500

    from NN_1D import Networks
    from Asym_1D import AsymNetworks
    network_sym = Networks.LinearRecurrentNetwork(n_neuron, R)
    network_asym = AsymNetworks.AsymLinearRecurrentNet(n_neuron, R)
    test = BlackBox(n_neuron, R, network_sym)
    test_asym = BlackBox(n_neuron, R, network_asym)

    #print(test.steady_inter)
    #print(test.white_noise_response(num_sample))
    #test.plot_eigval()
    #cov_sigma = test.sigma_spont(kappa, beta_spont, test.orthobasis)
    #print(test.n_th_response(3, cov_sigma, num_sample).shape)
    #print(test.align_ffrec(4, cov_sigma, num_sample))
    #test.histogram_ffrec([0,1,2,3], cov_sigma, num_sample)
    #basis_vec = test.sym_sort_eigvec
    #print(test.n_th_response(3, kappa, beta_spont, basis_vec, num_sample).shape)
    #print(test.align_ffrec(2, kappa, beta_spont, basis_vec, num_sample))
    #test.histogram_ffrec([0,1,2,5], kappa, beta_spont, basis_vec, num_sample)
    #test.noise_histogram_ffrec([0,1,2,6], num_sample)
    #test.histogram_ffrec_multiple_dim([0,1,2,3], [1,2,3,4], 30, kappa)


    #print(test.align_ffrec_PCA(4, cov_sigma, num_sample))
    #test.PCA_n_th_response([0,1,2,3], [1,2,3,4], num_sample, kappa)
    #test.ffrec_PC_multi_dim([0,1,2,3], [1,2,3,4], num_sample, kappa)

    #print(test.mean_ffrec_step(3, kappa, 10, 10))
    #print(test.mean_ffrec_step_multi_dim(5, kappa, [10, 20, 40], num_sample))
    #test.plot_mean_ffrec_step_multi_dim(5, kappa, [10, 20, 40], num_sample)

    #test.plot_ffrec_step_multi_dim(5, kappa, [10, 20, 40], num_sample)
    test_asym.plot_ffrec_step_multi_dim(100, kappa, [5], num_sample)




