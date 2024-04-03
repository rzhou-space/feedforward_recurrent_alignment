import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Asym_1D import AsymNetworks as AN # Asymmetrical recurrent interaction.
from NN_1D import Networks as SN # Symmetrical recurrent interaction.
from mpl_toolkits.axes_grid1 import ImageGrid


class Hebb_sym_FFinteraction: # Inputs are guassian distributed samples.
    """
    Consider firstly the case with symmetrical feedforward interaction, i.e., the number of input neuron
    equals the output number. And the interaction has the special case of symmetrical pattern.
    """

    def __init__(self, n, R):
        self.inout_neuron = n
        self.R = R
        # The recurrent interaction is here asymmetrical.
        self.recurrent_interaction = AN.AsymLinearRecurrentNet(n,R).interaction
        self.feedforward_interaction = self.feedforward_interaction_sym()
        self.steady_inter = np.linalg.inv(np.eye(self.inout_neuron) - self.recurrent_interaction) # (1-J)^-1

    def feedforward_interaction_sym(self):
        rng = np.random.default_rng(seed = 42)
        W = rng.normal(0, 1, size = (self.inout_neuron, self.inout_neuron))
        # W a symmetric matrix.
        W = (W + W.T)/2
        #normalize by largest eigenvalue
        eigvals = np.linalg.eigvals(W)
        W = W*self.R/np.max(eigvals)
        return W # (n,n) dimensional

    def input_samples(self, sample_size):
        rng = np.random.default_rng(seed = 42)
        u = rng.normal(0, 1, size = (self.inout_neuron, sample_size)) # (#neuron, sample_size) dimensional
        # Calculate the input autocorrelation between neurons.
        Q = np.corrcoef(u) # By default row-wise correlations.
        return u, Q

    def steady_recurrent_response(self, W_t, input_samples):
        # W_t: (n,n) dimentinoal, input_samples: (n, sample_size) dimensional.
        h_rrec_input = np.matmul(W_t, input_samples) # (n, sample_size) dimensional
        steay_response = np.matmul(self.steady_inter, h_rrec_input) # (n, sample_size) dimensional
        return steay_response

    def euler_weight_update(self, delta_t, W_old, Q_old):
        # delta_t = sample_size for this case.
        W_new = W_old + delta_t * self.steady_inter @ W_old @ Q_old
        return W_new # (n, n) dimensional

    def weight_response_update(self, total_T, delta_t, sample_size):
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # include the last time point total_T.
        # Initial conditions:
        W0 = self.feedforward_interaction_sym()
        W = [W0]
        # Generate the inital response extra.
        initial_input = self.input_samples(sample_size)
        u0 = initial_input[0]
        Q0 = initial_input[1]
        r0 = self.steady_recurrent_response(W0, u0)
        r = [r0]
        Q = [Q0]
        for i in range(1, len(t_list)):
            t_input = self.input_samples(sample_size)
            u_t = t_input[0]
            Q_t = t_input[1]
            W_t = self.euler_weight_update(delta_t, W[i-1], Q[i-1])
            r_t = self.steady_recurrent_response(W_t, u_t)
            W.append(W_t)
            r.append(r_t)
            Q.append(Q_t)
        return W, r # W: (n, n) dimensional; r: (n, sample_size) dimensional

    def correlation_heatmap(self, matrix, cmap):
        # Calculate the correlation coefficients.
        corr = np.corrcoef(matrix) # Per default row-wise correlation.
        # Show the correlation in a heatmap.
        fig = plt.figure()
        cormap = plt.imshow(corr, cmap=cmap, interpolation="nearest")
        fig.colorbar(cormap)
        plt.show()

    def multiple_heatmaps(self, matrix_list, cmap, total_T, delta_t):
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # include the last time point total_T.
        # Set up figure and image grid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                         nrows_ncols=(1,len(matrix_list)),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        # Add data to image grid
        for i in range(len(matrix_list)):
            ax = grid[i]
            corr = np.corrcoef(matrix_list[i])
            im = ax.imshow(corr, cmap=cmap, interpolation="nearest")
            ax.set_title("t = "+str(t_list[i]))
        # Colorbar
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        plt.show()


############################################################################################################
class Hebb_asym_FFinteraction:
    """
    For feedforward network: The number of inputs is not necessary equal the number of output neurons.
    """
    def __init__(self, input_n, output_n, sample_size, R):
        self.input_n = input_n
        self.output_n = output_n
        self.sample_size = sample_size
        self.R = R
        # Firstly consider the simple case of random symmetric recurrent interaction.
        self.recurrent_interaction = SN.LinearRecurrentNetwork(output_n, self.R).interaction
        self.J_eigval, self.J_eigvec = np.linalg.eigh(self.recurrent_interaction) # Unsorted eigvals and eigvec.
        self.steady_inter = np.linalg.inv(np.eye(self.output_n) - self.recurrent_interaction) # (1-J)^-1
        self.trans_rec_inter = self.recurrent_interaction @ self.steady_inter # J(1-J)^-1
        self.feedforward_interaction = self.feedforward_interaction_asym() # Initial feedforward interaction.
        #self.feedforward_interaction = self.feedforward_interaction_negative_align()

    def feedforward_interaction_asym(self):
        rng = np.random.default_rng(seed = 42)
        W = rng.normal(0, 1, size = (self.output_n, self.input_n))
        return W

    def feedforward_interaction_negative_align(self):
        trans_rec_eigval, trans_rec_eigvec = np.linalg.eigh(self.trans_rec_inter)
        # Select the negative eigenvalues and eigenvectors out.
        neg_indices = np.where(trans_rec_eigval < 0)[0]
        neg_eigval = trans_rec_eigval[neg_indices]
        neg_eigvec = trans_rec_eigvec[:, neg_indices] # Columns are eigenvectors.
        # Since the trans_rec_inter symmetric full rank. Its eigenvalues should be orthogonal to each other.
        return np.array(neg_eigvec[:,1].reshape((self.output_n,self.input_n)))

    def two_neuron_reverse_input_samples(self):
        """
        The special case of having two input neurons -- the class distribute self.input_n works not.
        Generate for two input neurons the reverse input, i.e., only one of the two neurons have input 1 and
        the other will then have input 0.
        """
        rng = np.random.default_rng(seed = 42)
        input_1 = rng.integers(2, size = self.sample_size)
        input_2 = abs(input_1 - 1)  # If input1 has 1 --> abs(1-1) = 0; if input2 has 0 --> abs(0-1) = 1.
        input_sample = np.array([input_1, input_2]) # (n_input=2, sample_size) dimensional.
        # The autocorrelation of inputs, i.e., the correlation between neurons.
        Q = np.corrcoef(input_sample) # (n_input, n_input) = (2, 2) dimensional.
        return input_sample, Q

    def random_input_samples(self):
        # Two input neurons with independent gaussian distributed inputs.
        rng = np.random.default_rng(seed = 42)
        input_sample = rng.normal(0, 1, size = (self.input_n, self.sample_size))
        Q = np.corrcoef(input_sample) # (n_input, n_input) dimensional.
        return input_sample, Q

    def one_neuron_binary_input_samples(self, active_percent = 0.5):
        # Only one input neuron with binary inputs, i.e., -1 or 1.
        # To avoid the length of h be 0. Which would be problimatic for ffrec calculation.
        rand_num = np.random.uniform(0, 1, self.sample_size)
        input_sample = np.repeat(-1, self.sample_size)
        for i in range(len(rand_num)):
            num = rand_num[i]
            if num <= active_percent:
                input_sample[i] = 1
        # Turn input_sample into (1, num_sample) dimensional array.
        input_sample = input_sample.reshape(1, -1)
        Q = np.corrcoef(input_sample) # = 1
        return input_sample, Q # Input sample (1, num_sample) dimensional. Q = 1.

    def steady_recurrent_response(self, W_t, input_samples):
        # W_t: (n_output,n_input) dimentinoal, input_samples: (n_input, sample_size) dimensional.
        h_rrec_input = W_t @ input_samples # (n_output, sample_size) dimensional.
        # Normalize the h_rrec_input for each sample (along columns).
        sample_norms = np.linalg.norm(h_rrec_input, axis=0)
        h_rrec_input = h_rrec_input/sample_norms
        steady_response = self.steady_inter @ h_rrec_input # (n_output, sample_size) dimensional
        return steady_response, h_rrec_input

    def euler_weight_update(self, delta_t, W_old, Q_old):
        # delta_t = sample_size for this case.
        if self.input_n == 1:
            W_new = W_old + delta_t * self.steady_inter @ W_old * Q_old
        else:
            W_new = W_old + delta_t * self.steady_inter @ W_old @ Q_old
        return W_new # (n_output, n_input) dimensional

    # TODO: Perhaps rewrite the function to a general function and applied under
    ## different modus (= input_sample_type).
    def weight_response_update(self, total_T, delta_t, input_sample_type, W0 = None):
        """
        Update the weights with the euler method. Store stepwise the responses, input autocorrelation,
        and the weight.

        u: inputs.
        W: weights.
        r: final recurrent responses.
        h: feedforward output = recurrent input.

        :param input_sampe_type: string.
               If = "two_reverse" then the input function self.two_neuron_reverse_input_semples.
               elif = "random" then the input function self.random_input_samples.
               elif = "one_binary" then the input function self.one_neuron_binary_input_samples.
        """
        if W0 is None:
            W0 = self.feedforward_interaction
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # include the last time point total_T.
        # Initial conditions:
        W = [W0]
        # Generate the inital response extra.
        # The input generation depends on the input function.
        if input_sample_type == "two_reverse":
            initial_input = self.two_neuron_reverse_input_samples()
        elif input_sample_type == "random":
            initial_input = self.random_input_samples()
        elif input_sample_type == "one_binary":
            initial_input = self.one_neuron_binary_input_samples()
        u0 = initial_input[0]
        Q0 = initial_input[1]
        steady = self.steady_recurrent_response(W0, u0)
        r0 = steady[0]
        h0 = steady[1]
        u = [u0] # input vector.
        r = [r0] # final recurrent responses.
        h = [h0] # feedforward output = recurrent input.
        Q = [Q0] # input autocorrelations.
        for i in range(1, len(t_list)):
            if input_sample_type == "two_reverse":
                t_input = self.two_neuron_reverse_input_samples()
            elif input_sample_type == "random":
                t_input = self.random_input_samples()
            elif input_sample_type == "one_binary":
                t_input = self.one_neuron_binary_input_samples()
            u_t = t_input[0]
            Q_t = t_input[1]
            W_t = self.euler_weight_update(delta_t, W[i-1], Q[i-1])
            steady_t = self.steady_recurrent_response(W_t, u_t)
            r_t = steady_t[0]
            h_t = steady_t[1]
            u.append(u_t)
            W.append(W_t)
            r.append(r_t)
            h.append(h_t)
            Q.append(Q_t)
        return np.array(W), \
               np.array(r), \
               np.array(h), \
               np.array(u)
        # W: (n_step, n_output, n_input) dimensional; r: (n_step, n_output, sample_size) dimensional;
        # h: (n_step, n_output, sample_size) dimenaional; u: (n_step, n_input, sample_size) dimensional.

    def multiple_cor_heatmaps(self, matrix_list, cmap, total_T, delta_t):
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # include the last time point total_T.
        # Set up figure and image grid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                         nrows_ncols=(1,len(matrix_list)),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        # Add data to image grid
        for i in range(len(matrix_list)):
            ax = grid[i]
            corr = np.corrcoef(matrix_list[i])  # row-wise correlation.
            im = ax.imshow(corr, cmap=cmap, interpolation="nearest")
            ax.set_title("t = "+str(t_list[i]))
        # Colorbar
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        plt.show()

    def multiple_vec_heatmaps(self, vector_list, cmap, total_T, delta_t):
        # delta_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # Include the last time point total_T.
        # Set up figure and image grid
        fig, ax = plt.subplots(ncols = len(t_list), figsize = (15, 0.5))
        # len(vector_list) = len(t_list)
        for i in range(len(t_list)):
            vec = vector_list[i]
            # Normalize the vectors through the maximal element so that the comparison between
            # neurons at each time step is clear.
            vec = vec/np.max(vec)
            print(vec)
            sns.heatmap(vec, ax = ax[i], cmap = cmap)#, cbar=False, xticklabels=False
                        #, yticklabels=False)
            ax[i].set_title("t = "+str(t_list[i]))
        plt.show()

    def mean_ffrec(self, h_rrec_input, total_T, delta_t):
        # Calculate the mean ffrec for each time step.
        # h_rrec_input could be generated with self.weight_respones_update
        t_list = list(range(0, total_T+1, delta_t))
        mean_ffrec = np.zeros(len(t_list))
        for t in range(len(t_list)):
            h_rrec_t = h_rrec_input[t]
            ffrec_t = np.zeros(self.sample_size)
            for i in range(self.sample_size):
                h_input = h_rrec_t[:, i]  # h_rrec_t: (n_output, sample_size) dimensional.
                # Normalization of h_input.
                # h_input = h_input/np.linalg.norm(h_input)
                ffrec_t[i] = h_input @ self.recurrent_interaction @ h_input
            print(ffrec_t)
            mean_ffrec[t] = np.mean(ffrec_t)
        # plot the mean ffrec against time.
        plt.figure()
        plt.plot(mean_ffrec)
        plt.xticks(range(0, len(t_list)))
        plt.xlabel("t")
        plt.ylabel("mean ffrec")
        plt.show()
        return mean_ffrec

    def factor_sum_t_one_input(self, W_t, eigval, eigvec):
        # Wt should be (1,n) dimensional numpy array!
        factor_sum = 0
        for i in range(len(eigval)):
            # Eigenvectors are the columns. Must firstly reshaped to (n, 1) dimensional.
            e_i = eigvec[:, i].reshape((-1, 1))
            # W_t @ e_i should be a real number.
            factor_sum += eigval[i] * (W_t @ e_i)**2
        return factor_sum.reshape((1,))

    def derivative_factor_sym_interact_one_input(self, t_list, all_Wt):
        # all_Wt would be generated by weight_response_update --> (t, #output, #input) dimensional array.
        # t_list contains the time steps.
        # Both recurrent interaction and steady interaction are symmetrical.
        tr_eigval, tr_eigvec = np.linalg.eigh(self.trans_rec_inter)
        all_factor = []
        for i in range(len(t_list)):
            # Turn W_t into (1,n) dimensional numpy array.
            W_t = all_Wt[i].reshape(1, self.output_n)
            all_factor.append(self.factor_sum_t_one_input(W_t, tr_eigval, tr_eigvec))
        # Plot the derivative factor along time.
        plt.figure()
        plt.yscale("log")
        plt.plot(all_factor)
        plt.xticks(range(0, len(t_list)))
        plt.xlabel("t")
        plt.ylabel("derivative factor")
        plt.show()
        return all_factor

    def W_projection_rec_eigvec_coeff_1D(self, Wt):
        '''
        Condition: one neuron input samples.
        Cauculate the coefficients that express Wt with eigenvectors of J
        <-> projection of Wt on eigenspace of J.
        Wt = A* coeff where A contains the eigenvectors of J as columns.
        -> coeff = A^-1 * Wt
        :param Wt: (n_output, n_input = 1) dimensional array.
        :return: kappa, the projection coefficients. (n_output, 1) dimensional.
        '''
        # Sort eigvec in A in descending order corresponding to eigenvalues.
        sort_index = np.argsort(self.J_eigval)[::-1]
        A = self.J_eigvec[:, sort_index] # sorted.
        # coeff = A^-1 * Wt
        return np.linalg.inv(A) @ Wt # (n_output, 1) dimensional.

    def all_W_projection_1D(self, all_Wt, t_list):
        """
        Calculate the projektion vector for all Wt in different time steps. Wt_all could be generated with 
        self.weight_response_update(). 
        :param all_Wt: (n_step, n_output, n_input=1) dimensional. 
        """
        coeff_1D = []
        for i in range(len(t_list)):
            Wt = all_Wt[i]
            coeff_values = np.abs(self.W_projection_rec_eigvec_coeff_1D(Wt))
            # See the percentage of the coefficients for the first 20 eigenvectors.
            first_ten = np.sum(coeff_values[:20])/np.sum(coeff_values)
            #coeff_1D.append(coeff_values) # Normalize inside each time step when plotted.
            coeff_1D.append(first_ten)
        #coeff_1D = np.abs(np.array(coeff_1D)) # Take the absolute values of the coefficients.
        return coeff_1D #(n_step, n_output, n_input=1) dimensional or for first_ten then (n_step, ) dimensional.

    def all_W_projection_statistic(self, repeats, delta_t, total_T):
        t_list = list(range(0, total_T+1, delta_t))
        all_coeff = []

        rng = np.random.default_rng(seed = 42)
        all_start_W = rng.normal(0, 1, size = (repeats, self.output_n, self.input_n))
        for i in range(repeats):
            W0 = all_start_W[i]
            all_Wt = self.weight_response_update(total_T, delta_t, "one_binary", W0 = W0)[0]
            # Calculate the projection coefficients.
            proj_coeff = self.all_W_projection_1D(all_Wt, t_list)
            all_coeff += proj_coeff
        # plot the statistics.
        plt.figure()
        plt.title("Projection Ration of the first 20 eigenvectors")
        plt.xlabel("step")
        plt.ylabel("Ratio")
        plt.xticks(range(len(t_list)))
        # x-axis for lineplot.
        steps = [i for rep in range(repeats) for i in range(len(t_list))]
        sns.lineplot(x=steps, y=all_coeff)
        plt.show()
        return all_coeff

#############################################################################################################
class FfrecTimeDevelop_1D:
    """
    Only consider one input neuron.
    Given at the beginning the inputs and let the network learning over time. Approximate the ffrec over time
    and compare the tendncy with the values of derivative factor.
    When considering one input neuron, the input autocorrelation = 1 and the constants in different samples
    does not make difference for the h_t and ffrec_t values. They are only depend on Wt.
    Therefore only use 1 for the input to get the ffrec values and derivative factor.
    """
    def __init__(self, output_n, R):

        self.input_n = 1
        self.output_n = output_n
        self.R = R

        # Firstly consider the simple case of random symmetric recurrent interaction.
        self.recurrent_interaction = SN.LinearRecurrentNetwork(output_n, self.R).interaction
        self.J_eigval, self.J_eigvec = np.linalg.eigh(self.recurrent_interaction) # Eigval, Eigvec of recurrent inter.

        self.steady_inter = np.linalg.inv(np.eye(self.output_n) - self.recurrent_interaction) # (1-J)^-1

        self.trans_rec_inter = self.recurrent_interaction @ self.steady_inter # J(1-J)^-1
        self.trans_eigval, self.trans_eigvec = np.linalg.eigh(self.trans_rec_inter)

        self.feedforward_interaction = self.feedforward_interaction_asym() # Initial feedforward interaction.

    def feedforward_interaction_asym(self):
        rng = np.random.default_rng(seed = 42)
        W = rng.normal(0, 1, size = (self.output_n, self.input_n)) # (n, 1) dimensional array.
        return W

    def weight_update_euler(self, W_old, delta_t):
        W_new = W_old + delta_t * self.steady_inter @ W_old
        W_new = W_new/np.linalg.norm(W_new)
        return W_new # (n,1) dimensional array.

    def ffrec_t(self, W_t):
        # W_t (n,1) dimensional array. Normalized when calculate ffrec.
        #W_t = W_t/np.linalg.norm(W_t)
        ffrec = W_t.T @ self.recurrent_interaction @ W_t
        return ffrec
    '''
    def derivative_factor(self, W_t):
        # W_t: (n, 1) dimensional array.
        W_t = W_t/np.linalg.norm(W_t)
        wt_eigvec = (W_t.T @ self.trans_eigvec)**2
        factor = wt_eigvec @ self.trans_eigval
        return factor
    '''
    def derivative_factor(self, W_t):
        """
        Fowrward estimation of functional derivative.
        For normalized Wt, factor = W_t.T ( J(1-J)^-1 + (1-J)^-T J ) W_t
        = 2 Wt.T J(1-J)^-1 Wt - 2 Wt.T J Wt Wt.T (1-J)^-1 Wt
        """
        # W_t: (n, 1) dimensioanl array.
        #W_t = W_t/np.linalg.norm(W_t) # Wt normalized.
        #sum_term = self.recurrent_interaction @ self.steady_inter \
        #            + self.steady_inter.T  @ self.recurrent_interaction
        #factor = W_t.T @ sum_term @ W_t

        #factor = 2 * W_t.T @ self.trans_rec_inter @ W_t - 2 * W_t.T @ self.recurrent_interaction \
        #         @ W_t @ W_t.T @ self.steady_inter @ W_t

        factor = W_t.T @ self.trans_rec_inter.T @ W_t + W_t.T @ self.trans_rec_inter @ W_t -\
                 2 * W_t.T @ self.recurrent_interaction @ W_t @ W_t.T @ self.steady_inter @ W_t
        return factor

    def time_update(self, delta_t, T, W0=None):
        if W0 is None:
            W0 = self.feedforward_interaction

        num_step = int(T/delta_t) + 1 # Inclusive the time 0.
        t_list = np.linspace(0, T, num_step)

        # Generate the W_t develop over time.
        W = [W0/np.linalg.norm(W0)]
        for i in range(1, len(t_list)): # exclusive t=0.
            W_old = W[i-1]
            W_new = self.weight_update_euler(W_old, delta_t)
            W.append(W_new)
        W = np.array(W)  # (num_step, n_output, 1) dimensional.

        # Calculate ffrec values and derivative values.
        ffrec = np.zeros(num_step)
        derivative_factor = np.zeros(num_step)
        for i in range(num_step):
            ffrec[i] = self.ffrec_t(W[i])
            derivative_factor[i] = self.derivative_factor(W[i])

        return W, ffrec, derivative_factor

    def plot_time_update_statistics(self, repeats, delta_t, T):
        rng = np.random.default_rng(seed = 42)
        all_start_W = rng.normal(0, 1, size = (repeats, self.output_n, self.input_n))
        all_ffrec = []
        all_factor = []
        for i in range(repeats):
            update = self.time_update(delta_t, T, W0 = all_start_W[i])
            all_ffrec += update[1].tolist()
            all_factor += update[2].tolist()

        x_axis = [i for rep in range(repeats) for i in range(int(T/delta_t) + 1)]
        plt.figure(figsize=(6,5))
        sns.lineplot(x=x_axis, y=all_ffrec, label="Feedforward recurrent \n alignment")
        sns.lineplot(x=x_axis, y=all_factor, label="Derivative")
        plt.xlabel("Time units", fontsize=18)
        plt.xticks([0, 10, 20, 30, 40, 50], fontsize=15)
        plt.yticks([0, 0.4, 0.8, 1.2, 1.6], fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()

    def weight_derivative(self, delta_t, T):
        W = self.time_update(delta_t, T)[0]
        dW_dt = self.steady_inter @ W
        return W, dW_dt

    def W_projection_rec_eigvec_coeff_1D(self, Wt):
        # Sort eigvec in A in descending order corresponding to eigenvalues.
        sort_index = np.argsort(self.J_eigval)[::-1]
        A = self.J_eigvec[:, sort_index] # sorted.
        # coeff = A^-1 * Wt
        return np.linalg.inv(A) @ Wt # (n_output, 1) dimensional.

    def all_W_projection_1D(self, all_Wt, t_list):
        coeff_1D = []
        for i in range(len(t_list)):
            Wt = all_Wt[i]
            coeff_values = np.abs(self.W_projection_rec_eigvec_coeff_1D(Wt)) # Consider absolut values.
            # See the percentage of the coefficients for the first 20 eigenvectors.
            first_ten = np.sum(coeff_values[:20])/np.sum(coeff_values)
            coeff_1D.append(first_ten)
        return coeff_1D # (n_step, ) dimensional.

    def all_W_projection_statistic(self, repeats, delta_t, total_T):
        num_step = int(total_T/delta_t) + 1 # Inclusive the time 0.
        t_list = np.linspace(0, total_T, num_step)
        all_coeff = []

        rng = np.random.default_rng(seed = 42)
        all_start_W = rng.normal(0, 1, size = (repeats, self.output_n, self.input_n))
        for i in range(repeats):
            W0 = all_start_W[i]
            all_Wt = self.time_update(delta_t, total_T, W0 = W0)[0]
            # Calculate the projection coefficients.
            proj_coeff = self.all_W_projection_1D(all_Wt, t_list)
            all_coeff += proj_coeff
        # plot the statistics.
        plt.figure()
        #plt.title("Projection Ration of the first 20 eigenvectors")
        plt.xlabel("Time units", fontsize=18)
        plt.ylabel("Ratio", fontsize=18)
        # x-axis for lineplot.
        steps = [i for rep in range(repeats) for i in range(len(t_list))]
        sns.lineplot(x=steps, y=all_coeff)
        plt.xticks(fontsize=15)
        plt.yticks([0.2, 0.6, 1.0], fontsize=15)
        plt.savefig("F:/Downloads/fig.pdf", bbox_inches='tight')
        plt.show()
        return all_coeff


#############################################################################################################
class AntiHebbRecurrentLearn_1DInput:
    """
    Take over the update rules from the feedforward interaction above and add the anti-Hebb
    learning rule of recurrent interaction.
    Initial recurrent interaction is the random symmetrical matrix.
    """
    def __init__(self, output_n, sample_size, R, rec_network):
        input_n = 1
        self.output_n = output_n
        self.sample_size = sample_size
        self.R = R # Limit the radius of the eigenvalue distribution of recurrent network.
        self.feedforward_net = Hebb_asym_FFinteraction(input_n, self.output_n, self.sample_size, self.R)
        #self.binary_input_samples = self.feedforward_net.one_neuron_binary_input_samples(active_percent=0.5)
        self.initial_rec = rec_network.interaction

    ''' ## Not sure if correct 
    def recurrent_euler_update(self, h_old, J_old, delta_t):
        # h_old is (n, 1) dimensional array, J is (n, n) dimensional matrix.
        steady_inter_old = np.linalg.inv(np.eye(self.output_n) - J_old) # (1-J_old)^-1
        J_new = J_old - delta_t * h_old @ h_old.T @ steady_inter_old.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new
    '''
    def recurrent_euler_update(self, h_old, J_old, delta_t):
        '''
        Update the recurrent interaction network with
        dJ/dt = - vh^T
        Euler method leads to:
        J_new = J_old - delta_t * (1-J_old)^-1 * h_old * h_old^T
        '''
        # h_old is (n, 1) dimensional array, J is (n, n) dimensional matrix.
        steady_inter_old = np.linalg.inv(np.eye(self.output_n) - J_old) # (1-J_old)^-1
        J_new = J_old - delta_t * steady_inter_old @ h_old @ h_old.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new

    def steady_state_response(self, J, h):
        steady_inter = np.linalg.inv(np.eye(self.output_n) - J) # (1-J)^-1
        steady_response = steady_inter @ h
        return steady_response # (n_output, 1) dimensional.

    def recurrent_euler_update_alter(self, h_old, J_old, delta_t):
        '''
        Update the recurrent interaction network with
        dJ/dt = -vv^T
        Euler method leads to:
        J_new = J_old - delta_t * v_old * v_old^T
        with v the steady state response.

        h_old: (n, 1) dimensional array
        J_old, J_new: (n, n) dimensional array
        '''
        v_steady = self.steady_state_response(J_old, h_old)
        J_new = J_old - delta_t * v_steady @ v_steady.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new

    def h_input(self, W, u):
        # W: (n_output, 1) dimensional. u: (1, sample_size) dimensional.
        h = W @ u
        # Normalization of each sample (along column).
        sample_norm = np.linalg.norm(h, axis=0)
        h = h/sample_norm
        return h #(n_output, sample_size) dimensional.

    def ffrec(self, h, J): # TODO: in the case of Anti-Hebbian could be -h^TJh a better measure?!
        # h (n_output, sample_size) dimensional, column-wise normalized.
        # J (n_output, n_output) dimensional recurrent interaction.
        ffrec = []
        for j in range(self.sample_size):
            ffrec.append(h[:,j] @ J @ h[:,j])
        return np.array(ffrec) # (sample_size, ) dimensional.

    def ff_rec_net_update(self, total_T, delta_t):
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t))
        # Initial conditions.
        W0 = self.feedforward_net.feedforward_interaction  # Random gaussian distributed. (n_output, 1) dimensional.
        J0 = self.initial_rec  # (n_output,n_output)dimensional.
        binary_input = self.feedforward_net.one_neuron_binary_input_samples(active_percent=0.5)
        u0 = binary_input[0] # (1, sample_size) dimensional.
        Q0 = binary_input[1] # = 1.
        h0 = self.h_input(W0, u0) # (n_output, sample_size) dimensional.
        r0 = self.steady_state_response(J0, h0)
        ffrec0 = self.ffrec(h0, J0)
        # Update Storing.
        W = [W0] # feedforward net weights.
        J = [J0] # recurrent net weights.
        u = [u0] # input samples.
        h = [h0] # feedforward outputs = recurrent inputs.
        r = [r0] # recurrent outputs.
        Q = [Q0] # input autocorrelations.
        ffrec = [ffrec0] # ffrec alignments.
        # Update parameters.
        for i in range(1, len(t_list)):
            # TODO: here could change the update rule. If necessary build in if-else loop.
            ## The results distinguish not much with different update rules.
            J_t = self.recurrent_euler_update(h[i-1], J[i-1], delta_t) # (n_output, n_output) dimensional.
            #J_t = self.recurrent_euler_update_alter(h[i-1], J[i-1], delta_t)
            J.append(J_t) # (t_step, n_output, n_output) dimensional.

            W_t = self.feedforward_net.euler_weight_update(delta_t, W[i-1], Q[i-1]) # (n_output, 1) dimensional.
            W.append(W_t) # (step, n_output, 1) dimensional.

            binary_input_t = self.feedforward_net.one_neuron_binary_input_samples(active_percent=0.5)

            u_t = binary_input_t[0] # (1, sample_size) dimensional.
            u.append(u_t) # (t_step, 1, sample_size) dimensional.

            Q_t = binary_input_t[1] # Q_t = 1
            Q.append(Q_t) # (t_step, ) dimenaional.

            h_t = self.h_input(W_t, u_t) # (n_output, sample_size) dimensional.
            h.append(h_t) # (t_step, n_output, sample_size) dimensional.

            r_t = self.steady_state_response(J_t, h_t) # (n_output, sample_size) dimensional.
            r.append(r_t) # (t_step, n_output, sample_size) dimensional.

            ffrec_t = self.ffrec(h_t, J_t) # (sample_size, ) dimensional.
            ffrec.append(ffrec_t) # (t_step, sample_size) dimensional

        return np.array(J), np.array(W), np.array(u), np.array(Q), np.array(h), np.array(r), np.array(ffrec)

    def mean_ffrec(self, ffrec_matrix, t_list):
        # ffrec_matrix (t_step, sample_size) dimensional. Get the row-wise mean.
        ffrec_means = np.mean(ffrec_matrix, axis=1)
        # Plot the mean ffrecs.
        plt.figure()
        plt.plot(ffrec_means)
        plt.xticks(range(0, len(t_list)))
        plt.xlabel("t")
        plt.ylabel("mean ffrec")
        plt.show()
        return ffrec_means

    def multiple_cor_heatmaps(self, matrix_list, cmap, total_T, delta_t):
        "Only for the case of considering correlation between neurons, e.g. in the case of responses."
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # include the last time point total_T.
        # Set up figure and image grid
        fig = plt.figure()
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                         nrows_ncols=(1,len(matrix_list)),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        # Add data to image grid
        for i in range(len(matrix_list)):
            ax = grid[i]
            corr = np.corrcoef(matrix_list[i])  # row-wise correlation.
            im = ax.imshow(corr, cmap=cmap, interpolation="nearest")
            ax.set_title("t = "+str(t_list[i]))
        # Colorbar
        ax.cax.colorbar(im)
        ax.cax.toggle_label(True)
        plt.show()

    def multiple_vec_heatmaps(self, vector_list, cmap, total_T, delta_t):
        # delta_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t)) # Include the last time point total_T.
        # Set up figure and image grid
        fig, ax = plt.subplots(ncols = len(t_list), figsize = (15, 0.5))
        # len(vector_list) = len(t_list)
        for i in range(len(t_list)):
            vec = vector_list[i]
            # Normalize the vectors through the maximal element so that the comparison between
            # neurons at each time step is clear.
            vec = vec/np.max(vec)
            sns.heatmap(vec, ax = ax[i], cmap = cmap)#, cbar=False, xticklabels=False
                        #, yticklabels=False)
            ax[i].set_title("t = "+str(t_list[i]))
        plt.show()


#############################################################################################################
class HebbRecurrentLearn_1DInput:
    """
    The only difference this class has compare to the class AntiHebbSymRecurrentLearn_1DInput is the
    update of recurrent network -- it is Hebbian learning instead of Anti-hebbian learning.
    Therefore, most functions could be taken from the AntiHebbSymRecurrentLearn_1DInput.
    """
    def __init__(self, output_n, sample_size, R, rec_network):
        input_n = 1
        self.output_n = output_n
        self.sample_size = sample_size
        self.R = R # Limit the radius of the eigenvalue distribution of recurrent network.
        self.initial_rec = rec_network.interaction
        self.feedforward_net = Hebb_asym_FFinteraction(input_n, self.output_n, self.sample_size, self.R)
        self.antihebb_class = AntiHebbRecurrentLearn_1DInput(self.output_n, self.sample_size, self.R, rec_network)

    ''' # Not sure if it is correct.
    def recurrent_euler_update(self, h_old, J_old, delta_t):
        # h_old is (n, 1) dimensional array, J is (n, n) dimensional matrix.
        steady_inter_old = np.linalg.inv(np.eye(self.output_n) - J_old) # (1-J_old)^-1
        J_new = J_old + delta_t * h_old @ h_old.T @ steady_inter_old.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new
    '''

    def recurrent_euler_update(self, h_old, J_old, delta_t):
        '''
        Update recurrent interaction with
        dJ/dt = vh^T
        Euler method leads to:
        J_new = J_old + delta_t * (1-J_old)^-1 * h_old * h_old^T
        :param h_old: (n, 1) dimensional array
        :param J: (n, n) dimensional matrix
        '''
        # h_old is (n, 1) dimensional array, J is (n, n) dimensional matrix.
        steady_inter_old = np.linalg.inv(np.eye(self.output_n) - J_old) # (1-J_old)^-1
        J_new = J_old + delta_t * steady_inter_old @ h_old @ h_old.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new

    def steady_state_response(self, J, h):
        steady_inter = np.linalg.inv(np.eye(self.output_n) - J) # (1-J)^-1
        steady_response = steady_inter @ h
        return steady_response # (n_output, 1) dimensional.

    def recurrent_euler_update_alter(self, h_old, J_old, delta_t):
        '''
        Update the recurrent interaction network with
        dJ/dt = vv^T
        Euler method leads to:
        J_new = J_old + delta_t * v_old * v_old^T
        with v the steady state response.

        h_old: (n, 1) dimensional array
        J_old, J_new: (n, n) dimensional array
        '''
        v_steady = self.steady_state_response(J_old, h_old)
        J_new = J_old + delta_t * v_steady @ v_steady.T
        # Normalization of J_new (generally not symmetrical) through maximal magnitude.
        J_new_eigval = np.linalg.eigvals(J_new)
        max_mag = max(map(abs, J_new_eigval))
        J_new = J_new * self.R/max_mag
        return J_new

    def h_input(self, W, u):
        return self.antihebb_class.h_input(W, u)

    def ffrec(self, h, J):
        # h (n_output, sample_size) dimensional, column-wise normalized.
        # J (n_output, n_output) dimensional recurrent interaction.
        ffrec = []
        for j in range(self.sample_size):
            ffrec.append(h[:,j] @ J @ h[:,j])
        return np.array(ffrec) # (sample_size, ) dimensional.

    def ff_rec_net_update(self, total_T, delta_t):
        # delte_t = sample_size.
        t_list = list(range(0, total_T+1, delta_t))
        # Initial conditions.
        W0 = self.feedforward_net.feedforward_interaction  # Random gaussian distributed. (n_output, 1) dimensional.
        J0 = self.initial_rec  # Random gaussian symmetrical. (n_output,n_output)dimensional.
        binary_input = self.feedforward_net.one_neuron_binary_input_samples(active_percent=0.5)
        u0 = binary_input[0] # (1, sample_size) dimensional.
        Q0 = binary_input[1] # = 1.
        h0 = self.h_input(W0, u0) # (n_output, sample_size) dimensional.
        r0 = self.steady_state_response(J0, h0)
        ffrec0 = self.ffrec(h0, J0)
        # Update Storing.
        W = [W0] # feedforward net weights.
        J = [J0] # recurrent net weights.
        u = [u0] # input samples.
        h = [h0] # feedforward outputs = recurrent inputs.
        r = [r0] # recurrent outputs.
        Q = [Q0] # input autocorrelations.
        ffrec = [ffrec0] # ffrec alignments.
        # Update parameters.
        for i in range(1, len(t_list)):
            # TODO: here could change the update rule. If necessary build in if-else loop.
            ## The results distinguish not much with different update rules.
            J_t = self.recurrent_euler_update(h[i-1], J[i-1], delta_t) # (n_output, n_output) dimensional.
            #J_t = self.recurrent_euler_update_alter(h[i-1], J[i-1], delta_t)
            J.append(J_t) # (t_step, n_output, n_output) dimensional.

            W_t = self.feedforward_net.euler_weight_update(delta_t, W[i-1], Q[i-1]) # (n_output, 1) dimensional.
            W.append(W_t) # (step, n_output, 1) dimensional.

            binary_input_t = self.feedforward_net.one_neuron_binary_input_samples(active_percent=0.5)

            u_t = binary_input_t[0] # (1, sample_size) dimensional.
            u.append(u_t) # (t_step, 1, sample_size) dimensional.

            Q_t = binary_input_t[1] # Q_t = 1
            Q.append(Q_t) # (t_step, ) dimenaional.

            h_t = self.h_input(W_t, u_t) # (n_output, sample_size) dimensional.
            h.append(h_t) # (t_step, n_output, sample_size) dimensional.

            r_t = self.steady_state_response(J_t, h_t) # (n_output, sample_size) dimensional.
            r.append(r_t) # (t_step, n_output, sample_size) dimensional.

            ffrec_t = self.ffrec(h_t, J_t) # (sample_size, ) dimensional.
            ffrec.append(ffrec_t) # (t_step, sample_size) dimensional

        return np.array(J), np.array(W), np.array(u), np.array(Q), np.array(h), np.array(r), np.array(ffrec)

    def mean_ffrec(self, ffrec_matrix, t_list):
        return self.antihebb_class.mean_ffrec(ffrec_matrix, t_list)

    def multiple_cor_heatmaps(self, matrix_list, cmap, total_T, delta_t):
        return self.antihebb_class.multiple_cor_heatmaps(matrix_list, cmap, total_T, delta_t)

    def multiple_vec_heatmaps(self, vector_list, cmap, total_T, delta_t):
        return self.antihebb_class.multiple_vec_heatmaps(vector_list, cmap, total_T, delta_t)


#############################################################################################################
if __name__ == "__main__":
    delta_t = sample_size = 50
    total_T = 200
    n = 500
    R = 0.85
    t_list = list(range(0, total_T+1, delta_t))
    initial_rec_sym_rand = SN.LinearRecurrentNetwork(n=n, R=R)

    '''
    # Symmetrical weight matrix.
    hebb_sym = Hebb_sym_FFinteraction(n, R)
    update_result = hebb_sym.weight_response_update(total_T, delta_t, sample_size)
    W = update_result[0]
    r = update_result[1]
    hebb_sym.multiple_heatmaps(W, "Blues", total_T, delta_t)
    hebb_sym.multiple_heatmaps(r, "Greens", total_T, delta_t)
    
    
    # Asymmetrical weight matrix with two reversed input neurons.
    hebb_two_neuron = Hebb_asym_FFinteraction(1, 50, sample_size)
    
    # Two neurons with reversed inputs.
    update_result = hebb_two_neuron.weight_response_update(total_T, delta_t, "two_reverse")
    W = update_result[0]
    r = update_result[1]
    h = update_result[2]
    #hebb_two_neuron.multiple_heatmaps(W, "Blues", total_T, delta_t)
    #hebb_two_neuron.multiple_heatmaps(r, "Greens", total_T, delta_t)
    hebb_two_neuron.mean_ffrec(h, total_T, delta_t) # Two neurons with random inputs.
    update_result_rand = hebb_two_neuron.weight_response_update(total_T, delta_t, "random")
    W_rand = update_result_rand[0]
    r_rand = update_result_rand[1]
    h_rand = update_result_rand[2]
    u_rand = update_result_rand[3]
    print(u_rand)
    print(W_rand)
    #hebb_two_neuron.multiple_heatmaps(W_rand, "Blues", total_T, delta_t)
    #hebb_two_neuron.multiple_cor_heatmaps(r_rand, "Greens", total_T, delta_t)
    #hebb_two_neuron.mean_ffrec(h_rand, total_T, delta_t)
    

    # One neuron with binary input. Only Hebbian learning of feedforward network.
    hebb_one_neuron = Hebb_asym_FFinteraction(1, 500, sample_size, R)
    update_result = hebb_one_neuron.weight_response_update(total_T, delta_t, "one_binary")
    W_onebinary = update_result[0] # vectors.
    r_onebinary = update_result[1] # matrices. --> plot the correlation.
    h_onebinary = update_result[2] # vectors. # TODO: Errors possible!
    u_onebinary = update_result[3] # vectors.
    #coeff_onebinary = hebb_one_neuron.all_W_projection_1D(W_onebinary, t_list)
    #statistic_proj_coeff = hebb_one_neuron.all_W_projection_statistic(50, delta_t, total_T)
    #hebb_one_neuron.multiple_vec_heatmaps(u_onebinary, "BuPu", total_T, delta_t)
    #hebb_one_neuron.multiple_vec_heatmaps(W_onebinary.reshape(len(t_list),
    #                                                          1, hebb_one_neuron.output_n),
    #                                     "Greens", total_T, delta_t)
    #hebb_one_neuron.multiple_vec_heatmaps(h_onebinary, "Blues", total_T, delta_t)
    #hebb_one_neuron.multiple_vec_heatmaps(coeff_onebinary.reshape(len(t_list), 1,
    #                                                              hebb_one_neuron.output_n),
    #                                      "Greens", total_T, delta_t)
    #hebb_one_neuron.multiple_cor_heatmaps(r_onebinary, "YlGnBu", total_T, delta_t)
    #hebb_one_neuron.mean_ffrec(h_onebinary, total_T, delta_t)
    #print(hebb_one_neuron.derivative_factor_sym_interact_one_input(t_list, W_onebinary))

    
    W0 = W_onebinary[0]
    W1 = W_onebinary[1]
    W2 = W_onebinary[2]
    #print(np.linalg.norm(W0), np.linalg.norm(W1), np.linalg.norm(W2))
    u0 = u_onebinary[0]
    u1 = u_onebinary[1]
    u2 = u_onebinary[2]
    h0 = hebb_one_neuron.steady_recurrent_response(W0, u0)[1]
    
    print(h0/np.linalg.norm(h0))
    print("____________________________")
    h1 = hebb_one_neuron.steady_recurrent_response(W1, u1)[1]
    print(h1/np.linalg.norm(h1))
    print("____________________________")
    h2 = hebb_one_neuron.steady_recurrent_response(W2, u2)[1]
    print(h2/np.linalg.norm(h2))
    print("____________________________")
    print(np.linalg.norm(h0), np.linalg.norm(h1), np.linalg.norm(h2))
    
    #print(np.shape(hebb_one_neuron.W_projection_rec_eigvec_coeff_1D(W0)))
    print(h_onebinary[0])

    # One neuron with binary input. Hebbian learning of feedforward network and anti Hebbian learning of
    # recurrent network.
    rec_antilearn_1D = AntiHebbRecurrentLearn_1DInput(output_n=n, sample_size=sample_size, R=R,
                                                         rec_network=initial_rec_sym_rand)
    update_anti = rec_antilearn_1D.ff_rec_net_update(total_T, delta_t)
    J_1D_anti = update_anti[0] # (t_step, n_output, n_output) dimensional.
    W_1D_anti = update_anti[1] # (step, n_output, 1) dimensional. While plotting reshape to e.g. ((len(t_list), 1, n)).
    u_1D_anti = update_anti[2] # (t_step, 1, sample_size) dimensional.
    Q_1D_anti = update_anti[3] # (t_step, ) dimenaional.
    h_1D_anti = update_anti[4] # (t_step, n_output, sample_size) dimensional.
    r_1D_anti = update_anti[5] # (t_step, n_output, sample_size) dimensional.
    ffrec_1D_anti = update_anti[6]
    #print(mean_ffrec_anti)
    #rec_antilearn_1D.multiple_vec_heatmaps(J_1D_anti, "YlGnBu", total_T, delta_t)
    #rec_antilearn_1D.multiple_vec_heatmaps(h_1D_anti, "YlGnBu", total_T, delta_t)
    #rec_antilearn_1D.multiple_vec_heatmaps(W_1D_anti.reshape((len(t_list), 1, n)), "YlGnBu", total_T, delta_t)
    #rec_antilearn_1D.multiple_vec_heatmaps(u_1D_anti, "YlGnBu", total_T, delta_t)
    #rec_antilearn_1D.multiple_cor_heatmaps(r_1D_anti, "YlGnBu", total_T, delta_t)
    mean_ffrec_anti = rec_antilearn_1D.mean_ffrec(ffrec_1D_anti, t_list)
    reverse_meam_ffrec_anti = rec_antilearn_1D.mean_ffrec(-ffrec_1D_anti, t_list)


    # One neuron with binary input. Hebbian learning in both feedforward and recurrent networks.
    rec_learn_1D = HebbRecurrentLearn_1DInput(output_n=n, sample_size=sample_size, R=R,
                                              rec_network=initial_rec_sym_rand)
    update_hebb = rec_learn_1D.ff_rec_net_update(total_T, delta_t)
    J_1D_hebb = update_hebb[0] # (t_step, n_output, n_output) dimensional.
    W_1D_hebb = update_hebb[1] # (step, n_output, 1) dimensional. While plotting reshape to e.g. ((len(t_list), 1, n)).
    u_1D_hebb = update_hebb[2] # (t_step, 1, sample_size) dimensional.
    Q_1D_hebb = update_hebb[3] # (t_step, ) dimenaional.
    h_1D_hebb = update_hebb[4] # (t_step, n_output, sample_size) dimensional.
    r_1D_hebb = update_hebb[5] # (t_step, n_output, sample_size) dimensional.
    ffrec_1D_hebb = update_hebb[6] # (t_step, sample_size) dimensional.
    mean_ffrec_hebb = rec_learn_1D.mean_ffrec(ffrec_1D_hebb, t_list)
    #rec_learn_1D.multiple_vec_heatmaps(J_1D_hebb, "YlGnBu", total_T, delta_t)
    #rec_learn_1D.multiple_vec_heatmaps(h_1D_hebb, "YlGnBu", total_T, delta_t)
    #rec_learn_1D.multiple_vec_heatmaps(W_1D_hebb.reshape((len(t_list), 1, n)), "YlGnBu", total_T, delta_t)
    #rec_learn_1D.multiple_vec_heatmaps(u_1D_hebb, "YlGnBu", total_T, delta_t)
    #rec_learn_1D.multiple_cor_heatmaps(r_1D_hebb, "YlGnBu", total_T, delta_t)
    '''

    time_exp_1D = FfrecTimeDevelop_1D(output_n=n, R=R)
    W0 = time_exp_1D.feedforward_interaction
    #print(time_exp_1D.weight_derivative(0.1, 10))
    #time_exp_1D.plot_time_update_statistics(50,0.1, 5)
    time_exp_1D.all_W_projection_statistic(50, delta_t=0.1, total_T=5)

































