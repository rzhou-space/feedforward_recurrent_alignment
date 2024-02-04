import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
from Networks import LinearRecurrentNetwork
from sklearn.decomposition import PCA

#########################################################################################################

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


    def EulerMaruyama(self, sigma_trial, Delta_t_euler, h_det, T, r0):

        sde_a = lambda r, h_det : -r + self.interaction @ h_det # r is the response vector.
        sde_b = lambda sigma_trial : sigma_trial

        #r0 = np.linalg.solve(np.identity(self.neurons) - self.interaction, h_det)  # (1-J)^(-1)*h_det
        #r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_values = [r0]

        num_steps = int(T/Delta_t_euler)
        np.random.seed(42)

        for i in range(num_steps):
            dW = np.random.normal(0, 1, size = self.neurons) * np.sqrt(Delta_t_euler)
            # Euler-Maruyama Scheme.
            r = r_values[i] + sde_a(r_values[i], h_det) * Delta_t_euler + sde_b(sigma_trial) * dW
            r_values.append(r)

        return np.array(r_values)[1:] # steps x n - dimensional.


    # TODO: EulerMaruyama_alter works differently than EulerMaruyama??!!
    '''
    def EulerMaruyama_alter(self, sigma_trial, Delta_t_euler, h_det, T, r0):

        sde_a = lambda r, h_det : -r + self.interaction @ h_det # r is the response vector.
        sde_b = lambda sigma_trial : sigma_trial

        #r0 = np.linalg.solve(np.identity(self.neurons) - self.interaction, h_det) # (1-J)^(-1)*h_det
        #r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_values = [r0]
        r = r0

        num_steps = int(T/Delta_t_euler)
        np.random.seed(42)

        for i in range(num_steps):
            dW = np.random.normal(0, 1, size = self.neurons) * np.sqrt(Delta_t_euler)
            # Euler-Maruyama Scheme.
            r += sde_a(r, h_det) * Delta_t_euler + sde_b(sigma_trial) * dW
            r_values.append(r)

        return np.array(r_values)[1:] # steps x n - dimensional.
    '''



    def activity_rand_align(self, sigma_trial, Delta_t_euler, T): # TODO: the result values are too small. Is the h_det correct?

        # Generate a random vector as h_det.
        np.random.seed(42)
        h_det = np.random.normal(0, 1, size = self.neurons)
        # Normalize h_det.
        h_det = h_det/np.linalg.norm(h_det)

        #r0 = np.zeros(self.neurons)
        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_value = self.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T, r0)
        #r_value = self.EulerMaruyama_alter(sigma_trial, Delta_t_euler, h_det, T)

        # Calculate the mean across neurons at each time step.
        r_value = r_value.mean(axis=1)

        return r_value



    def activity_max_align(self, sigma_trial, Delta_t_euler, T):

        # Set the h_det as the maximal eigenvector (already normalized).
        h_det = self.eigvec[:, np.argmax(self.eigval)]

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_value = self.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T, r0)
        #r_value = self.EulerMaruyama_alter(sigma_trial, Delta_t_euler, h_det, T)

        # Calculate the mean across neurons at each time step.
        r_value = r_value.mean(axis=1)

        return r_value



    def activity_plot(self, r_value_rand, r_value_max, T, Delta_t_euler):

        time = np.arange(0, T, Delta_t_euler)
        plt.figure()
        plt.xlabel("Time(ms)")
        plt.ylabel("normalized mean activity")
        plt.plot(time[10:], r_value_rand[10:], c = "green", alpha = 0.5, label = "random alignment")
        plt.plot(time[10:], r_value_max[10:], c = "blue", alpha = 0.5, label = "maximal alignment")
        plt.legend()
        plt.show()



    def intra_trial_stability(self, sigma_trial, Delta_t_euler, Delta_t_intra, T, h_det):
        """
        Calculate the intra trial stability.
        """
        # The steps that goes in euler approximated solution.
        step_width = int(Delta_t_intra/Delta_t_euler)
        # The total steps that iterate in euler approximated solution.
        steps = int((T-Delta_t_intra)/Delta_t_euler)
        # Generate responses through Euler Maruyama.
        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_values = self.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T, r0)
        # z-scored response vectors (z-scored rows).
        z_r = sp.stats.zscore(r_values, axis=1, ddof=1)

        # Store the intra correlation values.
        intra_c = np.zeros(steps)

        for t in range(int(steps)):
            # z-scored response.
            r_t = z_r[t]
            r_tdt = z_r[t+step_width]
            intra_c[t] = r_t @ r_tdt

        return intra_c, np.average(intra_c)



    def stability_sort_align(self, sigma_trial, Delta_t_euler, Delta_t_intra, T):
        """
        Applying eigenvectors sorted in ascending order of eigenvalues to calculate the average
        intra stability values across time for each eigenvector.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)

        average_intra_c = []
        ffrec_align = []

        for i in sort_index:
            h_det = self.eigvec[:, i]
            # Calculate the average intra trial stability for h_det.
            intra_c = self.intra_trial_stability(sigma_trial, Delta_t_euler, Delta_t_intra, T, h_det)
            # The second element of the result is the average value.
            avg_c = intra_c[1]
            # Store the average value.
            average_intra_c.append(avg_c)
            # Calculate and store the feedforward alignment score.
            ffrec_align.append(h_det @ self.interaction @ h_det)

        return np.array(average_intra_c), np.array(ffrec_align)



    def sort_intra_trial_plot(self, average_intra_c, ffrec_align):
        """
        Scatter plot of the averaged intra trial stability against the feedforward alignment score.
        """
        plt.figure()
        plt.scatter(ffrec_align, average_intra_c)
        plt.title("intra trial stability against feedforward alignment")
        plt.axvline(x=0, color = "black", label = "random")
        plt.axvline(x=self.R, color = "red", label = "R="+str(self.R))
        plt.xlabel("feedforward alignment")
        plt.ylabel("intra trial correlations")
        plt.legend()
        plt.show()



########################################################################################################################

class Rewrite_IntraTrial:

    def __init__(self, n, R):

        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)


    def euler_maruyama(self, r0, dt_euler, h_det, T, sigma_time):
        steps = int(T/dt_euler)
        r_vec = [r0] # r0 a (num_neurons,) dimensional array.

        for i in range(steps):
            dW = np.random.normal(0, 1, size = self.neurons)
            r_old = r_vec[i]
            # Euler Maruyama scheme.
            r_new = r_old + (-r_old + self.interaction @ h_det) * dt_euler +\
                sigma_time * np.sqrt(dt_euler) * dW

            r_vec.append(r_new)

        return np.array(r_vec)[1:]



    def rand_align(self, dt_euler, T, sigma_time):
        h_det = np.random.normal(0, 1, size = self.neurons)
        # Normalization.
        h_det = h_det/np.linalg.norm(h_det)

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)



    def max_align(self, dt_euler, T, sigma_time):
        # The eigenvector corresponding with the largest eigenvalue.
        h_det = self.eigvec[:, np.argmax(self.eigval)]

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)



    def min_align(self, dt_euler, T, sigma_time):
        # The eigenvector corresponding with the largest eigenvalue.
        h_det = self.eigvec[:, np.argmin(self.eigval)]

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time)
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



    def plot_min_max_align(self, r_max, r_min):
        '''
        r_rand and r_max are (num_neuron,) dimensional array, each for response of the same neuron
        under the random and maximal alignment.
        '''
        # Normalization of neuron response so that they have the same mean value after normalization.
        r_max_normal = r_max/np.mean(r_max)
        r_min_normal = r_min/np.mean(r_min)  # If the given response vector is under maximal alignment.

        plt.figure()
        plt.title("Intra Trial Neuron Activity")
        plt.xlabel("time (ms)")
        plt.ylabel("Normalized activity")
        plt.plot(np.linspace(0,120,1200), r_max_normal, c = "green", label = "max align", alpha = 0.7)
        plt.plot(np.linspace(0,120,1200), r_min_normal, c = "blue", label = "min align", alpha = 0.7)
        plt.legend()
        plt.show()




    def intra_trial_stab(self, dt, r_vec, t):
        r = sp.stats.zscore(r_vec[t])
        r_delay = sp.stats.zscore(r_vec[t + dt])
        return r @ r_delay


    def stab_hdet(self, h_det, dt, t_limit, dt_euler, T, sigma_time):
        # Calculate the response vector.
        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_vec = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time)

        # Calculate the intra trial stability at each time point.
        stab = []
        for t in range(t_limit):
            stab.append(self.intra_trial_stab(dt, r_vec, t))

        # Return the mean value across time as the stability for the certain h_det.
        return np.mean(stab)


    def sort_stab(self, dt_euler, dt_intra, T, sigma_time):

        # Sort eigenvectors in the order of ascending eigenvalues.
        sorted_indices = np.argsort(self.eigval)
        #sorted_eigval = self.eigval[sorted_indices]
        sorted_eigvec = self.eigvec[:, sorted_indices]

        # The dt and time-limit are the same for all stability calculations with different h_det.
        dt = int(dt_intra/dt_euler)
        t_limit = int((T-dt_intra)/dt_euler)

        ffrec_align = []
        mean_stab = []
        # Range over sorted eigenvectors to calculate the mean stability for each of them.
        for i in range(len(self.eigval)):
            h_det = sorted_eigvec[:, i]
            mean_stab.append(self.stab_hdet(h_det, dt, t_limit, dt_euler, T, sigma_time))
            ffrec_align.append(h_det @ self.interaction @ h_det)

        return ffrec_align, mean_stab



    def plot_stab(self, ffrec_align, mean_stab):
        plt.figure()
        plt.title("Intra Trial Stability")
        plt.xlabel("feedforward recurrent alignment")
        plt.ylabel("Intra Trial Stability")
        plt.scatter(ffrec_align, mean_stab)
        plt.show()






########################################################################################################################
class eigval_circle:

    def __init__(self, n, R):
        self.n = n
        self.R = R

    def rand_matrix_1(self):
        J = np.random.normal(0, self.R/np.sqrt(self.n), size = (self.n, self.n))
        # Turn to a real symmetric matrix.
        J = (J + J.T)/2
        # calculate the eigenvalues.
        eigval = np.linalg.eigvalsh(J)

        return eigval


    def rand_matrix_2(self):
        J = np.random.normal(0, 1, size = (self.n, self.n))
        # Turn to a real symmetric matrix.
        J = (J + J.T)/2
        # Normalize through its maximal eigenvalue.
        eigval = np.linalg.eigvalsh(J)
        max_eigval = np.max(eigval)
        J = self.R*J/max_eigval
        # Calculate the eigenvalues of the new matrix.
        eigval = np.linalg.eigvalsh(J)

        return eigval


    def eigval_plot(self, eigval_1, eigval_2):
        plt.figure()
        # eigval_1 generated from the matrix with std = R/sqrt(n).
        plt.scatter(eigval_1, np.zeros(self.n), c = "blue", alpha = 0.5, label = "std = R/$\sqrt{n}$")
        # eigval_2 generated from the matrix with std = 1.
        plt.scatter(eigval_2, np.zeros(self.n), c = "green", alpha = 0.5, label = "std = 1")
        plt.legend()
        plt.show()


    def diff_var_eigval(self, eigval_1, eigval_2):
        return np.var(eigval_1) - np.var(eigval_2)





########################################################################################################################
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
        self.rng = np.random.default_rng(seed = 42)


    '''
    def evok_activity(self, kappa, beta, L, basis_vectors, num_sample): # TODO: should be the only function containing error.
        """
        :param basis_vectors: An array containing basis vectors. nxn dimensional array.
        :param num_sample: the number of response samples.
        """
        # Determine the upper limit factor M.
        M = kappa * beta

        # Calculate the input variance Sigma^Dim.
        # TODO: The calculation of Sigma^Dim ist not correct.
        sigma_dim = 0
        for i in range(L-1, L-1+M):
            v_i = np.exp(-2*(i-L)/beta)
            # Getting vectors in columns.
            sigma_dim += v_i * (basis_vectors[:, i] @ basis_vectors[:, i])

        # Calculate the response variance Sigma^Act.
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = sigma_dim * (new_interact @ new_interact.T) # (1-J)^(-1)*sigma_dim*(1-J)^(-T) with sigma_dim real.
        # Samples from multivariate Gaussian distribution generate the response vectors.
        np.random.seed(42)
        act_vec = np.random.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        #return act_vec # num_sample x n_neuron dimensional #TODO: Need change output after correction.
        return sigma_dim
    '''


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
        act_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        # rng = np.random.default_rng(seed = 10)
        # act_vec = rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
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
        random_matrix = self.rng.normal(0, 1, size = (self.neurons, self.neurons))
        # rng = np.random.default_rng(seed = 42)
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
        var_aligned = DimObj.align_eigvec(kappa, beta, 1, num_sample)[:20]
        var_random = DimObj.align_random(kappa, beta, 1, num_sample)[:20]
        plt.figure()
        plt.title("Variance Ratio of Aligned and Random Inputs")
        plt.xlabel("PC Index")
        plt.ylabel("Variance ratio")
        plt.plot([i for i in range(20)], var_aligned, c="blue", label = "Aligned")
        plt.plot([i for i in range(20)], var_random, c="green", label = "Random")
        plt.legend()
        plt.show()


    def dim_to_ffrec(self, kappa, beta):

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


#########################################################################################################

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
        self.rng = np.random.default_rng(seed=42)


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

        #rng = np.random.default_rng(seed = 42)
        input_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_spont, size = num_sample)
        return input_vec, sigma_spont  # input_vec is a num_sample x n_neuron dimensional array.
        # sigma_spont will be used to generate spont. act in the method "spont_act".


    def spont_act(self, sigma_spont, num_sample):
        new_interact = np.linalg.inv(np.identity(self.neurons)-self.interaction) # (1-J)^(-1)
        sigma_act = new_interact @ sigma_spont @ new_interact.T # (1-J)^(-1)*sigma_spont*(1-J)^(-T)
        #rng = np.random.default_rng(seed = 42)
        act_vec = self.rng.multivariate_normal(np.full(self.neurons, 0), sigma_act, size = num_sample)
        return act_vec # num_sample x n_neuron dimensional array.


    def variance_spont_input_act(self, input_vec, act_vec):
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

    '''
    def evok_explained_by_spont_act(self, spont_act, sigma_evok):
        """
        Calculate the variance of evoked pattern A explained by the PCs of spontaneous activity.
        :param pc_set_B: The PCs of pattern B. Rows are the principal components.
        :param sigma_A: The covariance matrix of act. pattern A.
        """

        # Get the principal components of the spontaneous activity through PCA.
        pca = PCA(n_components=self.neurons)
        pca.fit_transform(spont_act)
        # Extract principal components (in rows).
        spont_pc = pca.components_
        # Calculate the variance of A explained by PC of spont.activity.
        var_align = np.zeros(self.neurons)
        for i in range(self.neurons):
            pc_i = spont_pc[i]
            var_align[i] = pc_i @ sigma_evok @ pc_i / np.trace(sigma_evok)

        return var_align
    '''

    def var_explain_A_by_B(self, act_patternA, act_patternB):
        """
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
        sorted_indices = np.argsort(self.eigval)[::-1]
        sorted_eigenvectors = self.eigvec[:, sorted_indices]
        align_act = Dimensionality.evok_activity(self, kappa, beta_dim, L, sorted_eigenvectors, num_sample)
        align_expl_var = self.var_explain_A_by_B(align_act, spont_act)

        # Access the explained variance ratio of random input evoked activity by spont. act.
        #rng = np.random.default_rng(seed = 42)
        random_matrix = self.rng.normal(0, 1, size = (self.neurons, self.neurons))
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
            spont_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[0]
            align_expl_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[1]
            rand_expl_var = self.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, L, num_sample)[2]
            # Consider the first 20 PCs for illustration.
            all_spont += spont_var[:20].tolist()
            all_align += align_expl_var[:20].tolist()
            all_rand += rand_expl_var[:20].tolist()

        # X-asis for the line plot.
        pc_index = [i for rep in range(num_stimuli) for i in range(1,21)] # A list with 1,...,20 repeat num_stimuli times.

        # Lineplot with confidence interval.
        plt.figure()
        plt.title("Alignment of evoked to spontaneous activity")
        plt.xlabel("spont. PC")
        plt.ylabel("variance ratio")
        sns.lineplot(x=pc_index, y=all_spont, color="red", label="spont act")
        sns.lineplot(x=pc_index, y=all_align, color="blue", label="aligned act")
        sns.lineplot(x=pc_index, y=all_rand, color="green", label="rand act")
        plt.show()



    def align_A_to_B(self, act_patternA, act_patternB, num_sample): # TODO: Fehler? Wie ist es in Sigrid code definiert/berechnet?
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
            r_iA = act_patternA[i, :] # The i-th row the i-th activity pattern.
            align_scores[i] = r_iA @ cov_B @ r_iA / ((np.linalg.norm(r_iA)**2)*np.trace(cov_B))

        return np.mean(align_scores)


    def align_to_ffrec(self, kappa, beta_spont, beta_dim, num_sample): #TODO: fig7 and fig10 neu ploten nach random generator setting.
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
            # Access the evoked activity under the given L_current.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, self.eigvec, num_sample) # TODO: könnte hier auch wegen random generator Fehler geben? insb. by dim vs. ffrec.
            # Calculate the alignment score for this current activity.
            align[L_current] = self.align_A_to_B(current_act, spont_act, num_sample)
            #align[L_current] = self.align_A_to_B_alter(current_act, spont_act)
            ffrec[L_current] = sorted_eigvec[:, L_current] @ self.interaction @ sorted_eigvec[:, L_current]
        '''
        # Plot the alignment with spontaneous activity against ff-rec.alignment.
        # The larger the L, the smaller the ff-rec. aligment. Therefore apply the flipped "align".
        plt.figure()
        plt.title("Alignment Spont.Activity against FF-Rec. Alignment")
        plt.xlabel("ff-rec. Alignment")
        plt.ylabel("Alignment to spont. activity")
        plt.scatter(np.flip(ffrec), np.flip(align))
        plt.show()
        '''
        return align



    def align_A_to_B_alter(self, act_patternA, act_patternB): #TODO: Seems to be not correct!!!
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
        #final_score = np.mean(align_scores)/np.trace(cov_B)
        final_score = np.mean(align_scores)

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

        return ffrec_align


    def align_to_ffrec_alter(self, kappa, beta_spont, num_sample, beta_dim):
        # Access the spontaneous activity. L = 1 already inserted.
        sigma_spont = self.spont_input(kappa, beta_spont, 1, num_sample)[1]
        spont_act = self.spont_act(sigma_spont, num_sample)

        # The range of L (the feedforward-alignment).
        # L = np.array([i for i in range(int(self.neurons/2))]) # L starts with 0 here!
        L = np.array([i for i in range(50)])

        # Storing the pattern to pattern alignment scores.
        pattern_align = np.zeros(len(L))

        for L_current in L: # L starts with 0.
            # Access the evoked activity under the given L_current.
            current_act = Dimensionality.evok_activity(self, kappa, beta_dim, L_current+1, self.eigvec, num_sample)
            pattern_align[L_current] = self.align_A_to_B_alter(current_act, spont_act)

        return pattern_align


########################################################################################################################

class IntraTrialStabCorrection:

    def __init__(self, n, R):

        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eig(self.interaction)
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



########################################################################################################################

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
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)


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
        rng = np.random.default_rng(seed = 42)
        rand_h = rng.normal(0, 1, size = self.neurons)
        h_det = rand_h/np.linalg.norm(rand_h)
        for i in range(repeats):
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



    def ttc_sort_align(self, sigma_trial, N_trial):
        """
        Caculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        :return: An array containing a trial to trial correlation for each h_det.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        result = []
        result_ffrec_align = []
        for i in sort_index:
            h_det = self.eigvec[:, i]
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            result_ffrec_align.append(h_det @ self.interaction @ h_det)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))

        # Plot the ttc against ffrec-align in a scatterplot.
        plt.figure()
        plt.scatter(np.array(result_ffrec_align), np.array(result))
        plt.axvline(x = 0, color = "black")
        plt.axvline(x = self.R, color = "red", label = "align = R = "+str(self.R))
        plt.xlabel("feedforward alignment")
        plt.ylabel("trial to trial correlation")
        plt.legend()
        plt.show()
        return np.array(result), np.array(result_ffrec_align)



########################################################################################################################

class TrialToTrialCor_NewTry:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param mode: There are three modes: max_align, random_align, and sort_align.
        """
        self.neurons = n
        self.R = R
        self.network = LinearRecurrentNetwork(self.neurons, self.R)
        self.interaction = self.network.interaction
        self.eigval, self.eigvec = np.linalg.eigh(self.interaction)


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

    '''
    def ttc_random_align(self, sigma_trial, N_trial, repeats):
        """
        Calculation of the mean trial to trial correlation under random alignment of the interaction
        matrix.
        :return: An array containing the number of repeats of trial to trial correlation.
        """
        result = []
        index = np.random.randint(0, len(self.eigval), size = repeats)
        for i in index:
            # Choose a random eigenvector as h_det.
            h_det = self.eigvec[:, i]
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)
    '''



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
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result)



          # plt.imshow(all_response)
            # plt.colorbar()
            # plt.show()
    def ttc_sort_align(self, sigma_trial, N_trial):
        """
        Caculation of the trial to trial correlation with deterministic input h_det equal to
        eigenvectors with ascending eigenvalues.
        :return: An array containing a trial to trial correlation for each h_det.
        """
        # Sort the eigenvalues in ascending order and get the index.
        sort_index = np.argsort(self.eigval)
        result = []
        result_ffrec_align=[]
        for i in sort_index:
            h_det = self.eigvec[:, i]
            all_response = self.network.trial_response(h_det, sigma_trial, N_trial)
            #print(all_response.shape)


            #print(np.corrcoef(all_response).shape, np.mean(np.corrcoef(all_response)))
            result_ffrec_align.append(h_det @ self.interaction @ h_det)
            result.append(self.trial_to_trial_correlation(all_response, N_trial))
        return np.array(result), np.asarray(result_ffrec_align)



    def hist_plot(self, ttc_max, ttc_rand):
        plt.figure()
        bins = np.linspace(-1, 1, 100)
        plt.title("Trial to Trial Correlations")
        plt.xlabel("Correlation Values")
        plt.ylabel("Frequency")
        plt.hist(ttc_max, bins, alpha = 0.5, label = "maximal alignment")
        plt.hist(ttc_rand, bins, alpha = 0.5, label = "random alignment")
        plt.legend(loc = "upper right")
        plt.show()




########################################################################################################################
########################################################################################################################
if __name__ == "__main__":
    '''
    # Eigenvalues Semicircle tests.
    obj = eigval_circle(500, 0.5)
    eigval_1 = obj.rand_matrix_1()
    eigval_2 = obj.rand_matrix_2()
    obj.eigval_plot(eigval_1, eigval_2)
    print(obj.diff_var_eigval(eigval_1, eigval_2))
    '''
    ####################################################################################################################
    # RNN Models

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


    '''
    # Intra trial stability.

    obj_stab = IntraTrialStab(n_neuron, R)

    # Test case of activities under random alignment and maximal alignment.
    
    r_value_rand = obj_stab.activity_rand_align(sigma_time, Delta_t_euler, T)
    r_value_max = obj_stab.activity_max_align(sigma_time, Delta_t_euler, T)
    #print(r_value_rand)
    #print(np.shape(r_value_rand))
    #print(r_value_max)
    #print(np.shape(r_value_max))
    obj_stab.activity_plot(r_value_rand, r_value_max, T, Delta_t_euler)

    

    # Test calculation of intra trial stability.
    
    h_det = np.random.normal(0, 1, size = n_neuron)
    h_det = h_det/np.linalg.norm(h_det)
    intra_c = obj_stab.intra_trial_stability(sigma_trial, Delta_t_euler, Delta_t_intra, T, h_det)
    print(intra_c[0])
    print(intra_c[0].shape)
    plt.figure()
    plt.scatter(np.linspace(0,1000, 1000), intra_c[0])
    plt.show()
    print(intra_c[1])
    
    # Test calculation of intra trial stability with sorted alignment.

    sorted = obj_stab.stability_sort_align(sigma_time, Delta_t_euler, Delta_t_intra, T)
    avg_intra_c = sorted[0]
    ffrec_align = sorted[1]
    obj_stab.sort_intra_trial_plot(avg_intra_c, ffrec_align)
    
    
    # Test of Rewrite Intra Trial stability
    Obj = Rewrite_IntraTrial(n_neuron, R)

    r_rand = Obj.rand_align(dt_euler, T, sigma_time)
    r_rand_1 = r_rand[:, 10]

    r_max = Obj.max_align(dt_euler, T, sigma_time)
    r_max_1 = r_max[:, 10]

    r_min = Obj.min_align(dt_euler, T, sigma_time)
    r_min_1 = r_min[:, 10]

    Obj.plot_max_rand_align(r_rand_1, r_max_1)
    Obj.plot_min_max_align(r_max_1,r_min_1)


    
    ###############################################################################
    # Comarison between EulerMaruyama & EulerMaruyama_alter.
    
    np.random.seed(42)
    h = np.random.normal(0, 1, size = n_neuron)
    r_rand_1 = obj_stab.EulerMaruyama(sigma_trial, Delta_t_euler, h, T)
    r_rand_2 = obj_stab.EulerMaruyama_alter(sigma_trial, Delta_t_euler, h, T)
    #print(r_rand_1 - r_rand_2)
    plt.figure()
    plt.plot(r_rand_1 - r_rand_2)
    plt.show()
    '''
    
    ################################################################################
    # Test of Dimentionality.
    
    DimObj = Dimensionality(n_neuron, R)
    #DimObj.plot_align(kappa, beta_dim, num_sample)
    DimObj.plot_dim_to_ffrec(kappa, beta_dim, num_sample)


    ################################################################################
    '''
    # Test of alignment with spontaneous activity.
    SpontObj = AlignmentSpontaneousAct(n_neuron, R)
    
    spont_input = SpontObj.spont_input(kappa, beta_spont, 1, num_sample)
    spont_act = SpontObj.spont_act(spont_input[1], num_sample)
    SpontObj.variance_spont_input_act(spont_input[0], spont_act)
    

    #SpontObj.multiple_stimuli_compare_align_rand_spont(10, kappa, beta_spont, beta_dim, 1, num_sample)
    

    #SpontObj.compare_var_explain_align_rand_spont(kappa, beta_spont, beta_dim, 1, num_sample)
    #print(SpontObj.align_A_to_B(spont_act, spont_act, num_sample))
    #TODO: still not work... should not be from random generator (although it looks like so) :/
    #print(SpontObj.align_to_ffrec(kappa, beta_spont, beta_dim, num_sample))
    align_scores = SpontObj.align_to_ffrec_alter(kappa, beta_spont, num_sample, beta_dim)
    ffrec = SpontObj.all_ffrec()
    plt.figure()
    plt.scatter(ffrec[-50:], np.flip(align_scores))
    plt.show()


    
    
    its = IntraTrialStabCorrection(n_neuron, R)
    max_align = its.max_align(dt_euler, T, sigma_time)[:, 12]
    rand_align = its.rand_align(dt_euler, T, sigma_time)[:, 12]
    its.plot_max_rand_align(rand_align, max_align)
    print(its.sorted_eigvec[:, -1][12])
    
    
    ttc = TrialToTrialCor(n_neuron, R)
    max_align = ttc.ttc_max_align(sigma_trial, N_trial, 100)
    rand_align = ttc.ttc_random_align(sigma_trial, N_trial, 100)
    ttc.hist_plot(max_align, rand_align)
    ttc.ttc_sort_align(sigma_trial, N_trial)
    

    
    test = AlignmentSpontaneousAct(3,0.85)
    pattern_A = np.array([[1,2,3],
                          [2,3,4]])
    pattern_B = np.array([[1,2,3],
                         [5,6,7]])
    print(test.align_A_to_B_alter(pattern_A, pattern_B))
    #print(test.align_to_ffrec(kappa, beta_spont, beta_dim, 5))
    '''





































