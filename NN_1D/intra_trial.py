import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from Networks import LinearRecurrentNetwork

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


    def EulerMaruyama(self, sigma_time, Delta_t_euler, h_det, T, r0):

        sde_a = lambda r, h_det : -r + self.interaction @ h_det # r is the response vector.

        r_values = [r0] # r0 is a (num_neurons,) dimensional array.

        num_steps = int(T/Delta_t_euler)
        np.random.seed(42)

        for i in range(num_steps):
            dW = np.random.normal(0, 1, size = self.neurons) * np.sqrt(Delta_t_euler)
            # Euler-Maruyama Scheme.
            r = r_values[i] + sde_a(r_values[i], h_det) * Delta_t_euler + sigma_time * dW
            r_values.append(r)

        return np.array(r_values)[1:] # steps x n - dimensional.



    def activity_rand_align(self, sigma_time, Delta_t_euler, T): # TODO: the result values are too small. Is the h_det correct?

        # Generate a random vector as h_det.
        np.random.seed(42)
        h_det = np.random.normal(0, 1, size = self.neurons)
        # Normalize h_det.
        h_det = h_det/np.linalg.norm(h_det)

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_value = self.EulerMaruyama(sigma_time, Delta_t_euler, h_det, T, r0)

        return r_value



    def activity_max_align(self, sigma_trial, Delta_t_euler, T):

        # Set the h_det as the maximal eigenvector (already normalized).
        h_det = self.eigvec[:, np.argmax(self.eigval)]

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r_value = self.EulerMaruyama(sigma_trial, Delta_t_euler, h_det, T, r0)

        return r_value



    def activity_plot(self, r_value_rand, r_value_max, T, Delta_t_euler):

        time = np.arange(0, T, Delta_t_euler)
        plt.figure()
        plt.xlabel("Time(ms)")
        plt.ylabel("normalized mean activity")
        plt.plot(time, r_value_rand[:,50], c = "green", alpha = 0.5, label = "random alignment")
        plt.plot(time, r_value_max[:,50], c = "blue", alpha = 0.5, label = "maximal alignment")
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
        # Sort eigenvectors in the order of ascending eigenvalues.
        sorted_indices = np.argsort(self.eigval)
        self.sorted_eigvec = self.eigvec[:, sorted_indices]


    '''
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

        return np.array(r_vec)[1:] # n_steps x n_neuron dimensional
    '''
    #TODO: The calculation of Euler scheme is different: l.173 vs. l.198.


    def euler_maruyama(self, dt_euler, T, h_det, sigma_time):

        start_act = np.linalg.inv(np.eye(self.neurons) - self.interaction) @ h_det
        num_steps = int(T/dt_euler)
        sqrtdt=np.sqrt(dt_euler)
        rng = np.random.default_rng(seed = None)

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
        rng = np.random.default_rng(seed=None)
        h_det = rng.normal(0, 1, size = self.neurons)
        # Normalization.
        h_det /= np.linalg.norm(h_det)

        r = self.euler_maruyama(dt_euler, T, h_det, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)



    def max_align(self, dt_euler, T, sigma_time):
        # The eigenvector corresponding with the largest eigenvalue.
        h_det = self.sorted_eigvec[:, -1]

        r = self.euler_maruyama(dt_euler, T, h_det, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)


    '''
    def min_align(self, dt_euler, T, sigma_time):
        # The eigenvector corresponding with the smallest eigenvalue.
        h_det = self.eigvec[:, 0]

        r0 = np.linalg.inv(np.identity(self.neurons) - self.interaction) @ h_det
        r = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time)
        return r
        # If want to look at the mean across neurons over time, i.e. at each time point the mean activity.
        #return r.mean(axis=1)
    '''



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


    '''
    def plot_min_max_align(self, r_max, r_min):
    
        # r_rand and r_max are (num_neuron,) dimensional array, each for response of the same neuron
        # under the random and maximal alignment.
        
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
    '''




    '''
    def intra_trial_stab(self, dt, r_vec, t):
        r = sp.stats.zscore(r_vec[t,:])
        r_delay = sp.stats.zscore(r_vec[t + dt,:])
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
        


    def stab_hdet_new(self, dt_euler, dt_intra, h_det, T, sigma_time, t_steps):

        # Define the function of intra-trial stability.
        intra_trial_stab = lambda r, r_delay : r @ r_delay
        # Access the response vector under the given h_det through Euler_Maruyama
        # with steady state as initial condition.
        r0 = np.linalg.inv(np.eye(self.neurons) - self.interaction) @ h_det # (1-J)^-1 * h_det
        r_vec = self.euler_maruyama(r0, dt_euler, h_det, T, sigma_time) # n_steps x n_neuron dimensional

        its = np.zeros(t_steps)
        for t in range(t_steps):
            r = sp.stats.zscore(r_vec[t,:])
            r_delay = sp.stats.zscore(r_vec[t + int(dt_intra/dt_euler),:])
            its[t] = intra_trial_stab(r, r_delay)

        return np.mean(its)
    '''


    def stab_hdet(self, dt_euler, dt_intra, h_det, T, sigma_time):

        # Access the response vector under the given h_det through Euler_Maruyama
        # with steady state as initial condition.
        r0 = np.linalg.inv(np.eye(self.neurons) - self.interaction) @ h_det # (1-J)^-1 * h_det
        r_vec = self.euler_maruyama(dt_euler, T, h_det, sigma_time) # n_steps x n_neuron dimensional

        dt = int(dt_intra/dt_euler)
        cor = np.corrcoef(r_vec)

        return np.mean(np.diag(cor, k=dt))


    def sort_stab(self, dt_euler, dt_intra, T, sigma_time):

        ffrec_align = []
        mean_stab = []
        # Range over sorted eigenvectors to calculate the mean stability for each of them.
        for i in range(len(self.eigval)):
            h_det = self.sorted_eigvec[:, i]
            mean_stab.append(self.stab_hdet(dt_euler, dt_intra, h_det, T, sigma_time))
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

if __name__ == "__main__":
    # Global parameters:
    sigma_trial = 0.02
    sigma_time = 0.3
    N_trial = 100
    n_neuron = 200
    R = 0.85
    T = 120
    dt_intra = 20
    dt_euler = 0.1


    Obj = Rewrite_IntraTrial(n_neuron, R)

    r_rand = Obj.rand_align(dt_euler, T, sigma_time)
    r_rand_1 = r_rand[:, 12]

    r_max = Obj.max_align(dt_euler, T, sigma_time)
    r_max_1 = r_max[:,12]

    #r_min = Obj.min_align(dt_euler, T, sigma_time)
    #r_min_1 = r_min[:, 5]

    Obj.plot_max_rand_align(r_rand_1, r_max_1)
    #Obj.plot_min_max_align(r_max_1,r_min_1)

    #plt.plot(np.linspace(0,120,1200), r_max_1, c="blue")
    #plt.plot(np.linspace(0,120,1200), r_rand_1, c="green")
    #plt.show()
    '''
    align = Obj.sort_stab(dt_euler, dt_intra, T, sigma_time)
    ffrec = align[0]
    stab = align[1]
    Obj.plot_stab(ffrec, stab)
    '''


    #h_det = np.random.normal(0,1, size = n_neuron)
    #h_det /= np.linalg.norm(h_det)
    #h_det = Obj.sorted_eigvec[:, 0]  # Der Wert sollte nah am 0 sein.
    #h_det = Obj.sorted_eigvec[:, -1]
    #h_det = Obj.sorted_eigvec[:, 100]
    '''
    its = Obj.stab_hdet(h_det, 200, 1000, dt_euler, T, sigma_time)
    print(its)
    its_new = Obj.stab_hdet_new(dt_euler, dt_intra, h_det, T, sigma_time, 1000)
    print(its_new)
    its_new2 = Obj.stab_hdet_new2(dt_euler, dt_intra, h_det, T, sigma_time)
    print(its_new2)
    '''


    #all_resp = Obj.euler_maruyama_alter(dt_euler, T, h_det, sigma_time)
    #exp = all_resp[:, 5]
    #plt.plot(np.linspace(0,120,1200), exp)
    #plt.show()














