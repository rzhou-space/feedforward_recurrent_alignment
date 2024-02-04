import numpy as np
from AsymNetworks import AsymLinearRecurrentNet
import matplotlib.pyplot as plt


class FPStability:

    def __init__(self, n, R):
        self.neuron = n
        self.R = R
        self.network = AsymLinearRecurrentNet(self.neuron, self.R)
        self.interaction = self.network.interaction


    def random_input(self, sigma_trial, num_trial): # Firstly with one trial starten.
        rng = np.random.default_rng(seed = 42)
        h_det = rng.normal(0, 1, size = self.neuron)
        h = rng.multivariate_normal(h_det, sigma_trial*np.eye(self.neuron), size = num_trial)
        return h # num_trial x n_neuron dimensional array.


    def euler_response(self, total_time, dt_euler, input_h, initial_r):
        """
        :param time: total time period length.
        :param dt_euler: time step width.
        :param input_h: n_neuron dimensioal numpy array.
        :param initial_r: n_neuron dimensional numpy array.
        :return: num_step x num_neuron dimensional array. The approximation of time dependent response.
        """
        # Define the differential equation for the response vector r.
        r_ODE = lambda r: -r + self.interaction @ r.T + input_h

        # Euler method calculating time dependent stepwise response vector.
        all_r = [initial_r]
        num_step = int(total_time/dt_euler)

        for istep in range(num_step):
            old_r = all_r[istep]
            new_r = old_r + dt_euler * r_ODE(old_r)
            all_r.append(new_r)

        return np.asarray(all_r)[1:]  # num_step x num_neuron dimensional array. Excl. the initial r.


    def fix_point(self, input_h):
        return np.linalg.inv(np.eye(self.neuron) - self.interaction) @ input_h  # (I-J)^-1 * h.


    def plot_time_resp(self, all_r, fix_point, last_time, dt_euler):
        plt.figure()
        for i in range(self.neuron):
            # The i the column of the all_r is the time dependent response approximated by euler.
            plt.plot(all_r[:, i])

        plt.scatter(np.repeat(last_time, self.neuron), fix_point)
        plt.title("Covergence of Neuron Activity to Fixpoints")
        plt.xlabel("time step (width = " + str(dt_euler) + "s )")
        plt.ylabel("activity")
        plt.show()


#####################################################################################################################

if __name__ == "__main__":
    n_neuron = 20
    R = 0.85
    sigma_trial = 0.05
    num_trial = 1
    total_time = 10 # seconds
    dt_euler = 0.1 # second
    # TODO: what should be the initial response vector? It should actually be near the fix
    ## point. So calculate the fix point vector and then how to determine the initial point
    ## near by then?
    # Firstly then try out with the random initial response.
    rng = np.random.default_rng(seed = 42)
    initial_r = rng.normal(0, 1, size = n_neuron)

    stab = FPStability(n_neuron, R)
    input_h = stab.random_input(sigma_trial, num_trial)
    # print(input_h)
    # print(np.shape(input_h))

    response = stab.euler_response(total_time, dt_euler, input_h[0], initial_r) # Consider the first trial.
    # print(response)
    # print(np.shape(response))

    fp = stab.fix_point(input_h[0])
    # print(fp)

    stab.plot_time_resp(response, fp, int(total_time/dt_euler)-1, dt_euler)













