import numpy as np

class LinearRecurrentNetwork:

    def __init__(self, n, R):
        """
        :param n: The number of neurons.
        :param R: The eigenvalue radius.
        """
        self.neuron = n
        self.R = R
        # Applying random interaction matrix.
        self.interaction = self.random_interaction_matrix()


    def random_interaction_matrix(self):
        """
        Generating the random interaction matrix (nxn dimensional) with normal distribution.
        The matrix is real, full rank and symmetric.
        :param n: The size of the network (the number of neurons).
        :param R: The overall strength of recurrent interactions.
        :return: The interaction matrix.
        """
        mu = 0
        #sigma = self.R/2*np.sqrt(self.neuron)
        #sigma = self.R/np.sqrt(self.neuron)
        # Set the seed value.
        np.random.seed(42)
        J = np.random.normal(mu, 1, size = (self.neuron,self.neuron))
        # J should be a symmetric matrix.
        J = (J + J.T)/2
        # Get the maximal eigenvalue of J.
        max_eigval = np.max(np.linalg.eigvalsh(J))
        # Normalize the J with RJ/maximal eigenvalue.
        J = self.R*J/max_eigval
        return J


    def trial_response(self, h_det, sigma_trial, N_trial):
        """
        The mean response vector given the trial input vector. Multivariate normal distributed.
        :param h_det: The deterministic part of the input (1xn dimensional).
        :param sigma_trial: The stength of the covariance matrix.
        :param N_trial: The number of trials.
        :return: The response vector matrix (Nxn dimensional) depending on the input for N trials.
        """
        new_inter = np.linalg.inv(np.identity(self.neuron) - self.interaction) # (1-J)^(-1).
        mean = np.matmul(new_inter, h_det)  # 1xn dimensional array.
        cov = (sigma_trial**2)*(np.matmul(new_inter, new_inter.T))

        # Response vector normally distributed.
        r_trial = np.random.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial

    '''
    def trial_response_alter(self, eigval, eigvec, h_det, sigma_trial, N_trial):
        # Calculations of mean and covariance matrix.
        n = len(eigval)
        mean = 0
        cov = 0
        for i in range(n):
            mean += (np.dot(eigvec[i], h_det)/(1-eigval[i])) * eigvec[i]
            vec = np.reshape(eigvec[i], (n,1))
            vecT = np.reshape(eigvec[i], (1,n))
            cov += np.matmul(vec, vecT)/(1-eigval[i])**2
        cov = sigma_trial**2*cov

        # Response vector normally distributed.
        r_trial = np.random.multivariate_normal(mean, cov, size = N_trial) # Nxn dimensional.
        return r_trial
    '''






if __name__ == "__main__":
    # Global setting.
    n_neuron = 200
    R = 0.85
    sigma_trial = 0.02
    N_trial = 100

    J = LinearRecurrentNetwork(n_neuron, R).interaction
    print(J)
    print(np.shape(J))
    print(np.linalg.matrix_rank(J))

    eigval, eigvec = np.linalg.eig(J)
    print(len(eigval))
    i = np.random.randint(0,len(eigval))
    print(i)
    h_det = eigvec[i]
    print(h_det)
    r = LinearRecurrentNetwork(n_neuron, R).trial_response(h_det, sigma_trial, N_trial)
    print(r)
    print(np.shape(r))






