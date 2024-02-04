import numpy as np

class EIHebbLearnLinear:
    # Paper source code: https://github.com/comp-neural-circuits/Synapse-type-specific-competitive-Hebbian-learning/blob/main
    # All response vectors should be (n, 1) dimensional (varied n).
    # In the original paper code, all initial weights are positive.
    # Therefore response update from inhibitory neurons with negative sign.
    # W_AB: A are the post-synaptic neurons and B are the pre-synaptic neurons.

    def __init__(self, f_neuron, ri_neuron, re_neuron):
        self.f_neuron = f_neuron # Number of feedforward neurons.
        self.ri_neuron = ri_neuron # Numebr of inhibition recurrent neurons.
        self.re_neuron = re_neuron # Number of excitatory recurrent neurons.
        self.r_neuron = ri_neuron + re_neuron # Total number of recurrent neurons.
        self.inputs = self.inputs_F() # Feedforward inputs at t0.
        # TODO: What are the initial responses?!

        '''
        # Determin randomly the set of E and I neurons. Determined by E/I rate.
        # If E/I = a -> I takes 1/a+1 percent and E takes a/a+1 percent of total recurrent neuron numbers.
        rng = np.random.default_rng(seed = 42)
        all_indices = np.arange(self.r_neuron)
        # Get the indeces of I neurons.
        i_number = np.rint(self.r_neuron / (1+self.ei_rate))
        self.i_indices = rng.choice(self.r_neuron, i_number, replace=False)
        # The rest indeces are for E neurons.
        self.e_indices = np.setdiff1d(all_indices, i_indices) # In first array but not in the second.
        '''
        # Determine randomly the E and I neurons.
        rng = np.random.default_rng(seed = 42)
        all_indices = np.arange(self.r_neuron)
        # Get the indeces of I neurons.
        self.i_indices = rng.choice(self.r_neuron, ri_neuron, replace=False)
        # The rest indeces are for E neurons.
        self.e_indices = np.setdiff1d(all_indices, i_indices) # In first array but not in the second.

    def initial_recurrent(self, mean_ee, std_ee, mean_ei, std_ei, mean_ie, std_ie, mean_ii, std_ii):
        rng = np.random.default_rng(seed=42)
        # The recurrent network ist the sum of W_AB with AB = EE, EI, IE, II.
        W_EE = np.zeros((self.r_neuron, self.r_neuron))
        W_EI = np.zeros((self.r_neuron, self.r_neuron))
        W_IE = np.zeros((self.r_neuron, self.r_neuron))
        W_II = np.zeros((self.r_neuron, self.r_neuron))
        # AB = EE
        for i in self.e_indices:
            for j in self.e_indices:
                W_EE[i, j] = abs(rng.normal(mean_ee, std_ee))
        # AB = EI
        for i in self.e_indices:
            for j in self.i_indices:
                W_EI[i, j] = abs(rng.normal(mean_ei, std_ei))
        # AB = IE
        for i in self.i_indices:
            for j in self.e_indices:
                W_IE[i, j] = abs(rng.normal(mean_ie, std_ie))
        # AB = II
        for i in self.i_indices:
            for j in self.i_indices:
                W_II[i, j] = abs(rng.normal(mean_ii, std_ii))
        # J = sum(W_AB)
        J = W_EE + W_EI + W_IE + W_II
        return W_EE, W_EI, W_IE, W_II, J

    def initial_feedforward(self, mean_ef, std_ef, mean_if, std_if):
        rng = np.random.default_rng(seed=20) # Another seed --> differs from recurrent.
        # The feedforward network is the sum of W_AB with AB = EF, IF.
        W_EF = np.zeros((self.r_neuron, self.f_neuron))
        W_IF = np.zeros((self.r_neuron, self.f_neuron))
        # AB = EF
        for i in self.e_indices:
            for j in range(self.f_neuron):
                W_EF[i, j] = abs(rng.normal(mean_ef, std_ef))
        # AB = IF
        for i in self.i_indices:
            for j in range(self.f_neuron):
                W_IF[i, j] = abs(rng.normal(mean_if, std_if))
        # W = sum(W_AB)
        W = W_EF + W_IF
        return W_EF, W_IF, W

    def inputs_F(self): # 1D inputs.
        # Gaussian distributed.
        rng = np.random.default_rng(seed = 42)
        f_input = rng.normal(0, 1, seize = (self.f_neuron,1))
        return f_input

    def response_update(self, rA_old, W_AF_old, W_AE_old, rE_old, W_AI_old
                        , rI_old, delta_t, rF = self.inputs): # Firstly consider given input only at t0.
        rA_new = rA_old + delta_t * (-rA_old + W_AF_old@rF + W_AE_old@rE_old - W_AI_old@rI_old)
        return rA_new

    def weight_update(self, W_AB_old, delta_t, lr_AB, rA_old, rB_old):
        '''
        :param rA_old: (r_neuron, 1) dimensional.
        :param rB_old: (r_neuron, 1) dimenaionsl.
        '''
        W_AB_new = W_AB_old + delta_t * lr_AB * rA_old @ rB_old.T
        return W_AB_new

    def norm_B_EF(self, W_AE, W_AF, W_AB):
        sum_AE = np.sum(W_AE)
        # Taking the sum of rows in W_AE and W_AF.
        # Then add them together and take inverse of nonzero elements.
        row_sum_AE_AF = np.sum(W_AE, axis = 1) + np.sum(W_AF, axis = 1)
        row_sum_AE_AF[row_sum_AE_AF!=0] = 1/row_sum_AE_AF[row_sum_AE_AF!=0]
        # Normalizationsfactor. (r_neuron, 1) dimensional.
        normal_factor = sum_AE * row_sum_AE_AF.reshape((self.r_neuron, 1))
        # Normalization of W_AB.
        W_AB = (W_AB.T @ normal_factor).T
        return W_AB

    def norm_B_I(self, W_AI):
        sum_AI = np.sum(W_AI)
        row_sum_AI = np.sum(W_AI, axis=1)
        row_sum_AI[row_sum_AI!=0] = 1/row_sum_AI[row_sum_AI!=0]
        normal_factor = sum_AI * row_sum_AI.reshape((self.r_neuron, 1))
        W_AI = (W_AI.T @ normal_factor).T
        return W_AI

    def weight_normalize(self, A, B, W_EE, W_EF, W_IE, W_IF, W_EI, W_II):
        # For the case of AB = EE, EF, IE and IF.
        if A == "E" and B == "E":
            return self.norm_B_EF(W_EE, W_EF, W_EE)
        elif A == "E" and B == "F":
            return self.norm_B_EF(W_EE, W_EF, W_EF)
        elif A == "I" and B == "E":
            return self.norm_B_EF(W_IE, W_IF, W_IE)
        elif A == "I" and B == "F":
            return self.norm_B_EF(W_IE, W_IF, W_IF)
        # For the case of AB = EI, II
        elif A == "E" and B == "I":
            return self.norm_B_I(W_EI)
        elif A == "I" and B == "I":
            return not self.norm_B_I(W_II)

    def ffrec_update(self, W_EE, W_EF, W_IE, W_IF, W_EI, W_II, rF = self.inputs):
        h = (W_EF + W_IF) @ rF
        h = h / np.linalg.norm(h)
        J = W_EE + W_EI + W_IE + W_II
        ffrec = h.T @ J @ h
        return ffrec

    def update_dynamics(self, delta_t, T,
                        mean_ef, std_ef, mean_if, std_if,
                        ean_ee, std_ee, mean_ei, std_ei, mean_ie, std_ie, mean_ii, std_ii,
                        rE_0, rI_0,
                        lr_EE, lr_EI, lr_IE, lr_II, lr_EF, lr_IF): # TODO: initial responses.
        t_list = list(range(0, T+1, delta_t))
        # Initial conditions
        init_ff = self.initial_feedforward(mean_ef, std_ef, mean_if, std_if)
        init_rec = self.initial_recurrent(ean_ee, std_ee, mean_ei, std_ei, mean_ie, std_ie, mean_ii, std_ii)
        # Storing the weight matrices.
        #W = [init_ff[2]]
        #J = [init_rec[4]]
        rE = [rE_0]
        rI = [rI_0]

        WEE_0 = init_rec[0]
        WEI_0 = init_rec[1]
        WIE_0 = init_rec[2]
        WII_0 = init_rec[3]
        WEF_0 = init_ff[0]
        WIF_0 = init_ff[1]
        # Normalization of the initial weights.
        WEE_0 = self.weight_normalize("E", "E", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)
        WEI_0 = self.weight_normalize("E", "I", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)
        WIE_0 = self.weight_normalize("I", "E", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)
        WII_0 = self.weight_normalize("I", "I", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)
        WEF_0 = self.weight_normalize("E", "F", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)
        WIF_0 = self.weight_normalize("I", "F", WEE_0, WEF_0, WIE_0, WIF_0, WEI_0, WII_0)

        W_EE = [WEE_0]
        W_EI = [WEI_0]
        W_IE = [WIE_0]
        W_II = [WII_0]
        W_EF = [WEF_0]
        W_IF = [WIF_0]
        ffrec = [self.ffrec_update(W_EE[0], W_EF[0], W_IE[0], W_IF[0], W_EI[0], W_II[0])]
        # Updating.
        for t in range(1, len(t_list)):
            # Updating.
            # Update the responses.
            rE_new = self.response_update(rE[t-1], W_EF[t-1], W_EE[t-1], rE[t-1], W_EI[t-1]
                        , rI[t-1], delta_t)
            rI_new = self.response_update(rI[t-1], W_IF[t-1], W_IE[t-1], rE[t-1], W_II[t-1]
                        , rI[t-1], delta_t)
            rE.append(rE_new)
            rI.append(rI_new)
            # Update the weights. (Assume the feedforward inputs only given at t=0)
            WEE_new = self.weight_update(W_EE[t-1], delta_t, lr_EE, rE[t-1], rE[t-1])
            WEI_new = self.weight_update(W_EI[t-1], delta_t, lr_EI, rE[t-1], rI[t-1])
            WIE_new = self.weight_update(W_IE[t-1], delta_t, lr_IE, rI[t-1], rE[t-1])
            WII_new = self.weight_update(W_II[t-1], delta_t, lr_II, rI[t-1], rI[t-1])
            WEF_new = self.weight_update(W_EF[t-1], delta_t, lr_EF, rE[t-1], rB_old = self.inputs)
            WIF_new = self.weight_update(W_IF[t-1], delta_t, lr_IF, rI[t-1], rB_old = self.inputs)
            # Normalization of weights.
            WEE_new = self.weight_normalize("E", "E", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            WEI_new = self.weight_normalize("E", "I", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            WIE_new = self.weight_normalize("I", "E", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            WII_new = self.weight_normalize("I", "I", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            WEF_new = self.weight_normalize("E", "F", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            WIF_new = self.weight_normalize("I", "F", WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)

            W_EE.append(WEE_new)
            W_EI.append(WEI_new)
            W_IE.append(WIE_new)
            W_II.append(WII_new)
            W_EF.append(WEF_new)
            W_IF.append(WIF_new)

            # Update the ffrec score.
            ffrec_new = self.ffrec_update(WEE_new, WEF_new, WIE_new, WIF_new, WEI_new, WII_new)
            ffrec.append(ffrec_new)

        return ffrec

###############################################################################################
    # TODO: test of all single functions.
    # TODO: a lot of unknown parameters -- could try to find them in the original paper...






