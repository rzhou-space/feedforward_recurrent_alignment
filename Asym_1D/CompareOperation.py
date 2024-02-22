import matplotlib.pyplot as plt
import matplotlib.colors as cm
import seaborn as sns
import numpy as np
import AsymNetworks as AN
import scipy as sp
from sklearn.decomposition import PCA
from NN_1D import NWoperations
import AsymOperations as AO
from NN_1D import Networks as SN  # Symmetrical networks.
# For trying out... import the Still Work module.
import AsymStillWork as ASW



# Trial to trial correlation comparisom of different interaction matrix.
#############################################################################################################
# Asymmetrical networks that could be taken into account.

'''
def compare_ttc(n_neuron, R, sigma_trial, N_trial):
    random_asymnet = AsymLinearRecurrentNet(n_neuron, R)
    combi_asymnet = CombiAsymNet(n_neuron, 0.5, R)  # J = a*J_sym + (1-a)*J_asym. a = 0.5
    # combi_asymnet2 = CombiAsymNet2(n_neuron, 0.5, R)  # J = J_sym + b*J_asym. b = 0.5
    networks = [random_asymnet, combi_asymnet]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    names = ["random asym", "J = a*J_sym + (1-a)*J_asym", "J = J_sym + b*J_asym"]

    # Trial to trial correlation for all asymmetrical networks.
    plt.figure()
    plt.title("Trial to trial Correlation (with real part)")
    plt.xlabel("ffrec-align")
    plt.ylabel("ttc")
    for i in range(len(networks)):
        net = networks[i]
        ttc_asymclass = AO.TrialtoTrialCor(n_neuron, R, net)
        # ttc.hist_real_cor_distribution(sigma_trial, N_trial)
        asym_results = ttc_asymclass.real_ttc_sort_align(sigma_trial, N_trial)
        ffrec_asym = asym_results[0]
        ttc_asym = asym_results[1]
        plt.scatter(ffrec_asym, ttc_asym, c=colors[i], alpha=0.5, label= names[i])

    # Import the results from symmetrical case.
    ttc_symclass = NWoperations.TrialToTrialCor(n_neuron, R)
    sym_results = ttc_symclass.ttc_sort_align(sigma_trial, N_trial)
    ffrec_sym = sym_results[1]
    ttc_sym = sym_results[0]
    plt.scatter(ffrec_sym, ttc_sym, c="red", alpha=0.3, label="random sym")

    plt.legend()
    plt.show()
'''


def compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, mode):
    """
    Compare the trial to trial correlation with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure()
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        ttc_class = AO.TrialtoTrialCor(n_neuron, R, network)
        if mode == "real part":
            results = ttc_class.real_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "symmetrized":
            results = ttc_class.sym_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "white noise":
            results = ttc_class.noise_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="a="+str(a))

    plt.legend(fontsize = 15)
    plt.show()


def compare_ttc_low_rank(n_neuron, R, sigma_trial, N_trial, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 10]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)] # Colors that not differ much.
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure()
    for i in range(len(rank_values)):
        D = rank_values[i]
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2) # Asymmetrical low rank without noise.
        #network = SN.NoisedLowRank_1D(n_neuron, R) # Symmetrical noised low rank.
        network = AN.NoisedLowRank(n_neuron, D, R) # Asymmetric with noise.
        ttc_class = AO.TrialtoTrialCor(n_neuron, R, network)
        if mode == "real part":
            results = ttc_class.real_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank="+str(D))
            #sns.lineplot(x=ffrec, y=ttc, label="rank="+str(D))
        elif mode == "symmetrized":
            results = ttc_class.sym_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=15)
            plt.ylabel("Trial-to-trial correlation", fontsize=15)
        elif mode == "white noise":
            results = ttc_class.noise_ttc_sort_align(sigma_trial, N_trial)
            ffrec = results[0]
            ttc = results[1]
            plt.scatter(ffrec, ttc, c=colors[i], alpha=0.5, label="rank="+str(D))

    #plt.xlabel("Feedforward recurrent alignment", fontsize=20)
    #plt.ylabel("Trial-to-trial correlation", fontsize=20)
    #plt.legend()
    plt.show()


# could plots/figures directly be put together?
'''
def all_plots(n_neuron, R, sigma_trial, N_trial):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle("Trial to Trial correlation with $J = a \cdot J_{sym} + (1-a) \cdot J_{asym}$ ")

    # With the real part.
    plt.sca(ax = axs.flat[0])
    real = compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, "real part")
    plt.scatter(real[0], real[1], c=colors[i], alpha=0.5, label="a="+str(a))
    plt.title("real part")

    # With symmetrized J.
    plt.sca(ax = axs.flat[1])
    sym = compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, "symmetrized")
    plt.scatter(sym[0], sym[1], c=colors[i], alpha=0.5, label="a="+str(a))
    plt.title("symmetrized")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels)
    plt.setp(axs[-1, :], xlabel='ffrec align')
    plt.setp(axs[:, 0], ylabel='ttc')
    plt.show()
'''

# Intra Trial Stability
##############################################################################################################
def compare_its_only_combi_inter(n_neuron, R, dt_euler, dt_intra, T, sigma_time, mode):
    """
    Compare the intra trial correlation with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure()
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        its_class = AO.IntraTrialStab(n_neuron, R, network)
        if mode == "real part":
            results = its_class.plot_real_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "symmetrized":
            results = its_class.sym_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "white noise":
            results = its_class.noise_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="a="+str(a))

    #plt.legend(fontsize=15)
    plt.show()


def compare_its_low_rank(n_neuron, R, dt_euler, dt_intra, T, sigma_time, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1,10]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure()
    for i in range(len(rank_values)):
        D = rank_values[i]
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        network = AN.NoisedLowRank(n_neuron, D, R)
        its_class = AO.IntraTrialStab(n_neuron, R, network)
        if mode == "real part":
            results = its_class.plot_real_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            results = its_class.sym_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=15)
            plt.ylabel("Intra-trial stability", fontsize=15)
        elif mode == "white noise":
            results = its_class.noise_sort_stab(dt_euler, dt_intra, T, sigma_time)
            ffrec = results[0]
            its = results[1]
            plt.scatter(ffrec, its, c=colors[i], alpha=0.5, label="rank="+str(D))

    #plt.legend()
    plt.show()



# Dimensionality
##############################################################################################################
def compare_dim_only_combi_inter(n_neuron, R, kappa, beta_dim, num_sample, mode):
    """
    Compare the dimensionality with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    # Here is the bijective projection from L to ffrec (0,1).
    plt.figure()
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        ffrec = np.linspace(0,1, int(n_neuron/2))
        dim_class = AO.Dimensionality(n_neuron, R, network)
        if mode == "real part":
            analytical_dim = dim_class.real_dim_to_ffrec(kappa, beta_dim)
            empir_dim = dim_class.real_dim_to_ffrec_empir(kappa, beta_dim, num_sample)
            #ffrec = dim_class.real_ffrec_dim()
            # Plot the analytical dimensionality as line and the empirical dimensionality
            # as dots in the same color.
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.5, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "symmetrized":
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.sym_dim_empir(kappa, beta_dim, num_sample)
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="a="+str(a))
        elif mode == "white noise":
            # Still apply J_sym for analytical dimensionality.
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.noise_dim_empir(kappa, beta_dim, num_sample)
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="a="+str(a))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="a="+str(a))

    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", ncol=5 ,  fontsize=15)
    plt.show()


def compare_dim_low_rank(n_neuron, R, kappa, beta_dim, num_sample, mode, sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 10]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    # Here is the bijective projection from L to ffrec (0,1).
    plt.figure()
    for i in range(len(rank_values)):
        D = rank_values[i]
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        network = AN.NoisedLowRank(n_neuron, D, R)
        #network = SN.NoisedLowRank_1D(n_neuron, R)
        ffrec = np.linspace(0,1, int(n_neuron/2))
        dim_class = AO.Dimensionality(n_neuron, R, network)
        if mode == "real part":
            analytical_dim = dim_class.real_dim_to_ffrec(kappa, beta_dim)
            empir_dim = dim_class.real_dim_to_ffrec_empir(kappa, beta_dim, num_sample)
            # Plot the analytical dimensionality as line and the empirical dimensionality
            # as dots in the same color.
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.sym_dim_empir(kappa, beta_dim, num_sample)
            #ffrec_sym = dim_class.sym_ffrec()
            #plt.scatter(np.flip(ffrec_sym), np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            #plt.scatter(np.flip(ffrec_sym), np.flip(empir_dim), c="green", alpha=0.6, label="rank="+str(D))
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.xlabel("Feedfroward recurrent alignment", fontsize=15)
            plt.ylabel("Dimensionality", fontsize=15)
        elif mode == "white noise":
            # Still apply J_sym for analytical dimensionality.
            analytical_dim = dim_class.sym_dim_analytical(kappa, beta_dim)
            empir_dim = dim_class.noise_dim_empir(kappa, beta_dim, num_sample)
            plt.plot(ffrec, np.flip(analytical_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
            plt.scatter(ffrec, np.flip(empir_dim), c=colors[i], alpha=0.6, label="rank="+str(D))
    #plt.legend()
    plt.show()

##############################################################################################################
# ALignment to spontaneous activity.
def compare_align_spont_combi(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, mode):
    """
    Compare the alignment to spontaneous activity with J = a*J_sym + (1-a)*J_asym of varias a value.
    """
    a_values = np.linspace(0, 1, 5)
    colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    plt.figure()
    for i in range(len(a_values)):
        a = a_values[i]
        network = AN.CombiAsymNet(n_neuron, a, R)
        align_class = AO.AlignmentSpontaneousAct(n_neuron, R, network)
        if mode == "real part":
            results = align_class.real_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="a="+str(a))
        elif mode == "symmetrized":
            results = align_class.sym_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="a="+str(a))
            plt.xlabel("Feedforward recurrent alignment", fontsize = 15)
            plt.ylabel("Alignment to spontaneous activity", fontsize=15)
        elif mode == "white noise":
            results = align_class.noise_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.6, label="a="+str(a))

    plt.legend(fontsize=15)
    plt.show()


def compare_align_spont_low_rank(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, mode,
                                 sigma_1=1, sigma_2=1):
    # Variation on the rank of the interaction.
    rank_values = [1, 10]
    #colors = [cm.to_hex(plt.cm.tab20b(i)) for i in range(20)]
    colors = [cm.to_hex(plt.cm.tab10(i)) for i in range(10)]
    plt.figure()
    for i in range(len(rank_values)):
        D = rank_values[i]
        #network = AN.LowRank(n_neuron, D, R, sigma_1, sigma_2)
        #network = SN.NoisedLowRank_1D(n_neuron, R)
        network = AN.NoisedLowRank(n_neuron, D, R)
        align_class = AO.AlignmentSpontaneousAct(n_neuron, R, network)
        if mode == "real part":
            results = align_class.real_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="rank="+str(D))
        elif mode == "symmetrized":
            results = align_class.sym_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.5, label="rank="+str(D))
            plt.xlabel("Feedforward recurrent alignment", fontsize=15)
            plt.ylabel("Alignment to spont.act", fontsize=15)
        elif mode == "white noise":
            results = align_class.noise_pattern_align(kappa, beta_spont, beta_dim, num_sample)
            ffrec = results[0]
            pattern_align = results[1]
            plt.scatter(ffrec, pattern_align, c=colors[i], alpha=0.6, label="rank="+str(D))
    #plt.legend()
    plt.show()
##############################################################################################################
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
    sigma_2 = 2

##############################################################################################################
    # Results of applying combined interaction matrix J = a*J_sym + (1-a)*J_asym

    #compare_ttc_only_combi_inter(n_neuron, R, sigma_trial, N_trial, "real part")

    #compare_its_only_combi_inter(n_neuron, R, dt_euler, dt_intra, T, sigma_time, "symmetrized")

    #compare_dim_only_combi_inter(n_neuron, R, kappa, beta_dim, num_sample, "real part")

    #compare_align_spont_combi(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, "symmetrized")

##############################################################################################################

    # Results of applying row ranked interation matrix.

    #compare_ttc_low_rank(n_neuron, R, sigma_trial, N_trial, "symmetrized")

    compare_its_low_rank(n_neuron, R, dt_euler, dt_intra, T, sigma_time, "symmetrized")

    #compare_dim_low_rank(n_neuron, R, kappa, beta_dim, num_sample, "symmetrized")

    #compare_align_spont_low_rank(n_neuron, R, kappa, beta_spont, beta_dim, num_sample, "symmetrized")






