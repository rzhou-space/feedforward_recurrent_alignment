import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from AsymNetworks import AsymLinearRecurrentNet, CombiAsymNet#, CombiAsymNet2



class FFRec_Alignment:

    def real_ffrec_align(self, h_det, interaction):
        """
        :param h_det: 1 x n_neuron numpy array. Deterministic part of input. Generally a complex vector.
        :return: feedforward recurrent alignment defined with the real part of h_det elementwise.
        """
        real_h = np.real(h_det)
        # Normalization of real_h.
        h = real_h/np.linalg.norm(real_h)
        # Insert into the general feedforward recurrent formular.
        ffrec = h @ interaction @ h
        return ffrec


    def mag_ffrec_align(self, h_det, interaction):
        """
        :param h_det: 1 x n_neuron numpy array. Deterministic part of input. Generally a complex vector.
        :return: feedforward recurrent alignment defined with the magnitude of h_det elementwise.
        """
        mag_h = np.abs(h_det)
        # Normalization of mag_h.
        h = mag_h/np.linalg.norm(mag_h)
        # Insert into the general formular.
        ffrec = h @ interaction @ h
        return ffrec


    def ffrec_align(self, h_det, interaction):
        # h_det real vector.
        h = h_det/np.linalg.norm(h_det)
        ffrec = h @ interaction @ h
        return ffrec



##############################################################################################################

class Eigval_Ffrec:

    def __init__(self, n, R, combi, a, b):
        self.neuron = n
        self.R = R
        self.combi = combi # Equals 0: no combined interaction matrix/ 1,2: with combined interaction matrix.
        self.a = a # For the calculation of combined interaction matrix 1.
        self.b = b # For the calculation of combined interaction matrix 2.
        if self.combi == 0:
            self.network = AsymLinearRecurrentNet(self.neuron, self.R)
            self.interaction = self.network.interaction
            # Here used the right eigenvector!
            self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        elif self.combi == 1:
            self.network = CombiAsymNet(self.neuron, self.a, self.R) # J = a*J_sym + (1-a)*J_asym
            self.interaction = self.network.interaction
            # Right eigenvectors.
            self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        '''
        elif self.combi == 2:
            self.network = CombiAsymNet2(self.neuron, self.b, self.R) # J = J_sym + b*J_asym
            self.interaction = self.network.interaction
            # Right eigenvectors.
            self.eigval, self.eigvec = np.linalg.eig(self.interaction)
        '''


    def real_eigval_ffrec(self):
        # Calculate the ffrec-alignment.
        ffrec = np.zeros(len(self.eigval))
        for i in range(len(self.eigval)):
            # The i-th column is the corresponding eigenvector of i-th eigenvalue.
            h_det = self.eigvec[:, i]
            ffrec[i] = FFRec_Alignment.real_ffrec_align(self, h_det, self.interaction)

        # Correlation between eigenvalues and ffrec-alignments.
        x = self.eigval.real
        y = self.eigval.imag
        #plt.title("Ffrec align in eigenvalue plane with h = Re($e_i$)")
        plt.xlabel(r'Re($\lambda$)', fontsize=15)
        plt.ylabel(r'Im($\lambda$)', fontsize=15)
        plt.scatter(x, y, c=ffrec, cmap = "PuBuGn")
        cbar = plt.colorbar()
        #cbar.set_label("ffrec align", fontsize=15)
        plt.show()

        return ffrec


    def mag_eigval_ffrec(self):
        # Calculate the ffrec-alignment.
        ffrec = np.zeros(len(self.eigval))
        for i in range(len(self.eigval)):
            # The i-th column is the corresponding eigenvector of i-th eigenvalue.
            h_det = self.eigvec[:, i]
            ffrec[i] = FFRec_Alignment.mag_ffrec_align(self, h_det, self.interaction)

        # Correlation between eigenvalues and ffrec-alignments.
        x = self.eigval.real
        y = self.eigval.imag
        #plt.title("Ffrec align in eigenvalue plane with h = |$e_i$|")
        plt.xlabel(r"Re($\lambda$)", fontsize=15)
        plt.ylabel(r"Im($\lambda$)", fontsize=15)
        plt.scatter(x, y, c=ffrec, cmap = "PuBuGn")
        cbar = plt.colorbar()
        cbar.set_label("Feedforward recurrent alignment", fontsize=15, rotation=270, labelpad=15)
        plt.show()

        return ffrec


    def sym_eigval_ffrec(self):
        # Symmetrisize the original asymmetrical interaction matrix.
        sym_inter = (self.interaction + self.interaction.T)/2
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)
        # Normalize the range of eigenvalue distribution limited by R.
        sym_inter = sym_inter * self.R/np.max(sym_eigval)
        # Calculate again the eigval and eigvec.
        sym_eigval, sym_eigvec = np.linalg.eigh(sym_inter)

        ffrec = np.zeros(len(sym_eigval))
        for i in range(len(sym_eigval)):
            # h_det is the i-th eigenvector of the symmtrisized matrix. They are therefore real.
            h_det = sym_eigvec[:, i]
            # Calculate the ffrec still with the original asymmetrical interaction matrix.
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)

        # Correlation between eigenvalues of original matrix and ffrec-alignments.
        #x = self.eigval.real
        #y = self.eigval.imag
        #plt.title("Ffrec align in eigenvalue plane with h = |$e_i$| of symmetrized J")
        #plt.xlabel("Re($\lambda$)")
        #plt.ylabel("Im($\lambda$)")
        #plt.scatter(x, y, c=ffrec, cmap = "PuBuGn")
        plt.xlabel(r"$\tilde{\lambda}$", fontsize=15)
        plt.ylabel("Feedforward recurrent alignment", fontsize=15)
        plt.scatter(sym_eigval, ffrec)
        #cbar = plt.colorbar()
        #cbar.set_label("ffrec align")
        plt.show()

        return ffrec, sym_eigval  # ffrec-align calculated with original J, eigenvalues from symmetrized interaction J.


    def noise_eigval_ffrec(self, num_sample):
        '''
        # Generate response of white noise.
        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1
        rng = np.random.default_rng(seed = 42)
        spont_act = rng.multivariate_normal(mean = np.zeros(self.neuron),
                                                    cov = np.matmul(steady_inter, steady_inter.T), # (I-J)^-1 * (I-J)^-T
                                                    size = num_sample).T
        pca = PCA(n_components=self.neuron)
        pc = pca.fit_transform(spont_act) # Get the principal components. All are real.
        variance_ratio = pca.explained_variance_ratio_  # The corresponding variance ratio.

        # Sort the eigenvalues of interaction matrix in descending order (done in the plot method below).
        ffrec = np.zeros(self.neuron)
        for i in range(self.neuron):
            # h_det is the i-th principal component <-> the i-th column of the PC matrix.
            h_det = pc[:, i]
            # h_det real. Calculate the normal ffrec.
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)
        '''
        # Alternative of determine the PCs directly through the eigenvectors and eigenvalues
        # of the covariance matrix.

        steady_inter = np.linalg.inv(np.eye(self.neuron) - self.interaction) # (I-J)^-1
        act_cov = np.matmul(steady_inter, steady_inter.T)
        cov_eigval, cov_eigvec = np.linalg.eigh(act_cov) # Covariance matrix symmetrical.
        # Sort eigenvectors (PCs) and eigenvalues (variance ratio) in descending order.
        sort_index = np.argsort(cov_eigval)[::-1]
        variance_ratio = cov_eigval[sort_index]
        PCs = cov_eigvec[:, sort_index]
        # Calculate the ffrec with PCs (i.e., eigvectors of cov) of spontaneous activity.
        ffrec = np.zeros(self.neuron)
        for i in range(self.neuron):
            # h_det is the i-th PC <-> the i-th cov_eigvec.
            h_det = PCs[:, i]
            ffrec[i] = FFRec_Alignment.ffrec_align(self, h_det, self.interaction)

        # Plot the correlation between odered eigenvalues and ffrec-alignments.
        '''
        plt.title("Ffrec align in eigenvalue plane with h = PCs from white noise")
        plt.xlabel("variance ratio")
        plt.ylabel("ffrec")
        plt.scatter(variance_ratio, ffrec)
        cbar = plt.colorbar()
        cbar.set_label("ffrec align")
        plt.show()
        '''

        return ffrec, variance_ratio  # ffrec-align calculated with original J, variance ratio from white noise.




    def all_plots(self, num_sample):

        x = self.eigval.real
        y = self.eigval.imag

        real_ffrec = self.real_eigval_ffrec()
        mag_ffrec = self.mag_eigval_ffrec()
        sym_ffrec = self.sym_eigval_ffrec()[0]
        noise_ffrec = self.noise_eigval_ffrec(num_sample)[0]
        # If needed: For the case of applying PC for ffrec-align.
        descend_sort_eigval = np.sort(self.eigval)[::-1]

        ffrecs = np.concatenate((real_ffrec, mag_ffrec, sym_ffrec, noise_ffrec))
        min_, max_ = ffrecs.min(), ffrecs.max()
        # Normalize the color values for scatter plots
        norm = Normalize(min_, max_)

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle("Ffrec align in eigenvalue planes")

        # Ffrec values without rescaling to colorbar-range.
        # Subplot for the case with real parts.
        plt.sca(ax = axs.flat[0])
        plt.scatter(x, y, c = real_ffrec, cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.title("real part $e_i$")
        plt.xlabel("Re($\lambda$)")
        plt.ylabel("Im($\lambda$)")

        # Subplot for the case with magnitude.
        plt.sca(ax = axs.flat[1])
        plt.scatter(x, y, c = mag_ffrec, cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.title("magnitude $e_i$")
        plt.xlabel("Re($\lambda$)")
        plt.ylabel("Im($\lambda$)")

        # Subplot for the case with symmetrized interaction matrix.
        # x-axis: the eigenvalues. y-axis: ffrec values. Color-coded with ffrec.
        plt.sca(ax = axs.flat[2])
        # plt.scatter(x, y, c = sym_ffrec, cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.scatter(self.sym_eigval_ffrec()[1], sym_ffrec, c = sym_ffrec,
                    cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.title("symmetrized J")
        plt.xlabel("$\lambda$")
        plt.ylabel("ffrec align")

        # Subplot for the case with PC from white noise. Eigenvalues sorted in descending order.
        # x-axis: the variance explained. y-axis: ffrec align calculated by corresponding PCs.
        plt.sca(ax = axs.flat[3])
        # plt.scatter(descend_sort_eigval.real, descend_sort_eigval.imag, c = noise_ffrec,
        #            cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.scatter(self.noise_eigval_ffrec(num_sample)[1], noise_ffrec, c = noise_ffrec,
                    cmap = "PuBuGn", vmin = min_, vmax = max_)
        plt.title("PCs from white noise")
        plt.xlabel("variance explained")
        plt.ylabel("ffrec align")

        # General settings.
        cbar = plt.colorbar(ax = axs.ravel().tolist())
        cbar.ax.set_title("ffrec-align")
        # Set the common x, y-axis labels.
        #plt.setp(axs[-1, :], xlabel='Re($\lambda$)')
        #plt.setp(axs[:, 0], ylabel='Im($\lambda$)')
        # Set uniformly the x and y axis range.
        # plt.setp(axs, xlim = (-self.R, self.R), ylim = (-self.R, self.R))

        plt.show()

        '''
        # Ffrec values with rescaling in colorbar-range.
        axs[0,0].scatter(x, y, c=real_ffrec, cmap = "PuBuGn")
        axs[0,1].scatter(x, y, c=mag_ffrec, cmap = "PuBuGn")
        axs[1,0].scatter(x, y, c=sym_ffrec, cmap = "PuBuGn")
        axs[1,1].scatter(descend_sort_eigval.real, descend_sort_eigval.imag, c=noise_ffrec, cmap = "PuBuGn")
        
        # TODO: share the same colorbar. All ffrecs will be recaled to the colorbar range.
        ## https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        # Create a common ScalarMappable
        sm = ScalarMappable(cmap='PuBuGn', norm=norm)
        sm.set_array([])  # Set an empty array to the ScalarMappable.
        # Create a common colorbar for subplots.
        cbar = fig.colorbar(sm, ax=axs)
        cbar.set_label('ffrec align')
        plt.show()
        '''








############################################################################################################
if __name__ == "__main__":
    n_neuron = 200
    R = 0.85
    num_sample = 500

    plots = Eigval_Ffrec(n_neuron, R, 1, 0.2, 0.5)
    plots.real_eigval_ffrec()
    plots.mag_eigval_ffrec()
    plots.sym_eigval_ffrec()
    # plots.noise_eigval_ffrec(num_sample)
    # plots.all_plots(num_sample)

