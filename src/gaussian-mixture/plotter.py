import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
from os import mkdir, path


class Plotter:
    """Generates plots for model training and evaluation data.
    
    Attributes:
        dir (str): Path to output directory.
        suffix (str): Suffix to apply to the end of output file names.
        max_gaussians: Maximum number of Gaussians used to train GMM on input data.

    """
    def __init__(self, max_gaussians, filename, suffix=datetime.now().strftime("%Y%m%d_%H%M%S")):

        self.suffix = suffix
        self.max_gaussians = max_gaussians
        self.filename = filename
        self.outdir = path.join(path.dirname(__file__), "Sur12_Task16_Results" + "\\" + self.filename + "\\" + self.suffix)
        if not path.exists(self.outdir):
            mkdir(self.outdir)


    def plot_bic(self, bic_values, gaussians=[]):
        """Plots BIC score vs. number of Gaussians for a set of Gaussian Mixture Models.
        
        Args:
            bic_values (list of float): List of BIC scores.
            gaussians (list of int): Optional x values for plot. If the list is empty,
                values are assumed to be [1, ..., len(bic_values)].

        """
        if gaussians == []:
            gaussians = np.arange(1, len(bic_values) + 1)

        fig = plt.figure()

        plt.plot(gaussians, bic_values)
        plt.title("BIC vs. Number of Gaussians")
        plt.xlabel("Number of Gaussians")
        plt.ylabel("BIC Score")
        plt.xticks(gaussians)
        # plt.show()

        fig.savefig(f"{self.outdir}/BIC_{self.suffix}.png")

    def plot_aic(self, aic_values, gaussians=[]):
        """Plots AIC score vs. number of Gaussians for a set of Gaussian Mixture Models.
        
        Args:
            aic_values (list of float): List of AIC scores.
            gaussians (list of int): Optional x values for plot. If the list is empty,
                values are assumed to be [1, ..., len(aic_values)].

        """
        if gaussians == []:
            gaussians = np.arange(1, len(aic_values) + 1)

        fig = plt.figure()

        plt.plot(gaussians, aic_values)
        plt.title("AIC vs. Number of Gaussians")
        plt.xlabel("Number of Gaussians")
        plt.ylabel("AIC Score")
        plt.xticks(gaussians)
        # plt.show()

        fig.savefig(f"{self.outdir}/AIC_{self.suffix}.png")
