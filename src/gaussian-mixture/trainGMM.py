"""
Train Gaussian Mixture Models
"""
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import os
from os import path, listdir, mkdir
from os.path import isfile, join
from scipy.io import savemat
import math
import random
import sklearn.datasets, sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from dvrkKinematicParser import DvrkKinematicParser
from plotter import Plotter
from misc.color import Color
# import matlab.engine

class TrainGMM():
    """
    Class for training the GMM model.
    """
    TOP_LEVEL_DIR = path.dirname(__file__)

    def __init__(self):
        self.date = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Arguments
        self.__args = None

        # Important Param to Train GMM
        self.__directory = None
        self.__file_names = None
        self.__max_gaussians = None
        self.__num_init = None
        self.__arms = None
        self.__validation_split = None

        # Set Training Input Parameters 
        self.__inputs = None
        self.__optimal_Gaussians = None
        self.__files = None
        self.__datapoints = None

        # Parsing Arguments
        self.__parseArg()
        self.__allocateArgs()

        # Read Training Data from Excel Files and store in a numpy array
        self.__read_TrainData()

        # Read Validation Data from Excel Files and store in a numpy array
        self.__read_ValidData()
        
        #Merge Training and Validation Data
        self.__mergeAllData()

        # Standardize the data. eg. z = (X - mean)/sqrt(variance)
        self.__standardize()

        # split into inputs (cols ~ ECM) and outputs (cols = ECM)
        self.__splitData()

        # perform PCA on inputs and outputs separately
        self.__PCA()

        # merge the outputs of the PCA of the input and output
        self.__mergeData()

        # Shuffle data and split based on training and validation
        self.__shuffleData()

        # Create directory to save file inside directory
        self.__createDir()

        # Training the Model
        self.__train()

        # Analysis
        self.__analyze()

        # Save Model
        self.__saveModel()

        # #Run GMR using Matlab session. This should return actual output, predicted output, nan column values and minimum similarity score
        # self.__runGMR()

        # #Revert PCA
        # self.__revertPCA()
        
        # #Revert Standardization
        # self.__revertStandardize()

        # #Plot Predicted vs actual with errors computed
        # self.__plot3D()

        # #Save all data
        # self.__save()

    def __parseArg(self):
        self.__parser = argparse.ArgumentParser(prog=program_name, description=program_desc)
        self.__parser.add_argument("-td", "--train_directory",
                            type=str,
                            default='data/Train/',
                            help="Enter training folder directory.")

        self.__parser.add_argument("-vd", "--valid_directory",
                            type=str,
                            default='data/Train/',
                            help="Enter validation folder directory.")
        
        self.__parser.add_argument("-id", "--input_data",
                            type=str,
                            nargs="+",
                            help="Spreadsheet(s) of training data.")
        self.__parser.add_argument("-g", "--max_gaussians",
                            type=int,
                            default=10,
                            help="Maximum number of Gaussians to try.")
        self.__parser.add_argument("-n", "--num_init",
                            type=int,
                            default=100,
                            help="Number of K-means initializations to try.")
        self.__parser.add_argument("-a", "--arms",
                            type=str,
                            default='123',
                            help="PSM 1, 2, or 3. Format: PSM1 and PSM2 => Enter '12' ")

        self.__parser.add_argument("-ti", "--train_inputs",
                            type=str,
                            default='123',
                            help="Train on position is 1, orientation is 2 and/or eye gaze is 3. Format: Position and Orientation => Enter '12' ")

        self.__parser.add_argument("-to", "--train_outputs",
                            type=str,
                            default='12',
                            help="If the output position is includeded then 1. If output orientation is included then 2")

        self.__parser.add_argument("-v", "--validate",
                            type=int,
                            default=-1,
                            help="Percent of data to reserve for validation")
                            
        self.__args = self.__parser.parse_args()


    def __allocateArgs(self):
        self.__train_directory = self.__args.train_directory
        self.__valid_directory = self.__args.valid_directory

        self.__file_names = self.__args.input_data
        self.__max_gaussians = self.__args.max_gaussians
        self.__num_init = self.__args.num_init
        self.__arms = list(map(lambda x: int(x), list(self.__args.arms)))

        self.__inputs = list(map(lambda x: int(x), list(self.__args.train_inputs)))
        self.__outputs = list(map(lambda x: int(x), list(self.__args.train_outputs)))

        self.__validation_split = self.__args.validate


    def __read_TrainData(self):
        if self.__file_names and len(self.__file_names) > 0:
            print("Using individual file names")
            data_dir = path.abspath(path.join(self.TOP_LEVEL_DIR, "data/OldData/"))
            train_file_names_to_process = [path.join(data_dir, file) for file in self.__file_names]
        elif self.__train_directory:
            print("Using directory")
            data_dir = path.abspath(path.join(self.TOP_LEVEL_DIR, self.__train_directory))
            files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('xlsx')]
            train_file_names_to_process = [path.join(data_dir, file) for file in files]

        self.__train_files = [os.path.basename(file) for file in train_file_names_to_process]
        print(self.__train_files)
        
        print("Training on: ")
        if 1 in self.__inputs:
            print("Position ")
        
        if 2 in self.__inputs:
            print("Orientation ")

        if 3 in self.__inputs:
            print("Left and Right POG")
        
        if 4 in self.__inputs:
            print("BPOG")

        if 5 in self.__inputs:
            print("Pupil Diameter")

        if 6 in self.__inputs:
            print("Gaze Distance")

        print("Training outputs: ")

        if 1 in self.__outputs:
            print("Position")
        
        if 2 in self.__outputs:
            print("Orientation")

        self.__data_training, self.__all_variable_names, self.__train_datapoints = DvrkKinematicParser.process_excel_data(
            train_file_names_to_process, self.__inputs, self.__outputs, arms=self.__arms)
        print("Training Datapoints: " + str(self.__train_datapoints))
        #data_training contains all the data with the timestamp
        self.__variable_names = self.__all_variable_names[1:]
        self.__data_train = self.__data_training[:, 1:]
        #data_train has all data without the timestamp. We store the timestamp in a separate array to merge later after standardization and PCA.
        self.__train_timestamps = self.__data_training[:, 0]
        print(self.__train_timestamps.shape)

    def __read_ValidData(self):
        if self.__file_names and len(self.__file_names) > 0:
            print("Using individual file names")
            data_dir = path.abspath(path.join(self.TOP_LEVEL_DIR, "data/OldData/"))
            valid_file_names_to_process = [path.join(data_dir, file) for file in self.__file_names]
        elif self.__valid_directory:
            print("Using directory")
            data_dir = path.abspath(path.join(self.TOP_LEVEL_DIR, self.__valid_directory))
            files = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith('xlsx')]
            valid_file_names_to_process = [path.join(data_dir, file) for file in files]

        self.__valid_files = [os.path.basename(file) for file in valid_file_names_to_process]
        print(self.__valid_files)
        
        self.__data_validation, self.__all_variable_names, self.__valid_datapoints = DvrkKinematicParser.process_excel_data(
            valid_file_names_to_process, self.__inputs, self.__outputs, arms=self.__arms)
        print("Validation Datapoints: " + str(self.__valid_datapoints))
        #data_training contains all the data with the timestamp
        self.__variable_names = self.__all_variable_names[1:]
        self.__data_valid = self.__data_validation[:, 1:]
        #data_train has all data without the timestamp. We store the timestamp in a separate array to merge later after standardization and PCA.
        self.__valid_timestamps = self.__data_validation[:, 0]

    def __mergeAllData(self):
        self.__data_all = np.vstack((self.__data_train, self.__data_valid))
        self.__all_timestamps = np.concatenate((self.__train_timestamps, self.__valid_timestamps))
        print("All Datapoints: " + str(len(self.__data_all)))

    def __standardize(self):

        self.__scaler = StandardScaler()
        self.__data_rescaled = self.__scaler.fit_transform(self.__data_all)

    def __splitData(self):

        output_vars = len([var for var in self.__variable_names if "ECM" in var])
        self.__scaler_means = self.__scaler.mean_[-output_vars:]
        self.__scaler_stds = self.__scaler.scale_[-output_vars:]
        self.__data_inputs = self.__data_rescaled[:, : -output_vars]
        self.__data_outputs = self.__data_rescaled[:, -output_vars:]

        print("Original Input Dimensions: " + str(len(self.__data_inputs[0])))
        print("Original Output Dimensions: " + str(len(self.__data_outputs[0])))

    def __PCA(self):

        self.__output_miu = np.mean(self.__data_outputs, axis=0)
        self.__pca_inputs = PCA(n_components = 0.98)
        self.__pca_outputs = PCA(n_components = 0.98)

        #scores here mean the projected data onto a lower dimensional space (such that 98% of the data is preserved) of the eigen vectors of the covariance matrix of the original data.
        self.__input_scores = self.__pca_inputs.fit_transform(self.__data_inputs)
        self.__PCA_input_vars = ["Input_PCA_" + str(i) for i in range(len(self.__input_scores[0]))]
        print("Input features have been reduced to: " + str(len(self.__input_scores[0])))
        self.__output_scores = self.__pca_outputs.fit_transform(self.__data_outputs)
        self.__PCA_output_vars = ["Output_PCA_" + str(i) for i in range(len(self.__output_scores[0]))]
        print("Output features have been reduced to: " + str(len(self.__output_scores[0])))

        self.__PCA_vars_training = [self.__all_variable_names[0]] + self.__PCA_input_vars + self.__PCA_output_vars
        print(self.__PCA_vars_training)

        self.__PCA_vars_train = self.__PCA_input_vars + self.__PCA_output_vars
        print(self.__PCA_vars_train)

    def __mergeData(self):

        self.__mergedData = np.concatenate((self.__input_scores, self.__output_scores), axis=1)
        self.__mergedData = np.concatenate((self.__all_timestamps.reshape(-1, 1), self.__mergedData), axis=1)
        #Now the mergedData also has timestamps. This is needed as the data will be shuffled after this.

    def __shuffleData(self):
        
        # if self.__validation_split <= 0 or self.__validation_split >= 100:
        #     print("Training on all data.")
        #     return
        
        # step = math.floor((self.__validation_split/100) * len(self.__mergedData))
        # upper_bound = math.floor((1 - (self.__validation_split/100))*len(self.__mergedData))
        # starting_point = random.randint(0, upper_bound)
        
        # rng = np.random.default_rng()
        # shuffled_indices = rng.permutation(len(self.__mergedData))
        # split_index = int(np.floor(len(self.__mergedData) * self.__validation_split / 100))
        # train_indices = shuffled_indices[split_index:]
        # validate_indices = np.sort(shuffled_indices[:split_index])

        self.__train_data = self.__mergedData[:self.__train_datapoints, :]
        self.__validate_data = self.__mergedData[self.__train_datapoints:, :]
        # self.__train_data = self.__mergedData[train_indices]
        #Training data has timestamps removed before passing it onto model training in GMM.
        self.__training_data = self.__train_data[:, 1:]
        self.__data_train_dpts = len(self.__train_data)

        print("Training Data Shape:")
        print(self.__train_data.shape)
        print("Validation Data Shape:")
        print(self.__validate_data.shape)

    def __createDir(self):

        filename = self.__valid_files
        text = os.path.splitext(filename[0])[0]
        text_arr = text.split('_')
        print(text_arr)
        self.filename = text_arr[0] + '_' + text_arr[1]
        
        self.outdir = path.join(path.dirname(__file__), "Sur12_Task16_Results/" + self.filename)
        if not path.exists(self.outdir):
            mkdir(self.outdir)

    def __train(self):
        self.filepath = f"Sur12_Task16_Results/{self.filename}/{self.date}"
        print(self.filepath)
        print("Training Gaussian Mixture Models...")
        self.__gmm_list = [None] * self.__max_gaussians
        for n in range(1, self.__max_gaussians + 1):
            self.__gmm_list[n - 1] = GaussianMixture(n_components=n, init_params="kmeans", max_iter = 300, n_init=self.__num_init)
            self.__gmm_list[n - 1].fit(self.__training_data)
            print(Color.YELLOW + f"\t{n}" + Color.END + " Gaussian" + "s" * np.sign(n - 1) + Color.CYAN + f"\t Model Score: {self.__gmm_list[n-1].score(self.__training_data)}" + Color.END, end="\r")

    def __analyze(self):
        print("\nSelecting best model according to BIC...")
        bic_values = [model.bic(self.__training_data) for model in self.__gmm_list]
        aic_values = [model.aic(self.__training_data) for model in self.__gmm_list]

        index_of_best = np.argmin(bic_values)
        self.optimal_gmm = self.__gmm_list[index_of_best]
        self.__optimal_Gaussians = index_of_best + 1
        self.__model_score = self.optimal_gmm.score(self.__training_data)

        print("="*80)
        print(Color.YELLOW + "Results: " + Color.END)
        print("Best # of Gaussians: " + Color.DARKCYAN + str(index_of_best + 1) + Color.END)
        print("Best Model Score: " + Color.DARKCYAN + str(self.optimal_gmm.score(self.__training_data)) + Color.END)

        # Plot the BIC/AIC Evaluations
        self.__plot(bic_values, aic_values)


    def __plot(self, bic_values, aic_values):
        self.__plotter = Plotter(self.__max_gaussians, self.filename, suffix=self.date)
        self.__plotter.plot_bic(bic_values)
        self.__plotter.plot_aic(aic_values)


    def __saveModel(self):
        
        gmm_params = {
            "priors": self.optimal_gmm.weights_,
            "means": self.optimal_gmm.means_,
            "covariances": self.optimal_gmm.covariances_,
            "variableNames": self.__variable_names,
            "PCA_evs": self.__pca_outputs.components_,
            "PCA_means": self.__output_miu,
            "Std_Scaler_means": self.__scaler_means,
            "Std_scaler_std": self.__scaler_stds
        }

        savemat(f"Sur12_Task16_Results/{self.filename}/{self.date}/GMM_params_{self.date}.mat", gmm_params)

        gmm_summary = {
            "MaxGaussians": self.__max_gaussians,
            "OptimalGaussians": self.__optimal_Gaussians,
            "Arms": self.__arms,
            "TrainingInputs": self.__inputs,
            "ValidationSplit": self.__validation_split,
            "FileNames":  self.__valid_files,
            "ModelScore": self.__model_score,
            "DataPoints": self.__train_datapoints,
            "ReducedInputDim": self.__PCA_input_vars,
            "ReducedOutputDim": self.__PCA_output_vars
        }

        savemat(f"Sur12_Task16_Results/{self.filename}/{self.date}/GMM_summary_{self.date}.mat", gmm_summary)

        if self.__validation_split > 0 and self.__validation_split < 100:
            validation_data_path = f"Sur12_Task16_Results/{self.filename}/{self.date}/validation_{self.date}.xlsx"
            df = pd.DataFrame(self.__validate_data, columns=self.__PCA_vars_training)
            df.to_excel(validation_data_path)

        training_data_path = f"Sur12_Task16_Results/{self.filename}/{self.date}/training_{self.date}.xlsx"
        df = pd.DataFrame(self.__train_data, columns=self.__PCA_vars_training)
        df.to_excel(training_data_path)

        # train_data_path = f"Sur12_Task16_Results/{self.filename}/{self.date}/train_{self.date}.xlsx"
        # df = pd.DataFrame(self.__training_data, columns=self.__PCA_vars_train)
        # df.to_excel(train_data_path)
    # def __runGMR(self):
    #     filepath = self.filepath.replace('\\','/')
    #     eng = matlab.engine.start_matlab()
    #     self.__actual, self.__prediction, self.__nullColumns, self.__similarityScore = eng.gaussianMixtureRegression(filepath, self.date)


if __name__ == "__main__":

    program_name = "Gaussian Mixture Model Trainer"
    program_desc = "Trains a GMM on input data from the provided Excel spreadsheet."

    trainGMM = TrainGMM()