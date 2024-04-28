from enum import Flag
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class DvrkKinematicParser:
    """Parser for Excel files generated using the DVRK Kinematic Logger.
    
    Attributes:
        PSM1_pos_cols: Column names for PSM1 end effector position data.
        PSM2_pos_cols: Column names for PSM2 end effector position data.
        PSM3_pos_cols: Column names for PSM3 end effector position data.
        ecm_pos_cols: Column names for ECM end effector position data.
        PSM1_orient_cols: Column names for PSM1 orientation matrix elements.
        PSM2_orient_cols: Column names for PSM2 orientation matrix elements.
        PSM3_orient_cols: Column names for PSM3 orientation matrix elements.
        ecm_orient_cols: Column names for ECM orientation matrix elements.
        PSM1_quat_cols: Column names for PSM1 quaternion components.
        PSM2_quat_cols: Column names for PSM2 quaternion components.
        PSM3_quat_cols: Column names for PSM3 quaternion components.
        ecm_quat_cols: Column names for ECM quaternion components.
        eye_gaze_cols: Column names for (x, y) point of gaze for left and right eyes.
    """
    timestamp_cols = ['timestamp']
    PSM1_pos_cols = ['PSM1_ee_x', 'PSM1_ee_y', 'PSM1_ee_z']
    PSM2_pos_cols = ['PSM2_ee_x', 'PSM2_ee_y', 'PSM2_ee_z']
    PSM3_pos_cols = ['PSM3_ee_x', 'PSM3_ee_y', 'PSM3_ee_z']
    ecm_pos_cols = ['ECM_ee_x', 'ECM_ee_y', 'ECM_ee_z']
    
    PSM1_orient_cols = []
    PSM2_orient_cols = []
    PSM3_orient_cols = []
    ecm_orient_cols = []

    for i in range(1,4):
        for j in range(1, 4):
            PSM1_orient_cols.append(f'PSM1_Orientation_Matrix_[{i},{j}]')
            PSM2_orient_cols.append(f'PSM2_Orientation_Matrix_[{i},{j}]')
            PSM3_orient_cols.append(f'PSM3_Orientation_Matrix_[{i},{j}]')
            ecm_orient_cols.append(f'ECM_Orientation_Matrix_[{i},{j}]')
    
    PSM1_quat_cols = ['PSM1_q_w', 'PSM1_q_x', 'PSM1_q_y', 'PSM1_q_z']
    PSM2_quat_cols = ['PSM2_q_w', 'PSM2_q_x', 'PSM2_q_y', 'PSM2_q_z']
    PSM3_quat_cols = ['PSM3_q_w', 'PSM3_q_x', 'PSM3_q_y', 'PSM3_q_z']
    ecm_quat_cols = ['ECM_q_w', 'ECM_q_x', 'ECM_q_y', 'ECM_q_z']

    gazeLR_cols = ['gaze_pixels_xL', 'gaze_pixels_yL', 'gaze_pixels_xR', 'gaze_pixels_yR']
    gazeBPOG_cols = ['gaze_pixels_xBPOG', 'gaze_pixels_yBPOG'] 
    gaze_pupilDiam_cols = ['gaze_pixels_pupilDiamL', 'gaze_pixels_pupilDiamR']
    gaze_dist_cols = ['gaze_pixels_distL', 'gaze_pixels_distR']

    
    @classmethod
    def process_excel_data(cls, file_paths, inputs, outputs, arms=[1, 2, 3]):
        """Reads data from Excel spreadsheets and returns a matrix.
        
        Args:
            file_path (list of str): Paths to the Excel files to read.
        
        Returns:
            (np.ndarray): Matrix of extracted data, formed by stacking
                the data from all spreadsheets.
        
        """
        data = []

        # if len(set(arms)) != 2:
        #     raise ValueError("Number of Distinct Arms must be 2! ")

        #Handle arms
        arms = sorted(arms)

        if not set(arms).issubset({1, 2, 3}):
            raise ValueError("Arms must be 1, 2, or 3! ")

        time_stamp = cls.timestamp_cols
        arm1 = cls.PSM1_pos_cols
        arm1_o = cls.PSM1_orient_cols
        arm1_q = cls.PSM1_quat_cols
        
        arm2 = cls.PSM2_pos_cols
        arm2_o = cls.PSM2_orient_cols
        arm2_q = cls.PSM2_quat_cols

        arm3 = cls.PSM3_pos_cols
        arm3_o = cls.PSM3_orient_cols
        arm3_q = cls.PSM3_quat_cols

        # if arms[0] == 1:
        #     arm1 = cls.PSM1_pos_cols
        #     arm1_o = cls.PSM1_orient_cols
        #     arm1_q = cls.PSM1_quat_cols
        #     if arms[1] == 2:
        #         arm2 = cls.PSM2_pos_cols
        #         arm2_o = cls.PSM2_orient_cols
        #         arm2_q = cls.PSM2_quat_cols
        #     else:
        #         arm2 = cls.PSM3_pos_cols
        #         arm2_o = cls.PSM3_orient_cols
        #         arm2_q = cls.PSM3_quat_cols

        # elif arms[0] == 2:
        #     arm1 = cls.PSM2_pos_cols
        #     arm1_o = cls.PSM2_orient_cols
        #     arm1_q = cls.PSM2_quat_cols
        #     arm2 = cls.PSM3_pos_cols
        #     arm2_o = cls.PSM3_orient_cols
        #     arm2_q = cls.PSM3_quat_cols


        input_arm1 = False
        input_arm2 = False
        input_arm3 = False

        if 1 in arms:
            input_arm1 = True
        
        if 2 in arms:
            input_arm2 = True
        
        if 3 in arms:
            input_arm3 = True


        #handle Training Inputs

        inputPosition = False
        inputOrientation = False
        inputGazeLR = False
        inputBPOG = False
        inputDiam = False
        inputGazeDist = False

        if 1 in inputs:
            inputPosition = True
        
        if 2 in inputs:
            inputOrientation = True

        if 3 in inputs:
            inputGazeLR = True

        if 4 in inputs:
            inputBPOG = True

        if 5 in inputs:
            inputDiam = True

        if 6 in inputs:
            inputGazeDist = True

        #handle Training Outputs

        outputPosition = False
        outputOrientation = False

        if 1 in outputs:
            outputPosition = True

        if 2 in outputs:
            outputOrientation = True
        print(file_paths)
        for file_path in file_paths:
            print("Loading data from: " + file_path.split('/')[-1])
            excel_data = pd.read_excel(file_path, engine='openpyxl')
            variable_names = []

            data_timestamp = excel_data.loc[:, time_stamp]
            arm1_pos = excel_data.loc[:, arm1]
            arm2_pos = excel_data.loc[:, arm2]
            arm3_pos = excel_data.loc[:, arm3]

            ecm_pos = excel_data.loc[:, cls.ecm_pos_cols]

            valid_columns = set(excel_data.columns)
            orient_matrix_columns = set(arm1_o + arm2_o + arm3_o + cls.ecm_orient_cols)
            # quat_columns = set(arm1_q + arm2_q + cls.ecm_quat_cols)
            
            gazeLR_columns = set(cls.gazeLR_cols)
            gazeBPOG_columns = set(cls.gazeBPOG_cols)
            gaze_pupilDiam_columns = set(cls.gaze_pupilDiam_cols)
            gaze_dist_columns = set(cls.gaze_dist_cols)


            new_data = np.empty([len(arm1_pos), 1])
            new_data = np.c_[new_data, data_timestamp.values]
            variable_names = variable_names + time_stamp
            
            if orient_matrix_columns.issubset(valid_columns):
                arm1_orient = excel_data.loc[:, arm1_o]
                arm2_orient = excel_data.loc[:, arm2_o]
                arm3_orient = excel_data.loc[:, arm3_o]

                ecm_orient = excel_data.loc[:, cls.ecm_orient_cols]
                arm1_quat = R.from_matrix(np.reshape(arm1_orient.values, (-1, 3, 3))).as_quat()
                arm2_quat = R.from_matrix(np.reshape(arm2_orient.values, (-1, 3, 3))).as_quat()
                arm3_quat = R.from_matrix(np.reshape(arm3_orient.values, (-1, 3, 3))).as_quat()

                ecm_quat = R.from_matrix(np.reshape(ecm_orient.values, (-1, 3, 3))).as_quat()

            else:
                arm1_quat = R.from_quat(excel_data.loc[:, arm1_q].values).as_quat()
                arm2_quat = R.from_quat(excel_data.loc[:, arm2_q].values).as_quat()
                arm3_quat = R.from_quat(excel_data.loc[:, arm3_q].values).as_quat()

                ecm_quat = R.from_quat(excel_data.loc[:, cls.ecm_quat_cols].values).as_quat()

            if input_arm1:
                if inputPosition:
                    new_data = np.c_[new_data, arm1_pos.values]
                    variable_names = variable_names + arm1

                if inputOrientation:
                    new_data = np.c_[new_data, arm1_quat]
                    variable_names = variable_names + arm1_q

            if input_arm2:
                
                if inputPosition:
                    new_data = np.c_[new_data, arm2_pos.values]
                    variable_names = variable_names + arm2

                if inputOrientation:
                    new_data = np.c_[new_data, arm2_quat]
                    variable_names = variable_names + arm2_q

            if input_arm3:

                if inputPosition:
                    new_data = np.c_[new_data, arm3_pos.values]
                    variable_names = variable_names + arm3

                if inputOrientation:
                    new_data = np.c_[new_data, arm3_quat]
                    variable_names = variable_names + arm3_q

            if inputGazeLR and gazeLR_columns.issubset(valid_columns):
                gaze_LR = excel_data.loc[:, cls.gazeLR_cols]
                new_data=np.c_[new_data, gaze_LR]
                variable_names = variable_names + cls.gazeLR_cols
            
            if inputBPOG and gazeBPOG_columns.issubset(valid_columns):
                gaze_BPOG = excel_data.loc[:, cls.gazeBPOG_cols]
                new_data = np.c_[new_data, gaze_BPOG]
                variable_names = variable_names + cls.gazeBPOG_cols

            if inputDiam and gaze_pupilDiam_columns.issubset(valid_columns):
                gaze_diam = excel_data.loc[:, cls.gaze_pupilDiam_cols]
                new_data = np.c_[new_data, gaze_diam]
                variable_names = variable_names + cls.gaze_pupilDiam_cols

            if inputGazeDist and gaze_dist_columns.issubset(valid_columns):
                gaze_dist = excel_data.loc[:, cls.gaze_dist_cols]
                new_data = np.c_[new_data, gaze_dist]
                variable_names = variable_names + cls.gaze_dist_cols

            new_data = np.c_[new_data, ecm_pos.values]
            variable_names = variable_names + cls.ecm_pos_cols

            if outputOrientation:
                new_data = np.c_[new_data, ecm_quat]
                variable_names = variable_names + cls.ecm_quat_cols

            new_data = np.delete(new_data, 0, 1)

            data.append(new_data)

        # Remove rows with NaN values
        data = np.vstack(data)
        data = data[~np.isnan(data).any(axis=1), :]
        datapoints = len(data)
        
        return data, variable_names, datapoints