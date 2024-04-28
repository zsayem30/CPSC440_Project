%% Load data from spreadsheet

clear; clc;

%% Select Folder to load spreadsheets and data from
topLevelFolder = uigetdir;
disp(topLevelFolder);% or whatever, such as 'C:\Users\John\Documents\MATLAB\work'
% Get a list of all files and folders in this folder.
files = dir(topLevelFolder);
% Get a logical vector that tells which is a directory.
dirFlags = [files.isdir];
% Extract only those that are directories.
subFolders = files(dirFlags); % A structure with extra info.
% Get only the folder names into a cell array.
subFolderNames = {subFolders(3:end).name}; % Start at 3 to skip . and ..
all_results = strings(331, 41);


count = 1;
for i = 1:length(subFolderNames)
    
    subFolderDir = strcat(topLevelFolder, "\", subFolderNames{i});
    subFolderFiles = dir(subFolderDir);
    subFolderdirFlags = [subFolderFiles.isdir];
    subsubFolders = subFolderFiles(subFolderdirFlags);
    subsubFolderNames = {subsubFolders(3:end).name};

    for k = 1:length(subsubFolderNames)
        
        userFolder = getenv ( 'userprofile' );
        folder = strcat(topLevelFolder,"\", subFolderNames{i} ,"\", subsubFolderNames{k});
        excel_filepath = strcat(folder, "\validation_", subsubFolderNames{k}, ".xlsx");
        training_filepath = strcat(folder, "\training_", subsubFolderNames{k}, ".xlsx");
        model_filepath = strcat(folder, "\GMM_params_", subsubFolderNames{k}, ".mat");
        summary_filepath = strcat(folder, "\GMM_summary_", subsubFolderNames{k}, ".mat");

        validation_data = readtable(excel_filepath);
        training_data = readtable(training_filepath);
        trainedGMM = load(model_filepath);
        
        %% Write model summaries
        S = load(summary_filepath);

        array = [string(S.FileNames), string(S.MaxGaussians), string(S.OptimalGaussians), strcat(num2str(S.Arms(:,1))," ",num2str(S.Arms(:,2)), " ",num2str(S.Arms(:,3))), string(size(S.ReducedInputDim, 1)), string(size(S.ReducedOutputDim, 1)), string(S.ModelScore), string(S.DataPoints)];
        inputArray = {};
        training_array = {'Position ', 'Orientation ', 'LR_POG ', 'BPOG ', 'PupilDiam ', 'PupilDist '};
        disp(training_array(S.TrainingInputs(:)));
        inputArray = [inputArray, training_array(S.TrainingInputs(:))];
        Summary_Array = [array, strjoin(inputArray)];
        disp(Summary_Array);
        %% Load Trained GMM
      
        priors = trainedGMM.priors;
        muGMR = trainedGMM.means;
        sigmaGMR = permute(trainedGMM.covariances, [2 3 1]);
        PCA_EigenVectors = trainedGMM.PCA_evs;
        PCA_Means = trainedGMM.PCA_means;
        Std_Scaler_Means = trainedGMM.Std_Scaler_means;
        Std_Scaler_std = trainedGMM.Std_scaler_std;

        %% Load Training and validation data

        train_cols = training_data.Properties.VariableNames(3:end);
        dataTrain = table2array(training_data(:, 3:end));
        train_timestamps = table2array(training_data(:, 2));

        cols = validation_data.Properties.VariableNames(3:end);
        dataVal = table2array(validation_data(:, 3:end));
        val_timestamps = table2array(validation_data(:, 2));
        
        is_column_output = cellfun(@(x) any(strfind(x, 'Output')), cols);
        inputs = find(~is_column_output);
        outputs = find(is_column_output);

        %% Remove repeated entries in both training and validation data
        [~, ia] = unique(val_timestamps, "rows", "stable");
        dataVal = dataVal(ia, :);
        val_timestamps = val_timestamps(ia, :);

        [~, ib] = unique(train_timestamps, "rows", "stable");
        dataTrain = dataTrain(ib, :);
        train_timestamps = train_timestamps(ib, :);

        %% Compute GMR output for validation data

        inputGMR_Validation = dataVal(:,inputs);
        [yQuery_Validation, ySigma_Validation] = GMR(priors, muGMR.', sigmaGMR, inputGMR_Validation.', inputs, outputs);
        yQuery_Validation = yQuery_Validation.';
%         ySigma_Validation = permute(ySigma_Validation, [3 1 2]);

        %% Compute GMR output for training data

        inputGMR_Training = dataTrain(:,inputs);
        [yQuery_Training, ySigma_Training] = GMR(priors, muGMR.', sigmaGMR, inputGMR_Training.', inputs, outputs);
        yQuery_Training = yQuery_Training.';
%         ySigma_Training = permute(ySigma_Training, [3 1 2]);
        
%         %% Compute similarity and cost of imitation for points
%         similarity_cost = immitation_cost(yQuery_Validation, dataVal(:, outputs), ySigma_Validation);

        %% Remove NaN rows from yQuery and also from Actual data
        [rows, columns] = find(isnan(yQuery_Validation));
        disp(size(yQuery_Validation, 1));
        disp(length(unique(rows)));

        yQuery_Validation(unique(rows), :) = [];
        dataVal(unique(rows), :) = [];
        val_timestamps(unique(rows), :) = [];

        [rows, column] = find(isnan(yQuery_Training));
        disp(size(yQuery_Training, 1));
        disp(length(unique(rows)));

        yQuery_Training(unique(rows), :) = [];
        dataTrain(unique(rows), :) = [];
        train_timestamps(unique(rows), :) = [];

        %% Revert PCAs for validating outputs
        validation_output = dataVal(:, outputs) * PCA_EigenVectors + PCA_Means;
        train_output = dataTrain(:, outputs) * PCA_EigenVectors + PCA_Means;

        predicted_validation_output = yQuery_Validation * PCA_EigenVectors + PCA_Means;
        predicted_training_output = yQuery_Training * PCA_EigenVectors + PCA_Means;

        %% Revert the standardization
        validation_output = Std_Scaler_std .* validation_output + Std_Scaler_Means;
        train_output = Std_Scaler_std .* train_output + Std_Scaler_Means;

        predicted_validation_output = Std_Scaler_std .* predicted_validation_output + Std_Scaler_Means;
        predicted_training_output = Std_Scaler_std .* predicted_training_output + Std_Scaler_Means;
        
        %% Add timesteps back

        validation_output = [val_timestamps validation_output];
        validation_output = sortrows(validation_output, 1);

        predicted_validation_output = [val_timestamps predicted_validation_output];
        predicted_validation_output = sortrows(predicted_validation_output, 1);

        train_output = [train_timestamps train_output];
        train_output = sortrows(train_output, 1);

        predicted_training_output = [train_timestamps predicted_training_output];   
        predicted_training_output = sortrows(predicted_training_output, 1);

        %% Compute errors between predicted validation output and actual validation output
        [avg_val_pos_err, std_val_pos_err, val_euclidean_errs, avg_val_euclidean_err, std_val_euclidean_err, val_orient_errs, avg_val_orient_err, std_val_orient_err] = calculate_errors(validation_output, predicted_validation_output);
        val_euclidean_errors = cell2mat(val_euclidean_errs);
        val_euclidean_errors = (val_euclidean_errors.')*1000;
        val_orientation_errors = cell2mat(val_orient_errs);
        val_orientation_errors = val_orientation_errors.';

        Prediction_Errs = [validation_output(:, 1), val_euclidean_errors, val_orientation_errors];
        actual_Prediction_Header = {'Timestamp', 'Val Euclidean Error (mm)', 'Val Orientation Error (Deg)'};

        actual_Prediction_filepath = strcat(folder, strcat('/GMR_Prediction_Errors_', subsubFolderNames{k}, '.xlsx'));
        xlswrite(actual_Prediction_filepath, actual_Prediction_Header, 1);
        xlswrite(actual_Prediction_filepath, Prediction_Errs, 1, 'A2');
        %% Compute errors between predicted training output and actual training output
        %% Comment out starts
        [avg_training_pos_err, std_training_pos_err, training_euclidean_errs, avg_training_euclidean_err, std_training_euclidean_err, training_orient_errs, avg_training_orient_err, std_training_orient_err] = calculate_errors(train_output, predicted_training_output);
        
        %% Form complete actual data
        all_actual_data = [train_output; validation_output];
        all_actual_data = sortrows(all_actual_data, 1);

        [C,ia] = unique(all_actual_data(:, 1), "rows", "stable");
        all_actual_data = all_actual_data(ia, :);

        %% Form complete predicted data
        all_predicted_data = [predicted_training_output; predicted_validation_output];        
        all_predicted_data = sortrows(all_predicted_data, 1);

        [C,ia] = unique(all_predicted_data(:, 1), "rows", "stable");
        all_predicted_data = all_predicted_data(ia, :);

        %% Compute errors between all the actual data and all the predicted data
        [avg_pos_err, std_pos_err, euclidean_errs, avg_euclidean_err, std_euclidean_err, orient_errs, avg_orient_err, std_orient_err] = calculate_errors(all_actual_data, all_predicted_data);
        
        %% Calculate percentage errors
        [mean_percentage_error, std_percentage_error] = calculate_percent_errors(all_actual_data, validation_output, val_euclidean_errs);


%         %% Depth and magification bounds
%         max_depth_Distance = 0.25;
%         min_depth_Distance = 0.20;
%         mid_depth_Distance = (max_depth_Distance + min_depth_Distance)/2;
% 
%         max_magnification_Factor = 15;
%         min_magnification_Factor = 10;
%         mid_magnification_Factor = (max_magnification_Factor + min_magnification_Factor)/2;
% 
%         %% Compute Geometry distance for 22.5 cm depth and 12.5 magnification
%         [mid_geom_dist, mid_geom_dist_avg, mid_geom_dist_std, mid_Area_Percent, mid_Width_Percent, mid_Height_Percent] = geometry_distance(validation_output, predicted_validation_output, mid_depth_Distance, mid_magnification_Factor); 
%         disp(mid_Area_Percent);
%         disp(mid_Width_Percent);
%         disp(mid_Height_Percent);
%         %% Compute Geometry distance for 20.0 cm depth and 10 magnification
%         [min_min_geom_dist, min_min_geom_dist_avg, min_min_geom_dist_std, min_min_Area_Percent, min_min_Width_Percent, min_min_Height_Percent] = geometry_distance(validation_output, predicted_validation_output, min_depth_Distance, min_magnification_Factor); 
%         %% Compute Geometry distance for 20.0 cm depth and 15 magnification
%         [min_max_geom_dist, min_max_geom_dist_avg, min_max_geom_dist_std, min_max_Area_Percent, min_max_Width_Percent, min_max_Height_Percent] = geometry_distance(validation_output, predicted_validation_output, min_depth_Distance, max_magnification_Factor); 
%         %% Compute Geometry distance for 25.0 cm depth and 10 magnification
%         [max_min_geom_dist, max_min_geom_dist_avg, max_min_geom_dist_std, max_min_Area_Percent, max_min_Width_Percent, max_min_Height_Percent] = geometry_distance(validation_output, predicted_validation_output, max_depth_Distance, min_magnification_Factor); 
%         %% Compute Geometry distance for 25.0 cm depth and 15 magnification
%         [max_max_geom_dist, max_max_geom_dist_avg, max_max_geom_dist_std, max_max_Area_Percent, max_max_Width_Percent, max_max_Height_Percent] = geometry_distance(validation_output, predicted_validation_output, max_depth_Distance, max_magnification_Factor);

%         %% Add all results to summary
%         Summary_Array = [Summary_Array, ...
%             avg_val_pos_err, std_val_pos_err, avg_val_euclidean_err, std_val_euclidean_err, avg_val_orient_err, std_val_orient_err, ... 
%             mid_geom_dist_avg, mid_geom_dist_std, mid_Area_Percent, mid_Width_Percent, mid_Height_Percent, ... 
%             min_min_geom_dist_avg, min_min_geom_dist_std, min_min_Area_Percent, min_min_Width_Percent, min_min_Height_Percent, ...
%             min_max_geom_dist_avg, min_max_geom_dist_std, min_max_Area_Percent, min_max_Width_Percent, min_max_Height_Percent, ...
%             max_min_geom_dist_avg, max_min_geom_dist_std, max_min_Area_Percent, max_min_Width_Percent, max_min_Height_Percent, ...
%             max_max_geom_dist_avg, max_max_geom_dist_std, max_max_Area_Percent, max_max_Width_Percent, max_max_Height_Percent, ...
%             mean_percentage_error, std_percentage_error, ...
%             avg_training_pos_err, std_training_pos_err, avg_training_euclidean_err, std_training_euclidean_err, avg_training_orient_err, std_training_orient_err, ...
%             avg_pos_err, std_pos_err, avg_euclidean_err, std_euclidean_err, avg_orient_err, std_orient_err];
        %% Add all results to summary
        Summary_Array = [Summary_Array, ...
            avg_val_pos_err, std_val_pos_err, avg_val_euclidean_err, std_val_euclidean_err, avg_val_orient_err, std_val_orient_err, ... 
            mean_percentage_error, std_percentage_error, ...
            avg_training_pos_err, std_training_pos_err, avg_training_euclidean_err, std_training_euclidean_err, avg_training_orient_err, std_training_orient_err, ...
            avg_pos_err, std_pos_err, avg_euclidean_err, std_euclidean_err, avg_orient_err, std_orient_err];

%         % Plot geometric distance error graphs with standard deviations
%         mid_geom_dist = cell2mat(mid_geom_dist);
%         geom_dist = geom_dist;
% 
%         x_axis = 1:size(mid_geom_dist);
%         f3 = figure('visible','off');
%         error_plot = plot(mid_geom_dist);
%         hold on;
%         std_plus_error = plot(mid_geom_dist+mid_geom_dist_std,'--');
%         hold on;
%         std_minus_error = plot(mid_geom_dist-mid_geom_dist_std,'--');
% 
%         legend('Geometric Error (m)','Geometric Error (m) + Std Dev', 'Geometric Error (m) - Std Dev')
%         title('Geometric Error (m) with Standard Deviation')
%         saveas(f3, strcat(folder, strcat('/Error_Plot_', subsubFolderNames{k}, '.png')));

%% Comment out ends
        %% Plot the validation output and the predicted validation outputs
%         f4 = figure('pos', [77,61,765,700], 'visible','off');
%         hold on; grid on;
%         
%         val_dots = plot3(validation_output(:, 2), validation_output(:, 3), validation_output(:, 4), ...
%             '.', 'linew', 2, 'DisplayName', 'ECM Position');
%         val_dots.Color = [0 0.8 0 0.1];
%         ecm_dots = plot3(predicted_validation_output(:, 2), predicted_validation_output(:, 3), predicted_validation_output(:, 4), '.', 'linew', 2, 'DisplayName', 'GMR Output');
%         ecm_dots.Color = [0 0 0 0.1];
%         
%         ave_error_text = strcat('err=', num2str(avg_val_euclidean_err));
% 
%         std_error_text = strcat('stderr=', num2str(std_val_euclidean_err));
%         
%         text(-0.06,1.05,{'Average Position Error (mm):', ave_error_text},'Units','normalized');
%         text(-0.06,0.97,{'Standard Deviation of Position Error (mm):', std_error_text},'Units','normalized');
%         text(-0.06,0.9,{'Average Orientation Error (degrees):', num2str(avg_val_orient_err)},'Units','normalized');
%         text(-0.06,0.83,{'Standard Deviation of Orientation Error (degrees):', num2str(std_val_orient_err)},'Units','normalized');
%         
%         % Set x, y, and z limits to show all data at a reasonable scale
%         min_vals = min(min(predicted_validation_output(:, 2:4), validation_output(:, 2:4)), [], 1);
%         max_vals = max(max(predicted_validation_output(:, 2:4), validation_output(:, 2:4)), [], 1);
%         offset = 0.001;
%         xlim([min_vals(1) - offset, max_vals(1) + offset]);
%         ylim([min_vals(2) - offset, max_vals(2) + offset]);
%         zlim([min_vals(3) - offset, max_vals(3) + offset]);
%         
%         xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
%         title({'Validation of GMR Output', strcat(num2str(length(priors)), " ", 'Gaussians'), strcat(num2str(size(muGMR, 2)), " ", 'Dimensions (input + output)')});
%         legend;
%         view([45 45]);
%         saveas(f4, strcat(folder, strcat('/GMR_validation_', subsubFolderNames{k}, '.png')));
%         saveas(f4, strcat(folder, strcat('/GMR_validation_', subsubFolderNames{k}, '.fig')));
%         hold off; grid off;

        %% Plot all the actual data output and the predicted data output
%         f5 = figure('pos', [77,61,765,700], 'visible','off');
%         hold on; grid on;
%         act_plot = plot3(all_actual_data(:, 2), all_actual_data(:, 3), all_actual_data(:, 4), 'DisplayName', 'ECM Actual Position');
%         act_plot.Color = [0 0.8 0 0.1];
%         d_act_pos = diff(all_actual_data(:, 2:4));
%         quiver3(all_actual_data(1:end-1, 2), all_actual_data(1:end-1, 3), all_actual_data(1:end-1, 4), d_act_pos(:, 1), d_act_pos(:, 2), d_act_pos(:, 3));
%         val_plot = plot3(all_predicted_data(:, 2), all_predicted_data(:, 3), all_predicted_data(:, 4), 'DisplayName', 'ECM Predicted Position');
%         val_plot.Color = [0 0 0 0.1];
%         d_pred_pos = diff(all_predicted_data(:, 2:4));
%         quiver3(all_predicted_data(1:end-1, 2), all_predicted_data(1:end-1, 3), all_predicted_data(1:end-1, 4), d_pred_pos(:, 1), d_pred_pos(:, 2), d_pred_pos(:, 3));
%         
%         ave_error_text = strcat('err=', num2str(avg_euclidean_err));
% 
%         std_error_text = strcat('stderr=', num2str(std_euclidean_err));
%         
%         text(-0.06,1.05,{'Average Position Error (mm):', ave_error_text},'Units','normalized');
%         text(-0.06,0.97,{'Standard Deviation of Position Error (mm):', std_error_text},'Units','normalized');
%         text(-0.06,0.9,{'Average Orientation Error (degrees):', num2str(avg_orient_err)},'Units','normalized');
%         text(-0.06,0.83,{'Standard Deviation of Orientation Error (degrees):', num2str(std_orient_err)},'Units','normalized');
%         
%         % Set x, y, and z limits to show all data at a reasonable scale
%         min_vals = min(min(all_predicted_data(:, 2:4), all_actual_data(:, 2:4)), [], 1);
%         max_vals = max(max(all_predicted_data(:, 2:4), all_actual_data(:, 2:4)), [], 1);
%         offset = 0.001;
%         xlim([min_vals(1) - offset, max_vals(1) + offset]);
%         ylim([min_vals(2) - offset, max_vals(2) + offset]);
%         zlim([min_vals(3) - offset, max_vals(3) + offset]);
%         
%         xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
%         title({'Complete GMR Output', strcat(num2str(length(priors)), " ", 'Gaussians'), strcat(num2str(size(muGMR, 2)), " ", 'Dimensions (input + output)')});
%         legend;
%         view([45 45]);
%         saveas(f5, strcat(folder, strcat('/GMR_trj_', subsubFolderNames{k}, '.png')));
%         saveas(f5, strcat(folder, strcat('/GMR_trj_', subsubFolderNames{k}, '.fig')));
%         hold off; grid off;
        %%
        %% Comment Out starts
        all_results(count, :) = Summary_Array;
        count = count + 1;
        %% Comment Out Ends
    end
end

%%
% header = {'Filename', 'Max Gaussians', 'Optimal Gaussians', 'Arms', 'Reduced_Input_Dim', 'Reduced_Output_Dim' ,'Model Score', 'Data_Points', 'Training Inputs', ...
%           'Mean Val ABS Error X (mm)', 'Mean Val ABS Error Y (mm)', 'Mean Val ABS Error Z (mm)', 'Std Val Error X (mm)', 'Std Val Error Y (mm)', 'Std Val Error Z (mm)', ...
%           'Mean Val ABS Error (mm)', 'Mean Val ABS Std (mm)', ...
%           'Mean Val Orientation Error (deg)', 'Std Val Orientation Error (deg)', ...
%           "Mid Avg Geometric Distance (m)", "Mid Std Geometric Distance (m)", "Mid Area Percentage", "Mid Width Percentage", "Mid Height Percentage" ...
%           "Min Depth Min Mag Avg Geometric Distance (m)", "Min Depth Min Mag Std Geometric Distance (m)", "Min Depth Min Mag Area Percentage", "Min Depth Min Mag Width Percentage", "Min Depth Min Mag Height Percentage" ...
%           "Min Depth Max Mag Avg Geometric Distance (m)", "Min Depth Max Mag Std Geometric Distance (m)", "Min Depth Max Mag Area Percentage", "Min Depth Max Mag Width Percentage", "Min Depth Max Mag Height Percentage" ...
%           "Max Depth Min Mag Avg Geometric Distance (m)", "Max Depth Min Mag Std Geometric Distance (m)", "Max Depth Min Mag Area Percentage", "Max Depth Min Mag Width Percentage", "Max Depth Min Mag Height Percentage" ...
%           "Max Depth Max Mag Avg Geometric Distance (m)", "Max Depth Max Mag Std Geometric Distance (m)", "Max Depth Max Mag Area Percentage", "Max Depth Max Mag Width Percentage", "Max Depth Max Mag Height Percentage" ...
%           "Mean Percentage Error", "Std Percentage Error", ...
%           'Mean Training ABS Error X (mm)', 'Mean Training ABS Error Y (mm)', 'Mean Training ABS Error Z (mm)', 'Std Training Error X (mm)', 'Std Training Error Y (mm)', 'Std Training Error Z (mm)', ...
%           'Mean Training ABS Error (mm)', 'Mean Training ABS Std (mm)', ...
%           'Mean Training Orientation Error (deg)', 'Std Training Orientation Error (deg)', ...
%           'Mean All ABS Error X (mm)', 'Mean All ABS Error Y (mm)', 'Mean All ABS Error Z (mm)', 'Std All Error X (mm)', 'Std All Error Y (mm)', 'Std All Error Z (mm)', ...
%           'Mean All ABS Error (mm)', 'Mean All ABS Std (mm)', ...
%           'Mean All Orientation Error (deg)', 'Std All Orientation Error (deg)'};
header = {'Filename', 'Max Gaussians', 'Optimal Gaussians', 'Arms', 'Reduced_Input_Dim', 'Reduced_Output_Dim' ,'Model Score', 'Data_Points', 'Training Inputs', ...
          'Mean Val ABS Error X (mm)', 'Mean Val ABS Error Y (mm)', 'Mean Val ABS Error Z (mm)', 'Std Val Error X (mm)', 'Std Val Error Y (mm)', 'Std Val Error Z (mm)', ...
          'Mean Val ABS Error (mm)', 'Mean Val ABS Std (mm)', ...
          'Mean Val Orientation Error (deg)', 'Std Val Orientation Error (deg)', ...
          "Mean Percentage Error", "Std Percentage Error", ...
          'Mean Training ABS Error X (mm)', 'Mean Training ABS Error Y (mm)', 'Mean Training ABS Error Z (mm)', 'Std Training Error X (mm)', 'Std Training Error Y (mm)', 'Std Training Error Z (mm)', ...
          'Mean Training ABS Error (mm)', 'Mean Training ABS Std (mm)', ...
          'Mean Training Orientation Error (deg)', 'Std Training Orientation Error (deg)', ...
          'Mean All ABS Error X (mm)', 'Mean All ABS Error Y (mm)', 'Mean All ABS Error Z (mm)', 'Std All Error X (mm)', 'Std All Error Y (mm)', 'Std All Error Z (mm)', ...
          'Mean All ABS Error (mm)', 'Mean All ABS Std (mm)', ...
          'Mean All Orientation Error (deg)', 'Std All Orientation Error (deg)'};

output_filepath = strcat(topLevelFolder, "/", "compile_PCA_All_leaveOne_updated.xlsx");
xlswrite(output_filepath,header, 1);
xlswrite(output_filepath,all_results, 1, 'A2');


%%
