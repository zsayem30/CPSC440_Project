clear; clc;

topLevelFolder = 'C:\Users\sayem\Documents\Intuitive Surgical Dataset\Matlab Processing\SurTask_Pairs_ALL_DATA';
files = dir(topLevelFolder);
disp(files);
dirFlags = [files.isdir];
all_files = files(~dirFlags);
SaveFolder = 'C:\Users\sayem\Documents\Intuitive Surgical Dataset\Matlab Processing\SurTask_Pairs_All_Data_Excel';

for i = 1:length(all_files)
    s = load(fullfile(topLevelFolder, all_files(i).name));
    [pathstr,name,ext] = fileparts(all_files(i).name);
    filename = strcat(name, ".xlsx");
    filepath = strcat(SaveFolder, "\", filename);
    data = s.dataTrain;
    data(any(isnan(data), 2), :) = [];
    data_cols = s.dataTrain_cols;
    data_cells = num2cell(data);
    output = [data_cols; data_cells];
    writecell(output, filepath);
end


