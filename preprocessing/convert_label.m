clear;close all;clc;

excel_file = 'data/VRSA-FR/SSQ_continues.xlsx';
output_folder = 'data/VRSA-FR/labels_continues';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

data_table = readtable(excel_file, 'PreserveVariableNames', true, 'ReadVariableNames', false);
data_table = varfun(@(x) regexprep(string(x), ',', '.'), data_table);
data_numeric = str2double(table2array(data_table(:, 2:end))); 

num_subjects = size(data_numeric, 1);
for subject_idx = 1:num_subjects
    subject_data = data_numeric(subject_idx, :); 
    
    output_file = fullfile(output_folder, sprintf('subj_%d.mat', subject_idx));
    save(output_file, 'subject_data');
    
    fprintf('Subject %d.\n label saved', subject_idx);
end

fprintf('All label saved to %s.\n', output_folder);