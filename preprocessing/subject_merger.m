clear; close all; clc;

data_folder = 'data/VRSA-FR/EEG';
interpolated_folder = 'data/VRSA-FR/interpolated_eeg';
output_folder = 'data/VRSA-FR/final_eeg';
labels_folder = 'data/VRSA-FR/labels';
group_splits_file = 'data/VRSA-FR/group_splits.xlsx';

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

group_splits = readtable(group_splits_file, 'VariableNamingRule', 'preserve');

num_scenes = 20;
num_subjects = 25;
scene_idx_list = 1:num_scenes;
missing_subjects_log = fopen(fullfile(output_folder, 'missing_subject.txt'), 'w');

fs = 500;
Wl = 4; Wh = 40; 
Wn = [Wl * 2, Wh * 2] / fs; 
[b, a] = cheby2(6, 60, Wn); 


for subj_idx = 1:num_subjects
    data = [];
    missing_scenes = [];
    labels = [];

    for scene_idx = scene_idx_list
        eeg_file = fullfile(data_folder, num2str(scene_idx), sprintf('subj_%d.mat',subj_idx));
        interpolated_file = fullfile(interpolated_folder, num2str(scene_idx), sprintf('subj_%d.mat', subj_idx));

        if isfile(eeg_file)
            loaded_data = load(eeg_file, 'final_array');
            mat_data = loaded_data.final_array;
        elseif isfile(interpolated_file)
            loaded_data = load(interpolated_file, 'final_array'); 
            mat_data = loaded_data.final_array;
            missing_scenes = [missing_scenes, scene_idx]; 
        else
            fprintf('Subject %d, Scene %d file is missing in both folders!', subj_idx, scene_idx);
             mat_data = zeros(80000, 29); 
        end

        if isempty(data)
            data = zeros(size(mat_data, 1), size(mat_data, 2), num_scenes);
        end
        data(:, :, scene_idx) = mat_data;
    end

    for scene_idx = 1:num_scenes
        data(:, :, scene_idx) = filtfilt(b, a, data(:, :, scene_idx)); 
    end

    labels_file = fullfile(labels_folder, sprintf('subj_%d.mat', subj_idx));
    if isfile(labels_file)
        loaded_labels = load(labels_file, 'subject_data'); 
        labels = loaded_labels.subject_data'; 
    else
        fprintf('Labels for Subject %d are missing!\n', subj_idx);
        labels = zeros(1, num_scenes);
    end

    if ~isempty(missing_scenes)
        fprintf(missing_subjects_log, 'Subject %d: %s\n', subj_idx, strjoin(string(missing_scenes), ', '));
    end
    
    group1_indices = str2num(group_splits.("Group 1 Indices"){subj_idx}); %#ok<ST2NM>
    group2_indices = str2num(group_splits.("Group 2 Indices"){subj_idx}); %#ok<ST2NM>
    
    training_data = data(:, :, group1_indices + 1);
    training_labels = labels(group1_indices + 1);
    
    evaluation_data = data(:, :, group2_indices + 1);
    evaluation_labels = labels(group2_indices + 1);

    data = training_data;
    label = training_labels;
    save(fullfile(output_folder, sprintf('subj_%dt.mat', subj_idx)), 'data', 'label');
    fprintf('Saved training data for subject %d\n', subj_idx);
    
    data = evaluation_data;
    label = evaluation_labels;
    save(fullfile(output_folder, sprintf('subj_%de.mat', subj_idx)), 'data', 'label');
    fprintf('Saved evaluation data for subject %d\n', subj_idx);
end

% Close the log file
fclose(missing_subjects_log);

disp('Data and labels merging, splitting, and saving completed.');