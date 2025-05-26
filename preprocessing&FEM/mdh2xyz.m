function mdh2xyz(folderPath)
    % Convert MDH parameters to XYZ coordinates and write back to CSV files
    % Input: folderPath - path to directory containing MDH parameter CSV files

    % Check if folder path exists
    if ~exist(folderPath, 'dir')
        error('Specified folder path does not exist.');
    end
    
    % Get all CSV files in the folder
    csvFiles = dir(fullfile(folderPath, '*.csv'));
    
    % Process each CSV file in the directory
    for i = 1:length(csvFiles)
        % Construct full file path
        csvFilePath = fullfile(folderPath, csvFiles(i).name);
        
        % Read CSV data with preserved headers
        tableData = readtable(csvFilePath, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
        
        % -------------------------------
        % Extract and prepare MDH parameters
        % -------------------------------
        % Expected column order: theta, d, a, alpha (angles in degrees)
        theta = tableData.theta;
        d = tableData.d;
        a = tableData.a;
        alpha = tableData.alpha;
        
        % Convert angular parameters to radians (MATLAB trigonometric functions use radians)
        theta_rad = deg2rad(theta);
        alpha_rad = deg2rad(alpha);
        
        % Combine parameters into MDH matrix [θ, d, a, α] in radians
        dh_params = [theta_rad, d, a, alpha_rad];
        
        % -------------------------------
        % Calculate joint positions
        % -------------------------------
        % Get homogeneous transformation matrices for all joints
        joint_positions = getJointPositions(dh_params);  % Assumes this returns 4x4xN matrix
        
        % Extract XYZ coordinates from transformation matrices
        xyz_data = zeros(size(joint_positions, 3), 3);  % Preallocate array
        for j = 1:size(joint_positions, 3)
            % Extract translation vector from homogeneous transformation matrix
            % Format: [X, Y, Z] = elements (1:3, 4) of 4x4 transformation matrix
            xyz_data(j, :) = joint_positions(1:3, 4, j);
        end
        
        % -------------------------------
        % Update and save data
        % -------------------------------
        % Display processing status
        disp(['Processing job ', num2str(i)]);
        disp(xyz_data);
        
        % Add XYZ coordinates to the table
        tableData.X_theo = xyz_data(:, 1);  % Theoretical X position
        tableData.Y_theo = xyz_data(:, 2);  % Theoretical Y position
        tableData.Z_theo = xyz_data(:, 3);  % Theoretical Z position
        
        % Write updated table back to original CSV file
        writetable(tableData, csvFilePath, 'Delimiter', ',');
    end
end