function ybcR2mdh(folderPath)
    disp('Starting ybcR2mdh function')
    % Accepts folder path containing CSV files, converts ybcR parameters to MDH parameters, and writes back to original files
    
    % Check if folder path exists
    if ~exist(folderPath, 'dir')
        error('Specified folder path does not exist.');
    end
    
    % Get all CSV files matching pattern in folder
    csvFiles = dir(fullfile(folderPath, 'job*.csv'));
    % Check if any CSV files exist
    if isempty(csvFiles)
        error('No CSV files found in specified folder.');
    end
    disp(csvFiles);
    
    % Process each CSV file
    for i = 1:length(csvFiles)
        % Construct full file path
        csvFilePath = fullfile(folderPath, csvFiles(i).name);
        
        % Read CSV file with headers preservation
        tableData = readtable(csvFilePath, 'Delimiter', ',', 'VariableNamingRule', 'preserve');
        
        % Extract required columns
        y = tableData.y;
        b = tableData.b;
        c = tableData.c;
        R = tableData.R;
        
        % Calculate MDH parameters: [theta (joint angle), d (link offset), a (link length), 
        % alpha (link twist angle), offset]
        alpha = b;
        d = zeros(size(alpha));
        theta = c;
        
        % Special handling for first element a(1)
        % Using boundary condition since there's no theta(i-1)
        a(1) = y(1) + tan(deg2rad(theta(1)/2)) * R(1);
        
        % Calculate intermediate elements of a vector
        for j = 2:length(y)
            a(j) = y(j) + tan(deg2rad(theta(j-1)/2)) * R(j-1) + tan(deg2rad(theta(j)/2)) * R(j);
        end
        
        % Special handling for last element a(end) (commented out in original)
        % a(end) = y(end) + tan(deg2rad(theta(end-1)/2)) * R(end-1);
        disp(a);
        disp(d);
        
        % Add new columns to table
        tableData.alpha = alpha;
        tableData.a = a';
        tableData.d = d;
        tableData.theta = theta;
        
        % Write updated table back to original CSV
        writetable(tableData, csvFilePath, 'Delimiter', ',');
        
        disp(['Processed file: ', csvFiles(i).name]);
    end
end