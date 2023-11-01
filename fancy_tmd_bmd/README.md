# Jupyter Notebook Cell Descriptions for TMD BMD Calculation

## Cell 1: Importing Required Libraries and Modules

- Import various Python libraries and modules for data manipulation and deep learning, including TensorFlow, OpenCV, NumPy, Pandas, Matplotlib, and more.

## Cell 2: Loading Pretrained Models and Configuration

- Change the working directory to a specified path.
- Load pre-trained models for head slice classification and masking.
- Configure image size, epochs, seed, and batch size for analysis.

## Cell 3: Analyzing and Generating CSV Files

- Defines a Python function, `analyse_folder`, to analyze a directory and create CSV files with file metadata.
- Handles 'died' mice files and regular BMP image files, excluding specific file types and folders.
- Generates CSV files and stores them in a subdirectory named 'csv_files_check_new_old'.

## Cell 4: Change Cohort Folder

- A directive to change the 'cohort_folder' variable before running the analysis.
- Modify 'cohort_folder' with the relevant directory path for the cohort analysis.

## Cell 5: Analyzing Data and Consolidating CSV Files

This cell performs the following actions:

- Calls the `analyse_folder(cohort_folder)` function to analyze data within the specified cohort folder.
- Prints "done" to indicate the completion of the analysis.
- Finds all CSV files within the 'csv_files_check_new_old' subdirectory and stores their paths in the `all_files` variable.
- Combines the data from all CSV files into a single DataFrame, `df_cohort`, by reading each file and concatenating them.
- Creates a consolidated CSV file named "Check_1278_1176_died.csv" containing the combined data.
- (The last line is commented out) Removes the temporary directory 'csv_files_check_new_old' to clean up, but it's commented to avoid accidental deletion.

## Cell 6: Reading and Sorting Consolidated Data 

This cell performs the following actions:

- Reads the consolidated CSV file "Check_1278_1176_died.csv" created in a previous cell.
- Sorts the data in the DataFrame `df_cohort` by the 'ID' column.
- Displays the sorted data, presenting information about the cohort in an organized format.
## Cell 7: Data Conversion and Filtering
This cell performs the following actions:

- Converts the 'IMAGE_WIDTH' and 'IMAGE_HEIGHT' columns in the DataFrame `df_cohort` to numeric data types.
- Filters the data to create a new DataFrame, `df_cohort_1176`, containing entries where the 'SCANNER' column is equal to '1176'.
- Displays the filtered data, providing information about the cohort specifically for the '1176' scanner.
## Cell 8: If you need to redo some items please change dataframe `df_cohort_1176` here

## Cell 9: Processing Data Through Multiple Functions To Compute TMD BMD

This cell processes data through a series of functions. It iterates over paths in the 'df_cohort_1176' DataFrame and performs the following steps for each path:

- **Part 1**: Runs a function to predict the head slice using 'predict_head_slice_1176' and measures the time taken for this step.
- **Part 2**: Executes a function to mask the body without the head using 'masking_predict_1176' and records the time taken.
- **Part 3**: Runs a function to mask the holder using 'holder_masking_predict' and measures the time taken.
- **Part 4**: Calculates Bone Mineral Density (BMD) and Tissue Mineral Density (TMD) using 'TMD_BMD_1176' and records the time taken. It also creates a 'BMD_TMD' directory if it doesn't exist.

Each part of the processing is timed, and the elapsed time is displayed.

The purpose of this cell is to perform various operations on the data for Scanner in a sequential manner.
