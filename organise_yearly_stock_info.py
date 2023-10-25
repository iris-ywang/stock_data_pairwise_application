import pandas as pd
import os


def extract_ith_rows_from_csv(input_folder_path, output_folder_path, save=False):
    # Initialize a dictionary to store dataframes
    result_dataframes = {}

    # Loop through the files in the folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".csv"):
            ticker = filename.split("_")[0]
            file_path = os.path.join(input_folder_path, filename)
            try:
                # Read the CSV file into a pandas dataframe
                df = pd.read_csv(file_path, index_col=0)
                # Iterate through each row and create dataframes for each row
                for i in range(len(df)):
                    year = df.index[i][:4]
                    if year not in result_dataframes:
                        result_dataframes[year] = []
                    year_data = df.iloc[i]
                    year_data.name = ticker
                    result_dataframes[year].append(year_data)
            except Exception as e:
                print(f'Error reading {filename}: {str(e)}')

    # Create separate dataframes for each ith row
    result_dataframes = {i: pd.concat(rows, axis=1) for i, rows in result_dataframes.items()}

    if save:
        for i, df_year in result_dataframes.items():
            df_year.T.to_csv(output_folder_path + "/stock_data_" + i + ".csv") 

    return result_dataframes

if __name__ == "__main__":
    input_folder_path = os.getcwd() + "/company_data/"
    output_folder_path = os.getcwd() + "/stock_yearly_data/"

    result_dataframes = extract_ith_rows_from_csv(input_folder_path, output_folder_path, save=True)
