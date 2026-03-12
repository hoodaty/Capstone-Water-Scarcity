import pandas as pd
import os

def remove_columns(file_path: str, columns_to_drop: list) -> None:
    """
    Reads a CSV file, removes specified columns if present, 
    and saves the result.
    """
    # 1. Verify file existence
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found in the current directory.")
        return

    try:
        # 2. Load the dataset into a DataFrame
        df = pd.read_csv(file_path)

        # 3. Identify which target columns actually exist in the file
        valid_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

        if not valid_columns_to_drop:
            print(f"Info: Neither 'temperatures' nor 'precipitations' were found in '{file_path}'.")
            return

        # 4. Remove the columns
        df.drop(columns=valid_columns_to_drop, inplace=True)

        # 5. Overwrite the original file with the updated DataFrame
        df.to_csv(file_path, index=False)
        print(f"Success: Removed {valid_columns_to_drop} from '{file_path}'.")

    except Exception as e:
        print(f"An unexpected error occurred processing '{file_path}': {e}")


if __name__ == "__main__":
    # Define the target files and the exact column headers to remove
    target_files = [
        'C:\\Users\\DELL\\Capstone-Water-Scarcity\\data\\input\\dataset_train_aug_renamed_cmip6.csv',
        #'C:\\Users\\DELL\\Capstone-Water-Scarcity\\data\\evaluation\\dataset_eval_aug_renamed_cmip6.csv'
    ]
    columns_to_remove = ['temperatures', 'precipitations']

    # Process each file
    for file in target_files:
        remove_columns(file, columns_to_remove)