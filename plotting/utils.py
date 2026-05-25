import os
import pandas as pd
import numpy as np
from tqdm import tqdm

###############
def load_excel_sheets(*data_dirs): 
    # 1. Handle the default if no paths are passed (Note: added comma to make it a tuple)
    if not data_dirs:
        data_dirs = ('data/MM/Phase_1',)

    dataframes = {}
    
    # 2. Loop through each directory provided in the arguments
    for current_dir in data_dirs:
        if not os.path.exists(current_dir):
            print(f"Warning: The directory '{current_dir}' does not exist. Skipping.")
            continue

        # 3. Find Excel files (ignoring temporary open files that start with ~)
        files = [f for f in os.listdir(current_dir) if f.endswith(".xlsx") and not f.startswith("~")]
        
        if not files:
            print(f"Warning: No Excel files found in '{current_dir}'.")
            continue

        # 4. Extract data and inject metadata
        for file_name in files:
            file_path = os.path.join(current_dir, file_name)
            xls = pd.ExcelFile(file_path)
            
            total_sheets = len(xls.sheet_names)
            file_base_name = os.path.splitext(file_name)[0]
            
            # Safely extract model name
            if "_" in file_base_name:
                model_name = file_base_name[:file_base_name.rfind("_")]
            else:
                model_name = file_base_name
                
            # Safely get the folder name 
            folder_tag = os.path.basename(os.path.normpath(current_dir))

            for n, sheet_name in enumerate(xls.sheet_names, 1):
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df.columns = df.columns.str.strip()
                
                # ---> THE UPGRADE: Inject metadata directly into the DataFrame <---
                df['Dir'] = folder_tag
                df['Model'] = model_name
                df['Sheet'] = sheet_name
                
                # Keep the key for the dictionary tracking
                key = f"{folder_tag}_{model_name}_{sheet_name}"
                dataframes[key] = df

                print(f"\r Loaded: {file_name} Sheets:[{n}/{total_sheets}]", end="", flush=True)
            print("") # Move to the next line after finishing a file
    
    return dataframes
#####################

def process_benchmark_stats(dfs):
    """
    Calculates mean, std dev, and percentage marks for benchmark DataFrames.
    Requires exactly 3 solution columns and an 'available marks' column.
    """
    for key, df in tqdm(dfs.items(), desc="Processing DataFrames"):
        # Normalize column names to lowercase for easier matching
        cols_lower = {col.lower(): col for col in df.columns}
        sol_cols = [cols_lower[c] for c in ['solution 1', 'solution 2', 'solution 3'] if c in cols_lower]
        marks_col = cols_lower.get('available marks')

        if len(sol_cols) != 3:
            print(f"Skipped {key}: Expected 3 solution columns, found {len(sol_cols)}.")
            continue

        # Give dfs knowledge of their own key
        df['key'] = key

        # Core Statistics
        df['mean solution'] = df[sol_cols].mean(axis=1)
        df['std solution'] = df[sol_cols].std(axis=1)

        # Percentage Calculation
        if marks_col:
            # Replace 0 with 1 to avoid DivisionByZero errors
            df['mean percentage mark'] = (df['mean solution'] / df[marks_col].replace(0, 1)) * 100
        else:
            print(f"Processed {key}: Missing marks column.")

        # STD Percentage Calculation
        if marks_col:
            # Replace 0 with 1 to avoid DivisionByZero errors
            df['std percentage'] = (df['std solution'] / df[marks_col].replace(0, 1)) * 100
        else:
            print(f"Processed {key}: Missing std column.")

    return dfs

###################################

# Global list of model names in chrocological order.
global_model_list = ['ChatGPT-4o','Gemini 1.5','ChatGPT-o1','Gemini 2.0','DeepSeek-V3','ChatGPT-o3','Gemini 2.5','ChatGPT-5.1','Gemini 3.0','DeepSeek-V3.2','Human'] # Chronological

# Order data
def order_master_df(master_df):
    # Check what models are in master df, and keep model_list order
    unique_models = master_df['Model'].unique()
    ordered_models = [model for model in global_model_list if model in unique_models]

    # Sort df by Dir, then Model, then Sheet
    master_df['Model'] = pd.Categorical(master_df['Model'], categories=ordered_models, ordered=True)
    master_df.sort_values(['Dir', 'Model', 'Sheet'], inplace=True)

    return master_df

####################################

def aggregate_stats(dfs):
    # 1. Create master df
    master_df = pd.concat(dfs.values(), ignore_index=True)

    # 2. Order the master df by our global lists
    master_df = order_master_df(master_df)

    # 3. Create aggregated summary sheet
    summary_stats = master_df.groupby(['Dir', 'Model', 'Sheet']).agg(
        mean_solution=('mean solution', 'sum'),
        Available_Marks=('Available Marks', 'sum'),
        Total_Std=('mean percentage mark', 'std'),
        N=('mean percentage mark', 'count')  # <--- This pulls the length (number of questions) for you!
    ).reset_index()

    # 4. Calculate Weighted Percentage
    summary_stats['Total_Mean%'] = (summary_stats['mean_solution'] / summary_stats['Available_Marks']) * 100

    # 5. Calculate the Std error: Std / sqrt(N)
    summary_stats['Total_Std_error%'] = summary_stats['Total_Std'] / np.sqrt(summary_stats['N'])

    # 6. Keep only what you need
    summary_stats = summary_stats[['Dir', 'Model', 'Sheet', 'Total_Mean%', 'Total_Std', 'Total_Std_error%']]

    return master_df, summary_stats

####################################





