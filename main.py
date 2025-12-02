import argparse
import pandas as pd
import numpy as np
from pathlib import Path 





#Ptakes the first argument as the data path
#checks if the file exists and if it does so, returns the file path
#If wrong name doesnt exist or not given , throw error
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str, required=True, help='dataset name')
    args = parser.parse_args()
    
    file_path = args.ds_name
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Dataset '{file_path}' not found.")
    else:
        return file_path




def main():
    ds = parser()
    ds = pd.read_csv(ds)
    print(ds)
    

if __name__ == "__main__":
    main()
