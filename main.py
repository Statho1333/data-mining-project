import argparse
import pandas as pd
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt




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

#Plots the dataframe given as input
def plotDframe(df):
    plt.scatter(df[0], df[1], alpha=0.5, marker='o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('scatter_plot.png')
    #plt.show()

def dfPreprocess(df):
    pass

def main():
    ds_name = parser()
    df = pd.read_csv(ds_name, on_bad_lines='skip', header=None)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    print(df.dtypes)
    print(df.columns)
    plotDframe(df)
    print(df)
    

if __name__ == "__main__":
    main()
