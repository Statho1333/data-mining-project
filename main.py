import argparse
import pandas as pd
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#Τakes the first argument as the data path
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
def plotDframe(df, title):
    plt.figure()
    plt.scatter(df[:,0], df[:,1], alpha=0.5, marker='o')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.close()
    #plt.show()

#Plots the KMeans clustering results
def plotKmeanResults(df, results, title = "KMeans Clustering Results"):
    plt.figure()

    unique_labels = np.unique(results['labels'])
    for label in unique_labels:
        #create a mask for the current cluster eg [T F F T T ...]
        cluster_mask = results['labels'] == label
        #fetvch the points that belong to the current cluster
        cluster_points = df[cluster_mask & ~results['mask_outliers']] 
        #plot the cluster points
        plt.scatter(cluster_points[:,0], cluster_points[:,1], alpha=0.5, marker='o', label=f'Cluster {label}')
    #plot cluster centers
    plt.scatter(results['centers'][:,0], results['centers'][:,1], color='black', marker='X', s=10, label='Centers')
    #plot outliers
    outlier_points = df[results['mask_outliers']]
    plt.scatter(outlier_points[:,0], outlier_points[:,1], color='red', alpha=0.5, marker='x', label='Outliers')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.close()
        
       
    #plt.show()

#Simple preprocessing function that reads a csv file into a dataframe
#Converts all values to numeric, replacing non-numeric with 0.0
def dfSimplePreproccess(ds_name):
    df = pd.read_csv(ds_name, on_bad_lines='skip', header=None)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df


def calculateZscores(df):
    to_return =  (df - np.mean(df,axis=0)) / np.std(df, axis=0)
    return to_return

#Runs KMeans clustering on the given dataframe
#Returns labels, centers, distances from centers, outlier indices and mask for outliers
def runKMeans(df, k=5, outlier_percentile=99):
    kmeans = KMeans(n_clusters=k,random_state=42, n_init='auto', init='k-means++')
    labels = kmeans.fit_predict(df)
    centers = kmeans.cluster_centers_
    dists = np.linalg.norm(
        df - centers[labels], axis=1
    )
    threshold = np.percentile(dists, 99)
    mask_outliers = dists > threshold
    outliers_idx = np.where(mask_outliers)[0]
    return {
        "labels": labels,
        "centers": centers,
        "distances": dists,
        "outliers_idx": outliers_idx,
        "mask_outliers": mask_outliers
    }

def main():
    ds_name = parser()
    df = dfSimplePreproccess(ds_name)
    df = df.to_numpy()

    plotDframe(df, "Original Data Scatter Plot")

    dfZscores = calculateZscores(df)
    plotDframe(dfZscores,"Z-scores Scatter Plot")
    
    simpleKmeansResults = runKMeans(dfZscores, k=5, outlier_percentile=99)
    plotKmeanResults(dfZscores, simpleKmeansResults, title="KMeans Clustering with Outliers")

    

if __name__ == "__main__":
    main()
