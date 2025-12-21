#Part 3 is implemented by Nefeli Tychala

import argparse
import pandas as pd
import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import time


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


def txt_to_csv(txt_file, csv_file):
    try:
        df = pd.read_csv(txt_file, delimiter=',', header=None, on_bad_lines='skip')
        df.columns = ['x', 'y']
        df.to_csv(csv_file, index=False)
        print(f"Converted {txt_file} → {csv_file} (bad lines skipped)")
        return df
    except Exception as e:
        print(f"Error converting {txt_file}: {e}")
        raise

# --- Καθαρισμός δεδομένων ---
def clean_dataset(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# --- K-means + Outliers ---
def kmeans_outliers(df, dataset_name, k=5, threshold=99.5):

    print(f"\nRunning K-Means for {dataset_name} with threshold={threshold}")

    X = df[['x', 'y']].values

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Distances of points from assigned centroid
    distances = np.linalg.norm(X - centroids[labels], axis=1)

    # Percentile cutoff for outliers
    cutoff = np.percentile(distances, threshold)
    outlier_mask = distances > cutoff
    outliers = df[outlier_mask]

    print(f"Detected {len(outliers)} outliers (>{threshold} percentile)")
    print(outliers.head())

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'], c=labels, cmap='tab10', s=10, alpha=0.5)
    plt.scatter(outliers['x'], outliers['y'], c='red', s=20, label='Outliers')
    plt.scatter(centroids[:,0], centroids[:,1], c='black', s=60, marker='x', label='Centroids')
    plt.title(f"K-means clustering with outliers — {dataset_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # ---- SAVE IMAGE ----
    img_dir = Path("images")
    img_dir.mkdir(parents=True, exist_ok=True)   # για σιγουριά

    safe_name = Path(dataset_name).name          # πετάει τα folders (π.χ. "202526files/xxx" -> "xxx")
    img_name = f"{safe_name}_kmeans_outliers.png"

    plt.savefig(img_dir / img_name, dpi=300, bbox_inches="tight")
    print(f"[Saved plot → {img_dir / img_name}]")

    plt.show()

    return outliers, centroids, labels
    
    
#part4 pipeline is implemented by Ioannis Stathopoulos
def step4_pipeline(ds_name: str):

    def load_and_scale_dataset(path, header=None):
    # load dataset
        dataset = pd.read_csv(path, header=header, on_bad_lines='skip')

        # convert to numeric όπου γίνεται, αλλιώς NaN
        dataset = dataset.apply(pd.to_numeric, errors='coerce')

        # drop NaN rows
        dataset = dataset.dropna()
        X = dataset.to_numpy()
        # scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, scaler, dataset
    
    def find_global_knn_outliers(X, k=7, pct=99.9):

        n = X.shape[0]

    
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)

        # 2) distances to the k+1 nearest (1ος = the point iself)
        dists, indices = nbrs.kneighbors(X)
        # dists.shape = (n, k+1)

        # 3) mean distance to the k nearest (excluding self)
        mean_dist = dists[:, 1:].mean(axis=1)   # shape: (n,)

        # 4) threshold: pct% if the points with mean_dist <= thr
        thr = np.percentile(mean_dist, pct)

        # 5) outliers = all those with mean_dist > thr
        outlier_mask = mean_dist > thr

        return outlier_mask, mean_dist, thr

    def find_best_k_silhouette(X, k_min=1, k_max=10, sample_size=10000, plot_elbow=True):

        wcss = []
        sil_scores = []
        ks_for_sil = []

        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(X)

            # Elbow metric
            wcss.append(kmeans.inertia_)

            # Silhouette only for k >= 2
            if k >= 2:
                sil = silhouette_score(
                    X,
                    kmeans.labels_,
                    sample_size=min(sample_size, X.shape[0]),
                    random_state=42
                )
                sil_scores.append(sil)
                ks_for_sil.append(k)

        # find best siluette
        if sil_scores:
            best_k_sil = ks_for_sil[int(np.argmax(sil_scores))]
        else:
            best_k_sil = None

        print('Best k with silhouette is:', best_k_sil)

        # Elbow plot
        if plot_elbow:
            plt.plot(range(k_min, k_max + 1), wcss)
            plt.title('Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            plt.show()

        return best_k_sil, wcss, sil_scores, ks_for_sil

    def run_kmeans(X, n_clusters, random_state=42):

        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            random_state=random_state
        )
        y_kmeans = kmeans.fit_predict(X)
        return kmeans, y_kmeans
    
    def find_cluster_knn_outliers(X, labels, k=7, pct=99.99):

        n = X.shape[0]
        outlier_mask = np.zeros(n, dtype=bool)

        for c in np.unique(labels):              # for each cluster
            idx = np.where(labels == c)[0]
            X_c = X[idx]

            # if cluster is too small, bypass
            if len(X_c) <= k:
                continue

            n_neighbors = min(k + 1, len(X_c))
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_c)
            dists, indices = nbrs.kneighbors(X_c)
            # dists.shape = (cluster_points, n_neighbors)

            # find mean of ngb without it self (χωρίς τον εαυτό)
            mean_dist = dists[:, 1:].mean(axis=1)   # shape: (m,)

            # threshold for pct of points to have mean_dist <= thr
            thr = np.percentile(mean_dist, pct)

            # mean_dist > thr points are too sparse -> outliers
            cluster_outliers = mean_dist > thr      # shape: (m,)

            # write outliers to total mask
            outlier_mask[idx] = cluster_outliers

        # return mask
        outliers = X[outlier_mask]
        return outlier_mask, outliers
    
    def plot_clusters_with_outliers(X, labels, kmeans, outliers, title='Clusters of data'):
        unique_clusters = np.unique(labels)

        plt.figure()

        # scatter each cluster
        for c in unique_clusters:
            plt.scatter(
                X[labels == c, 0],
                X[labels == c, 1],
                s=8,
                label=f'Cluster {c}'
            )

        # Centroids
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=40,
            c='black',
            label='Centroids'
        )

        # Outliers
        if outliers is not None and len(outliers) > 0:
            plt.scatter(
                outliers[:, 0],
                outliers[:, 1],
                c='red',
                marker='x',
                s=80,
                label='Outliers'
            )

        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    name = ds_name

    X, scaler, dataset = load_and_scale_dataset(name, header=None)

    outlier_mask, mean_dist, thr = find_global_knn_outliers(X, k=4, pct=95.0)
    print("Total points:", X.shape[0])
    print("Outliers found:", outlier_mask.sum())

    X_clean = X[~outlier_mask]
    dataset_clean = dataset[~outlier_mask]
    best_k_sil, wcss, sil_scores, ks_for_sil = find_best_k_silhouette(
        X,
        k_min=1,
        k_max=10,
        sample_size=10000,
        plot_elbow=True)
    
    kmeans, y_kmeans = run_kmeans(X, best_k_sil, random_state=42)

    outlier_mask, outliers = find_cluster_knn_outliers(
        X,
        y_kmeans,
        k=7,
        pct=99.99)
    
    plot_clusters_with_outliers(X, y_kmeans, kmeans, outliers, title='Clusters of data with k-NN outliers')
    

    

def main():
    ds_name = parser()

    #part 3
    t0 = time.perf_counter()
    



    dataset_name = ds_name.replace(".txt", "")

    csv_file = dataset_name + ".csv"

    # TXT -> CSV
    df = txt_to_csv(ds_name, csv_file)

    # Cleaning
    df = clean_dataset(df)

    print(f"\nAfter cleaning {csv_file}:")
    print(df.shape)
    print(df.head())
    print("-" * 40)

    # K-means + Outliers (threshold 99.5)
    outliers, centroids, labels = kmeans_outliers(
        df,
        dataset_name=dataset_name,
        k=5,
        threshold=99.5
    )
    print("=" * 60)
    print("\nAll datasets have been processed successfully.\n")
    t1 = time.perf_counter()
    print(f"Part 3 execution time: {t1 - t0:.2f} seconds")
    #end of part 3

    #part 4
    t0 = time.perf_counter()
    print(f'\nStarting Part 4 pipeline on dataset: {ds_name}\n')
    step4_pipeline(ds_name)
    t1 = time.perf_counter()
    print(f"Part 4 execution time: {t1 - t0:.2f} seconds")
    #end of part 4

    

if __name__ == "__main__":
    main()
