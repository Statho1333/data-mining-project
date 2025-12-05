import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# --- Μετατροπή TXT -> CSV ---
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
    img_name = f"{dataset_name}_kmeans_outliers.png"
    plt.savefig(img_name, dpi=300)
    print(f"[Saved plot → {img_name}]")

    plt.show()

    return outliers, centroids, labels


# --- main ---
def main():

    txt_files = ["dataseta.txt", "datasetb.txt"]

    for txt_file in txt_files:

        dataset_name = txt_file.replace(".txt", "")

        csv_file = dataset_name + ".csv"

        # TXT -> CSV
        df = txt_to_csv(txt_file, csv_file)

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


if __name__ == "__main__":
    main()