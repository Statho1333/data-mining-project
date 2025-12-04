

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# --- Προϋπάρχουσες συναρτήσεις ---
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

def clean_dataset(df):
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# --- Νέα συνάρτηση k-means + outliers ---
def kmeans_outliers(df, k=5, threshold=95):
    """
    df: pandas DataFrame με στήλες 'x' και 'y'
    k: αριθμός clusters
    threshold: ποσοστό για outliers (π.χ. 95 → πάνω από 95ο ποσοστημόριο)
    """
    X = df[['x', 'y']].values

    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Υπολογισμός απόστασης κάθε σημείου από το centroid του cluster
    distances = np.linalg.norm(X - centroids[labels], axis=1)

    # Όριο για outliers
    cutoff = np.percentile(distances, threshold)
    outlier_mask = distances > cutoff
    outliers = df[outlier_mask]

    print(f"Detected {len(outliers)} outliers (>{threshold} percentile)")

    # Οπτικοποίηση
    plt.figure(figsize=(8,6))
    plt.scatter(df['x'], df['y'], c=labels, cmap='tab10', s=10, alpha=0.5)
    plt.scatter(outliers['x'], outliers['y'], c='red', s=20, label='Outliers')
    plt.scatter(centroids[:,0], centroids[:,1], c='black', s=50, marker='x', label='Centroids')
    plt.title("K-means clustering with outliers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    return outliers, centroids, labels

# --- main ---
def main():
    txt_files = ["dataseta.txt", "datasetb.txt"]
    all_dataframes = []

    for txt_file in txt_files:
        csv_file = txt_file.replace(".txt", ".csv")
        df = txt_to_csv(txt_file, csv_file)
        df = clean_dataset(df)

        print(f"\nAfter cleaning {csv_file}:")
        print(df.shape)
        print(df.head())
        print("-" * 40)

        # --- Κλήση k-means + outliers ---
        outliers, centroids, labels = kmeans_outliers(df, k=5, threshold=95)
        print(f"Outliers sample:\n{outliers.head()}")
        print("="*50)

        all_dataframes.append(df)

    print("\nAll datasets are cleaned and processed for clustering.")

if __name__ == "__main__":
    main()