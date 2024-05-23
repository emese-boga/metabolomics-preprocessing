#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def plot_clustered_points(df):
    plt.scatter(
        df["Retention Time"],
        df["Intensity"],
        c=df["Cluster"],
        cmap="viridis",
        label="Data Points",
    )
    plt.title("Data Points with Clustering based on m/z")
    plt.xlabel("Retention Time")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()


def plot_cluster_curves_and_peaks(df, df_intensities):
    plt.figure(figsize=(8, 6))
    for cluster_id, cluster_data in df.groupby("Cluster"):
        print(cluster_id)
        if cluster_id != -1:
            plt.plot(
                cluster_data["Retention Time"],
                cluster_data["Intensity"],
                label=f"Cluster {cluster_id}",
            )

    plt.scatter(
        df_intensities["Retention Time"],
        df_intensities["Intensity"],
        marker="x",
        color="red",
        s=100,
        label="Max Intensity",
    )

    plt.title("Retention Time vs Intensity for Each Cluster")
    plt.xlabel("Retention Time")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()


def dbscan_clustering(df):
    X = StandardScaler().fit_transform(df[["m/z"]])
    dbscan = DBSCAN(eps=0.16, min_samples=2)
    return dbscan.fit_predict(X)


def deconvolve_data():
    mean1, std1 = 3, 1
    mean2, std2 = 4, 0.5
    mean3, std3 = 2, 1

    num_points = 100

    central_mz = 89.0

    x = np.linspace(0, 8, num_points)

    pdf1 = norm.pdf(x, mean1, std1)
    pdf2 = norm.pdf(x, mean2, std2)
    pdf3 = norm.pdf(x, mean3, std3)

    mz1 = np.random.normal(loc=central_mz, scale=0.5, size=num_points)
    mz2 = np.random.normal(loc=central_mz + 4, scale=0.5, size=num_points)

    bimodal_distribution = 0.6 * pdf1 + 0.4 * pdf2

    df1 = pd.DataFrame(
        {"Retention Time": x, "Intensity": bimodal_distribution, "m/z": mz1}
    )
    df2 = pd.DataFrame({"Retention Time": x, "Intensity": pdf3, "m/z": mz2})
    df = pd.concat([df1, df2], ignore_index=True)
    df.sort_values("Retention Time")

    df["Cluster"] = dbscan_clustering(df)
    df = df[df["Cluster"] != -1]
    df_peaks = df.loc[df.groupby("Cluster")["Intensity"].idxmax()]

    average_mz = df.groupby("Cluster")["m/z"].mean()
    df_peaks["m/z"] = df_peaks["Cluster"].map(average_mz)
    df_peaks = df_peaks[["Cluster", "Retention Time", "Intensity", "m/z"]].reset_index(
        drop=True
    )
    plot_cluster_curves_and_peaks(df, df_peaks)
    print(df_peaks)


if __name__ == "__main__":
    deconvolve_data()
