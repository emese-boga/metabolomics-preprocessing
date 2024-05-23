from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


def apply_dbscan_clustering(peaks_df: DataFrame) -> DataFrame:
    """
    Applies the DBSCAN classification algorithm.
    Args
        peaks_df (DataFrame): A pandas DataFrame with peak data. Columns: "RT", "mz", "intensity"
                              Each row represents one peak.
    Returns
        (DataFrame): A pandas DataFrame with columns: "RT", "mz", "intensity", "label".
                     Each row represents one peak and its' corresponding label.
                     Peaks that were assigned the same label most likely belong to the same compound.
    """
    X = StandardScaler().fit_transform(peaks_df[["mz"]])
    dbscan = DBSCAN(eps=0.16, min_samples=2)

    peaks_df["label"] = dbscan.fit_predict(X)
    peaks_df = peaks_df[peaks_df["label"] != -1]
    return peaks_df
