from pandas import DataFrame


def deconvolve_peaks(peaks_df: DataFrame) -> DataFrame:
    """
    Receives a DataFrame with peaks and their assigned label and returns the deconvolved (or filtered out) DataFrame.
    Steps
        - It groups the data together by the "label" column.
        - For each group, it identifies the highest "intensity" value (this will be the highest peak).
        - It saves the "intensity" and "RT" value of the highest peak, and calculates the average for the "mz" value for this group.
        - Finally it puts together the "intensity", "RT", and the calculated "mz" values.
    Args
        peaks_df (DataFrame): A pandas DataFrame with the following columns: "RT", "mz", "intensity", "label".
                              Each row of this DataFrame represents one peak, and their label.
                              Peaks with the same label, most likely represent the same compound.
    Returns
        (DataFrame): A pandas DataFrame with the deconvolved data, with the following columns: "RT", "mz", "intensity"
    """
    center_peaks_df = peaks_df.loc[peaks_df.groupby("label")["intensity"].idxmax()]

    average_mz = peaks_df.groupby("label")["m/z"].mean()
    center_peaks_df["m/z"] = center_peaks_df["label"].map(average_mz)
    center_peaks_df = center_peaks_df[["RT", "intensity", "m/z"]].reset_index(drop=True)
    return center_peaks_df
