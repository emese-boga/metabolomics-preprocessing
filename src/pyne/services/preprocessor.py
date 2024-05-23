import logging
from typing import List

from pandas import DataFrame
from pyne.models.peak import Peak

from .deconvolution import deconvolve_peaks
from .peak_alignment import align_peaks
from .peak_clustering import apply_dbscan_clustering
from .peak_normalization import normalize_peaks
from .spectrum_reader import read_spectra

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def preprocess_data(source_files: List[str]) -> DataFrame:
    """
    Performs the preprocessing steps.
    Args:
        source_files (List[str]): A list of .csv files to perform data preprocessing on.
    Returns:
        DataFrame: A pandas DataFrame containing RT, mz, and intensity columns.
    """
    spectrum_list = read_spectra(source_files=source_files)
    for spectrum in spectrum_list:
        spectrum.align_baselines(deg=6)
        spectrum.filter_noise(sigma=0.7)
        spectrum.detect_peaks(thres=7000000, min_dist=60)

    LOGGER.info(
        "Finished baseline alignment, noise filtering and peak detection for each spectrum."
    )
    aligned_peaks = align_peaks(spectrum_list=spectrum_list)
    normalized_peaks = normalize_peaks(peaks=aligned_peaks)

    peak_df = retrieve_feature_matrix(normalized_peaks)
    clustered_peaks = apply_dbscan_clustering(peaks_df=peak_df)
    return deconvolve_peaks(peaks_df=clustered_peaks)


def retrieve_feature_matrix(peaks: List[Peak]) -> DataFrame:
    """
    Convert preprocessed data (peaks we get after the preprocessing steps) to a feature matrix.
    The feature matrix is in the form of a pandas Dataframe.

    Args:
        peaks (List[Peak]): A list of peak objects which will transformed to a DataFrame.

    Returns:
        DataFrame: A pandas DataFrame containing RT, mz, and intensity columns.
    """
    matrix = {"RT": [], "mz": [], "intensity": []}

    for peak in peaks:
        matrix["RT"].append(peak.retention_time)
        matrix["mz"].append(peak.mz)
        matrix["intensity"].append(peak.intensity)

    return DataFrame(matrix)
