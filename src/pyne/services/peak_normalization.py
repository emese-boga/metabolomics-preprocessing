from logging import getLogger
from typing import List

from numpy import array as np_array
from pyne.models.peak import Peak
from sklearn.preprocessing import quantile_transform

LOGGER = getLogger(__name__)


def normalize_peaks(peaks: List[Peak]) -> List[Peak]:
    """
    Uses the quantile normalization technique to normalize the peaks' intensity values.
    Args:
        peaks (List[Peak]): A list of peaks that will be normalized.
    Returns:
        List[Peak]: A list of peaks that normalization was performed on.
    """
    x_values = np_array([peak.intensity for peak in peaks]).reshape(-1, 1)
    normalized_values = quantile_transform(x_values)
    normalized_values = np_array(normalized_values).flatten()
    normalized_peaks = []

    for i, peak in enumerate(peaks):
        updated_peak = Peak(
            scan_id=peak.scan_id,
            peak_index=peak.peak_index,
            retention_time=peak.retention_time,
            intensity=normalized_values[i],
            mz=peak.mz,
        )

        normalized_peaks.append(updated_peak)
    LOGGER.info("Finished normalization")
    return normalized_peaks
