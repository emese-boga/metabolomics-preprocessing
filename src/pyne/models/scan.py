from typing import List
from uuid import uuid4

from numpy import array as np_array
from peakutils import indexes as peakutils_indexes
from peakutils import baseline as peakutils_baseline
from .peak import Peak
from scipy.ndimage import gaussian_filter1d


class Scan:
    """
    Represents a single scan in mass spectrometry data.
    Attributes:
        scan_id (str): A unique identifier for the scan.
        retention_time (float): Retention Time, indicating when the scan was taken.
        mz_array (List[float]): List of mass-to-charge ratios (m/z) detected in this scan.
        intensity_array (List[float]): List of intensities corresponding to each m/z value in mz_array.
    Note:
        Changes intensity_array attribute to type NDArray.
    """

    def __init__(
        self, retention_time: float, mz_array: List[float], intensity_array: List[float]
    ):
        self.scan_id = str(uuid4())
        self.retention_time = retention_time
        self.mz_array = mz_array
        self.intensity_array = np_array(intensity_array)

    def align_baseline(self, deg: int):
        """
        Aligns the baseline of the scan's intensity array.

        Args:
            deg (int): Degree of the polynomial for fitting the baseline.
        """
        baseline = peakutils_baseline(self.intensity_array.astype(float), deg=deg)
        self.intensity_array = self.intensity_array.astype(float) - baseline.astype(
            float
        )

    def filter_noise(self, sigma: float):
        """
        Applies Gaussian filtering to the scan's intensity array to reduce noise.

        Args:
            sigma (float): Standard deviation for Gaussian kernel, controls the degree of smoothing.
        """
        self.intensity_array = gaussian_filter1d(self.intensity_array, sigma)

    def get_peaks(self, thres: float, min_dist: int) -> List[Peak]:
        """
        Detects peaks in the scan's intensity array and stores them in the peaks attribute.

        Args:
            thres (float): The threshold relative to the maximum intensity for peak detection.
            min_dist (int): The minimum distance between each detected peak.
        """
        peak_indexes = peakutils_indexes(
            self.intensity_array, thres=thres, min_dist=min_dist, thres_abs=True
        ).tolist()
        peaks = []
        for index in peak_indexes:
            peaks.append(
                Peak(
                    scan_id=self.scan_id,
                    peak_index=index,
                    retention_time=self.retention_time,
                    intensity=float(self.intensity_array[index]),
                    mz=self.mz_array[index],
                )
            )
        return peaks

    def __repr__(self) -> str:
        return (
            f"Scan(retention_time={self.retention_time}, "
            f"observations={len(self.mz_array)}, "
            f"scan_id={self.scan_id})"
        )
