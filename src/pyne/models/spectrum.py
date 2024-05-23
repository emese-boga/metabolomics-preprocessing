from pandas import DataFrame
from .scan import Scan


class Spectrum:
    """
    Represents the complete set of readings obtained from the scan of a sample. Intuitively this is the same thing as
    a Sample in the API layer, however the internals here are different, thus the different name.

    Attributes:
        scans (List[Scan]): A list of Scan objects representing the individual scans in the spectrum.
        peaks (List[Peak]): A list of Peak objects representing all peaks found across scans in the spectrum.
        peak_count (int): The number of Peak objects found.
    """

    def __init__(self, df: DataFrame):
        """
        Initializes the Spectrum object from a pandas DataFrame.

        The DataFrame should have the following columns:
        - 'RT': Retention Time (float)
        - 'mz_array': List of mass-to-charge ratios (List[float])
        - 'intensity_array': List of intensities (List[float])

        Args:
            df (pd.DataFrame): DataFrame containing mass spectrometry scan data.
        """
        self.scans = [
            Scan(
                retention_time=row["RT"],
                mz_array=row["mz_array"],
                intensity_array=row["intensity_array"],
            )
            for _, row in df.iterrows()
        ]
        self.peaks = []
        self.peak_count = 0

    def align_baselines(self, deg: int):
        """
        Aligns the baselines of the intensity arrays of all scans in the spectrum.

        Args:
            deg (int): Degree of the polynomial for fitting the baseline.
        """
        for scan in self.scans:
            scan.align_baseline(deg)

    def filter_noise(self, sigma: float):
        """
        Applies Gaussian filtering to the intensity arrays of all scans to reduce noise.

        Args:
            sigma (float): Standard deviation for Gaussian kernel, controls the degree of smoothing.
        """
        for scan in self.scans:
            scan.filter_noise(sigma)

    def detect_peaks(self, thres: float, min_dist: int):
        """
        Applies peak detection to all scans in the spectrum and stores the detected peaks.

        Args:
            thres (float): The threshold relative to the maximum intensity for peak detection.
            min_dist (int): The minimum distance between each detected peak.
        """
        for scan in self.scans:
            scan_peaks = scan.get_peaks(thres, min_dist)
            self.peaks.extend(scan_peaks)

        self.peak_count = len(self.peaks)

    def __repr__(self) -> str:
        return f"SpectrumData(scans_count={len(self.scans)})"
