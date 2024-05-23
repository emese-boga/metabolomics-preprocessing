class Peak:
    """
    Represents a single peak in a scan from mass spectrometry data.
    Attributes:
        scan_id (str): The ID of the scan the peak belongs to.
        peak_index (int): Index of the peak in the scan intensity_array and mz_array
        retention_time (float): Retention Time, indicating when the scan was taken.
        intensity (float): The intensity value of the given peak.
        mz (float): The mass-to-charge ratio (m/z) of a the given peak.
    """

    def __init__(
        self,
        scan_id: str,
        peak_index: int,
        retention_time: float,
        intensity: float,
        mz: float,
    ):
        self.scan_id = scan_id
        self.peak_index = peak_index
        self.retention_time = retention_time
        self.intensity = intensity
        self.mz = mz

    def __repr__(self) -> str:
        return f"Peak(scan_id= {self.scan_id}, RT={self.retention_time}, intensity={self.intensity}, mz={self.mz})"
