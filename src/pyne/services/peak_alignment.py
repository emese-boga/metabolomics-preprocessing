from logging import getLogger
from typing import List, Tuple

from numpy import array as np_array
from pyne.models.peak import Peak
from pyne.models.spectrum import Spectrum
from statsmodels.nonparametric.smoothers_lowess import lowess

LOGGER = getLogger(__name__)


def align_peaks(spectrum_list: List[Spectrum]) -> List[Peak]:
    """
    Performs the peak alignment steps.
    Steps:
        - A Spectrum with the highest number of peaks across samples is selected called Master Data (master_spectrum).
        - The rest of the spectrums that have peaks are selected and called Test Data (test_spectrums).
        - Peaks in Test Data are transformed:
            - Each peak from the Test Data is compared to peaks from the Master Data
            - If a Test Data peak m/z and RT values are within a pre-defined m/z (mz_adj_win) and RT (rt_adj_win) range from
                the Master Data Peak, then we perform transformation
            - By transformation, two new peaks get generated
            - The first peak will have RT value of (Master Data peak RT + Test Data peak RT) / 2
            - The second peak will have RT value of (Master Data peak RT - Test Data peak RT)
        - The LOESS regression algorithm is used to smooth the transformed peaks.
    Args
        spectrum_list (List[Spectrum]): The list of Spectrums and their peaks which will be aligned.
        return_transformed_flag (bool): Flag to indicate whether the transformed peaks should also be returned.
    Returns
        (List[Peak]): The list of aligned peaks.
    """
    LOGGER.info("Starting peak alignment.")
    master_spectrum = find_master_spectrum(spectrum_list=spectrum_list)
    test_spectrums = find_test_spectrums(
        spectrum_list=spectrum_list, master_spectrum=master_spectrum
    )
    transformed_peaks = transform_peaks(
        mz_adj_win=0.2,
        rt_adj_win=20,
        master_spectrum=master_spectrum,
        test_spectrums=test_spectrums,
    )
    smoothened_peaks = apply_loess_regression(transformed_peaks=transformed_peaks)
    smoothened_peaks = sorted(smoothened_peaks, key=lambda peak: (peak.retention_time))
    return smoothened_peaks


def find_master_spectrum(spectrum_list: List[Spectrum]) -> Spectrum:
    """
    Args
        spectrum_list (List[Spectrum]): The Spectrum list in which we need to find the Master Data (Spectrum with the highest number of peaks).
    Returns
        (Spectrum): The Spectrum which has the most peaks across its' samples.
    """
    master_spectrum = max(spectrum_list, key=lambda spectrum: (spectrum.peak_count,))
    LOGGER.info(
        f"Found master spectrum with peak count: {master_spectrum.peak_count if master_spectrum is not None else 0}."
    )
    return master_spectrum


def find_test_spectrums(
    spectrum_list: List[Spectrum], master_spectrum: Spectrum
) -> List[Spectrum]:
    """
    Args
        spectrum_list (List[Spectrum]): The spectrum list from which the Master Data will be filtered out from.
        master_spectrum (Spectrum): The spectrum with the highest number of peaks from all samples (Master Data).
    Returns
        (List[Spectrum]): The list of Spectrums that have peaks, except the master spectrum (Test Data).
    """
    test_spectrums = [
        spectrum
        for spectrum in spectrum_list
        if spectrum.peak_count > 0 and spectrum != master_spectrum
    ]
    LOGGER.info(f"Found {len(test_spectrums)} test spectrums.")
    return test_spectrums


def transform_peaks(
    mz_adj_win: float,
    rt_adj_win: float,
    master_spectrum: Spectrum,
    test_spectrums: List[Spectrum],
) -> List[Tuple[Peak, Peak]]:
    """
    Transforms the peaks from the test spectrums list.
    Args
        mz_adj_win (float): Restriction for the mz value of a Peak.
        rt_adj_win (float): Restriction for the retention_time value of a Peak.
        master_spectrum (Spectrum): The Spectrum with the highest number of peaks from all samples (Master Data).
        test_spectrums (List[Spectrum]): List of spectrums that have peaks, except the master spectrum (Test Data).
    Returns
        (List[Tuple[Peak, Peak]]): A list of all transformed Peak tuples.
    """
    transformed_peaks = list()
    LOGGER.info("Started peak transformation based on master spectrum peaks.")

    for peak in master_spectrum.peaks:
        transformed_peaks.extend(
            transform_spectrum_peaks(
                mz_adj_win=mz_adj_win,
                rt_adj_win=rt_adj_win,
                peak=peak,
                test_spectrums=test_spectrums,
            )
        )
    LOGGER.info("Finished peak transformation.")
    return transformed_peaks


def transform_spectrum_peaks(
    mz_adj_win: float,
    rt_adj_win: float,
    peak: Peak,
    test_spectrums: List[Spectrum],
) -> List[Tuple[Peak, Peak]]:
    """
    Transforms the Peaks from the test_spectrums list, using one Peak from the Spectrum with the highest number of peaks (Master Data).
    Args
        mz_adj_win (float): Restriction for the mz value of a Peak.
        rt_adj_win (float): Restriction for the retention_time value of a Peak.
        peak (Peak): A peak from the Master Data.
        test_spectrums (List[Spectrum]): List of spectrums that have peaks, except the master spectrum (Test Data).
    Returns
        (List[Tuple[Peak, Peak]]): A list of Peak tuples transformed according to the Peak from Master Data.
    """
    transformed_peaks = list()
    for spectrum in test_spectrums:
        for test_peak in spectrum.peaks:
            mz_diff = abs(peak.mz - test_peak.mz)
            rt_diff = abs(peak.retention_time - test_peak.retention_time)
            if mz_diff <= mz_adj_win and rt_diff <= rt_adj_win:
                m = Peak(
                    scan_id=peak.scan_id,
                    peak_index=peak.peak_index,
                    retention_time=(peak.retention_time + test_peak.retention_time) / 2,
                    intensity=peak.intensity,
                    mz=peak.mz,
                )
                t = Peak(
                    scan_id=test_peak.scan_id,
                    peak_index=test_peak.peak_index,
                    retention_time=peak.retention_time - test_peak.retention_time,
                    intensity=test_peak.intensity,
                    mz=test_peak.mz,
                )
                transformed_peaks.append((m, t))

    return transformed_peaks


def apply_loess_regression(transformed_peaks: List[Tuple[Peak, Peak]]) -> List[Peak]:
    """
    Applies the LOESS regression algorithm to the retention_time value of transformed peaks.
    Args
        transformed_peaks (List[Tuple[Peak, Peak]]): The list of peaks that the regression algorithm will be perfomed on.
    Returns
        (List[Peak]): The list of peaks with smoothed retention_time values.
    """
    LOGGER.info("Applying LOESS regression.")
    x_values = np_array(
        [peak_tuple[0].retention_time for peak_tuple in transformed_peaks]
    )
    y_values = np_array(
        [peak_tuple[1].retention_time for peak_tuple in transformed_peaks]
    )
    LOGGER.info(
        f"Got {len(x_values)} retention time for x axis and {len(y_values)} for y axis."
    )

    y_smoothened = lowess(x_values, y_values, frac=0.01, return_sorted=False)
    smoothened_peaks = list()

    for i, (_, test_peak) in enumerate(transformed_peaks):
        updated_test_peak = Peak(
            scan_id=test_peak.scan_id,
            peak_index=test_peak.peak_index,
            retention_time=float(y_smoothened[i]),
            intensity=test_peak.intensity,
            mz=test_peak.mz,
        )

        smoothened_peaks.append(updated_test_peak)

    LOGGER.info("Finished regression algorithm.")

    return smoothened_peaks
