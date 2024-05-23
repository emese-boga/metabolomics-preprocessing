from pathlib import Path
from typing import List

from pyne.models.spectrum import Spectrum
from pyne.services.peak_alignment import (
    align_peaks,
    apply_loess_regression,
    find_master_spectrum,
    find_test_spectrums,
    transform_peaks,
)
from pyne.services.peak_normalization import normalize_peaks
from pyne.services.preprocessor import preprocess_data
from pyne.services.spectrum_reader import read_spectra

EXPERIMENT_PATH = Path("/home/emeseboga/freelance/malaria_dataset")
FILE_LIST = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))


def align_baselines(spectrums: List[Spectrum]):
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)


def filter_noise(spectrums: List[Spectrum]):
    for spectrum in spectrums:
        spectrum.filter_noise(sigma=20)


def detect_peaks(spectrums: List[Spectrum]):
    for spectrum in spectrums:
        spectrum.detect_peaks(thres=7000000, min_dist=30)


def test_experiment_read(benchmark):
    benchmark(read_spectra, FILE_LIST)


def test_baseline_alignment(benchmark):
    spectrums = read_spectra(FILE_LIST)
    benchmark(align_baselines, spectrums)


def test_noise_filtering(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
    benchmark(filter_noise, spectrums)


def test_peak_detection(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
    benchmark(detect_peaks, spectrums)


def test_detect_master_spectrum(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    benchmark(find_master_spectrum, spectrums)


def test_detect_test_spectrums(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    master_spectrum = find_master_spectrum(spectrums)
    benchmark(find_test_spectrums, spectrums, master_spectrum)


def test_peak_transformation(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    master_spectrum = find_master_spectrum(spectrums)
    test_spectrums = find_test_spectrums(
        spectrum_list=spectrums, master_spectrum=master_spectrum
    )
    benchmark(
        transform_peaks,
        0.2,
        20,
        master_spectrum,
        test_spectrums,
    )


def test_loess_regression(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    master_spectrum = find_master_spectrum(spectrums)
    test_spectrums = find_test_spectrums(
        spectrum_list=spectrums, master_spectrum=master_spectrum
    )
    transformed_peaks = transform_peaks(
        mz_adj_win=0.2,
        rt_adj_win=20,
        master_spectrum=master_spectrum,
        test_spectrums=test_spectrums,
    )
    benchmark(apply_loess_regression, transformed_peaks)


def test_peak_alignment(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    benchmark(align_peaks, spectrums)


def test_normalization(benchmark):
    spectrums = read_spectra(FILE_LIST)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=30)

    aligned_peaks = align_peaks(spectrums)
    benchmark(normalize_peaks, aligned_peaks)


def test_preprocessing(benchmark):
    benchmark(preprocess_data, FILE_LIST)
