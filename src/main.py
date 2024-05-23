#!/usr/bin/env python
import ast
import csv
import os
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from pyne.services.peak_alignment import (
    apply_loess_regression,
    find_master_spectrum,
    find_test_spectrums,
    transform_peaks,
)
from pyne.services.preprocessor import preprocess_data
from pyne.services.spectrum_reader import read_spectra
from pyteomics import mzml

current_directory = Path(os.path.abspath(os.getcwd()))
EXPERIMENT_PATH = current_directory / "src/data"


def convert_mzml_to_csv():
    for path in EXPERIMENT_PATH.glob("*.mzML"):
        entry_list = list()
        with mzml.read(str(path)) as reader:
            for scan in reader:
                entry = {
                    "RT": scan["scanList"]["scan"][0]["scan start time"],
                    "intarray": scan["intensity array"],
                    "mzarray": scan["m/z array"],
                }
                entry_list.append(entry)
        df = pd.DataFrame(entry_list, columns=["RT", "intarray", "mzarray"])
        file_name = path.name.split(".")[0]
        df.to_csv(f"{path.parent}/{file_name}.csv", index=False)


def plot_raw_csv():
    _, ax = plt.subplots()
    for path in EXPERIMENT_PATH.glob("*.csv"):
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            intensity_array = ast.literal_eval(row["intensity_array"])
            mz_array = ast.literal_eval(row["mz_array"])
            ax.plot(mz_array, intensity_array)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        break

    plt.show()


def plot_scan_functionalities():
    """
    Observations
        - The malaria test dataset which I used for testing, has fever scans in one spectrum \
          than what I've seen before. The fewer the scans in one spectrum, the smaller the \
          sigma value for the filter_noise function has to be in order for it to work well. 
    """
    file_list = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))
    spectrums = read_spectra(source_files=file_list)
    spectrum = spectrums[0]
    for scan in spectrum.scans:
        plt.plot(scan.mz_array, scan.intensity_array, label="Raw data")
        scan.align_baseline(deg=4)
        plt.plot(scan.mz_array, scan.intensity_array, label="Baseline aligned")
        scan.filter_noise(sigma=15)
        plt.plot(scan.mz_array, scan.intensity_array, label="Filtered noise")
        peaks = scan.get_peaks(thres=7000000, min_dist=60)
        mz_values = [peak.mz for peak in peaks]
        intensity_values = [peak.intensity for peak in peaks]
        plt.scatter(mz_values, intensity_values, marker="x", color="red", label="Peaks")

        plt.legend()
        plt.show()


def plot_spectrum_functionalities():
    """
    Observations:
        - A lot of small peaks are detected
    """
    file_list = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))
    spectrums = read_spectra(source_files=file_list)
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=60)

        mz_values = [peak.mz for peak in spectrum.peaks]
        intensity_values = [peak.intensity for peak in spectrum.peaks]

        plt.scatter(mz_values, intensity_values, marker="x", color="red", label="Peaks")
        plt.xlabel("M/Z")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()
        break


def read_csv():
    path = "/home/emeseboga/Downloads/Naive_071.csv"
    read_spectra(source_files=[path])


def read_csv_files():
    file_list = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))
    read_spectra(source_files=file_list)


def save_preprocessor_functionalities():
    file_list = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))
    spectrums = read_spectra(source_files=[file_list[0], file_list[1]])
    for spectrum in spectrums:
        spectrum.align_baselines(deg=4)
        spectrum.filter_noise(sigma=20)
        spectrum.detect_peaks(thres=7000000, min_dist=60)

    color_map = cm.get_cmap("tab10")

    for i, spectrum in enumerate(spectrums):
        for peak in spectrum.peaks:
            plt.scatter(
                peak.retention_time,
                peak.intensity,
                color=color_map(i),
            )
    master_spectrum = find_master_spectrum(spectrum_list=spectrums)
    test_spectrums = find_test_spectrums(
        spectrum_list=spectrums, master_spectrum=master_spectrum
    )
    transformed_peaks = transform_peaks(
        mz_adj_win=0.2,
        rt_adj_win=20,
        master_spectrum=master_spectrum,
        test_spectrums=test_spectrums,
    )
    smoothened_peaks = apply_loess_regression(transformed_peaks=transformed_peaks)
    smoothened_peaks = sorted(
        smoothened_peaks, key=lambda peak: (peak.mz, peak.retention_time)
    )
    transformed_csv = "transformed.csv"
    for peak_tuple in transformed_peaks:
        plt.scatter(
            peak_tuple[0].retention_time,
            peak_tuple[0].intensity,
            color=color_map(4),
        )
        plt.scatter(
            peak_tuple[1].retention_time,
            peak_tuple[1].intensity,
            color=color_map(4),
        )

    for peak in smoothened_peaks:
        plt.scatter(
            peak.retention_time,
            peak.intensity,
            color=color_map(5),
        )

    plt.xlabel("Retention Time")
    plt.ylabel("Intensity")
    plt.title("Peaks for Each Spectrum")
    plt.show()

    with open(transformed_csv, "w", newline="") as csvfile:
        fieldnames = ["Peak Index", "Retention Time", "Intensity", "m/z"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for element in transformed_peaks:
            writer.writerow(
                {
                    "Peak Index": element[0].peak_index,
                    "Retention Time": element[0].retention_time,
                    "Intensity": element[0].intensity,
                    "m/z": element[0].mz,
                }
            )
            writer.writerow(
                {
                    "Peak Index": element[1].peak_index,
                    "Retention Time": element[1].retention_time,
                    "Intensity": element[1].intensity,
                    "m/z": element[1].mz,
                }
            )

    smoothened_csv = "smoothened.csv"

    with open(smoothened_csv, "w", newline="") as csvfile:
        fieldnames = ["Peak Index", "Retention Time", "Intensity", "m/z"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for element in smoothened_peaks:
            writer.writerow(
                {
                    "Peak Index": element.peak_index,
                    "Retention Time": element.retention_time,
                    "Intensity": element.intensity,
                    "m/z": element.mz,
                }
            )


def run_preprocessor():
    file_list = list(str(path) for path in EXPERIMENT_PATH.glob("*.csv"))
    feature_matrix = preprocess_data(source_files=file_list)
    csv_file_path = "output.csv"

    feature_matrix.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    plot_spectrum_functionalities()
