from typing import List

import pandas as pd
from pyne.models.spectrum import Spectrum


def read_spectra(source_files: List[str]) -> List[Spectrum]:
    """
    Returns a Spectrum list from the given source files. The source files are .csv files.
    The .csv files should have the following columns:
        - 'RT': Retention Time (float)
        - 'mzarray': List of mass-to-charge ratios (List[float] as str)
        - 'intarray': List of intensities (List[float] as str)

    Args:
        source_files (List[str]): A list of .csv files to perform data preprocessing on.
    """
    spectrum_list = list()
    for file in source_files:
        data_frame = pd.read_csv(
            file,
            converters={
                "mzarray": convert_str_to_float_list,
                "intarray": convert_str_to_float_list,
            },
        )
        data_frame = data_frame.rename(
            columns={"mzarray": "mz_array", "intarray": "intensity_array"}
        )
        spectrum_list.append(Spectrum(data_frame))

    return spectrum_list


def convert_str_to_float_list(s: str):
    values = s.strip("[]").split()
    return [float(x) for x in values if x != "..."]
