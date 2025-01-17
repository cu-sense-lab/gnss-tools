"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import os, tarfile, gzip, shutil, re, numpy as np
from typing import Tuple
import netCDF4
from datetime import datetime, timedelta
from gnss_lib.misc.data_utils import format_filepath, ftp_download
from gnss_lib.time.gpst import GPSTime


def download_ISDC_NAVBIT_file(
    file_dt: datetime, prn: int, data_dir: str, overwrite: bool = False
) -> str:
    """Given a datetime object for a particular day, along with the PRN and user data directory,
    retrieves GPS NAVBIT files from the GFZ Potsdam ISDC FTP site.
    """
    ftp_host = "isdcftp.gfz-potsdam.de"
    server_tar_filepath_template = (
        "gnss/GNSS-GPS-1-NAVBIT/y{yyyy}/GNSS-GPS-1-NAVBIT+{yyyy}_{ddd}_A.tar"
    )
    server_tar_filepath = format_filepath(server_tar_filepath_template, dt=file_dt)

    local_tar_filepath = os.path.join(data_dir, server_tar_filepath)
    download_dir = os.path.dirname(local_tar_filepath)
    os.makedirs(download_dir, exist_ok=True)

    nc_filename_pattern_template = "NAVBIT-GPS-L1CA-{yyyy}-{ddd}-{prn:02}-{treg}.nc"
    treg_pattern = "(\d{5})"
    nc_filename_pattern = format_filepath(
        nc_filename_pattern_template, dt=file_dt, prn=prn, treg=treg_pattern
    )
    gz_filename_pattern = nc_filename_pattern + ".gz"

    nc_filepath = None
    for filename in os.listdir(download_dir):
        match = re.match(nc_filename_pattern, filename)
        if match:
            nc_filepath = os.path.join(download_dir, filename)
    if not overwrite and nc_filepath is not None:
        return nc_filepath
    else:
        # Download day-of-year TAR
        download_success = ftp_download(
            ftp_host, server_tar_filepath, local_tar_filepath
        )
        if not download_success:
            raise Exception("Failed to download {0}".format(server_tar_filepath))
        # Extract TAR
        with tarfile.TarFile(local_tar_filepath, "r") as f:
            f.extractall(download_dir)
        # Unzip each .gz
        extracted_filepaths = []
        for filename in os.listdir(download_dir):
            if filename.endswith(".gz"):
                gz_filepath = os.path.join(download_dir, filename)
                extracted_filepath = gz_filepath[:-3]
                with gzip.open(gz_filepath, "rb") as f_in, open(
                    extracted_filepath, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_filepath)
            if re.match(nc_filename_pattern, filename):
                extracted_filepaths.append(extracted_filepath)
        # Check if one of the extracted .nc files matches the requested file and if so, return it.
        for filename in os.listdir(download_dir):
            match = re.match(nc_filename_pattern, filename)
            if match:
                nc_filepath = os.path.join(download_dir, filename)
                return nc_filepath
        # If we can't find the requested file at this point, then raise an error
        raise Exception(
            "Filepath {0} not found in extracted netCDF filepaths".format(nc_filepath)
        )


def get_nav_bits(
    start_time: GPSTime,
    end_time: GPSTime,
    prn: int,
    data_dir: str,
    overwrite: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given a start time and end time in GPS seconds, retrieves and parses the GPS navigation bits
    for epochs between the start and end times.  Returns `times, bits`"""
    GPS_EPOCH = datetime(year=1980, month=1, day=6, hour=0, minute=0, second=0)
    start_time_dt = start_time.to_datetime()
    end_time_dt = end_time.to_datetime()
    day = start_time_dt
    nc_filepaths = []
    while day < end_time_dt + timedelta(days=1):
        nc_filepath = download_ISDC_NAVBIT_file(day, prn, data_dir, overwrite=overwrite)
        nc_filepaths.append(nc_filepath)
        day += timedelta(days=1)
    segments = []
    for nc_filepath in nc_filepaths:
        with netCDF4.Dataset(nc_filepath) as f:
            # I think the timing of these variables may work the same way as SP3 files
            # So we can use timezone naive GPS_EPOCH
            year = int(f["year"][0])
            day_of_year = int(f["day_of_year"][0])
            file_start_dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
            # file_start_gpst = dt2gpst(file_start_dt)
            # Nevermind, looks like we have to grab it this way
            file_start_gpst = (file_start_dt - GPS_EPOCH).total_seconds()

            # Flatten the words in `navbits` and create time array corresponding to each word
            navbit_words = f["navbits"][:, :].T.flatten()
            n_subframes = int(f["nofsfr"][0])
            subframe_times = f["time"][
                :
            ]  # Time of each subframe (10 words per subframe, 30 bits per word)
            word_times = subframe_times.repeat(10) + numpy.tile(
                numpy.arange(10) * 0.6, n_subframes
            )
            word_times_gpst = file_start_gpst + word_times

            # Get words that are within our time range
            indices = (start_time <= word_times_gpst) * (word_times_gpst < end_time + 1)
            segments.append(
                {
                    "words": navbit_words[indices],
                    "word_times_gpst": word_times_gpst[indices],
                }
            )
    # Concatenate all the relevant words together
    words = np.concatenate([seg["words"] for seg in segments]).reshape((-1, 1))
    word_times_gpst = np.concatenate([seg["word_times_gpst"] for seg in segments])
    n_words = len(word_times_gpst)

    # The first segment should contain the bits right after `start_time_gpst`
    # If remainders are larger than 1 seconds, then the retrieved navbits from this file do not start from the correct word
    start_time_remainder = word_times_gpst[0] - start_time
    end_time_remainder = end_time - word_times_gpst[-1]
    assert 0 <= start_time_remainder < 1
    # assert(0 < end_time_remainder <= 1)

    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")[:, :30][
        :, ::-1
    ].flatten()
    bit_times_gpst = word_times_gpst.repeat(30) + np.tile(
        np.arange(30) * 0.02, n_words
    )
    indices = (start_time <= bit_times_gpst) * (bit_times_gpst < end_time + 1)
    return bit_times_gpst[indices], bits[indices]
