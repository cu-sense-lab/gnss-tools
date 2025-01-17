"""
Author: Brian Breitsch
Date: 2025-01-02
"""

import os
import requests
import gzip
from datetime import datetime
from typing import List, Tuple, Optional
from gnss_tools.misc.data_utils import format_filepath


def get_cddis_ionex_file_list(
    dt: datetime,
    auth: Optional[Tuple[str, str]] = None,
    cddis_url: str = "https://cddis.nasa.gov/",
    ionex_path_template: str = "archive/gnss/products/ionex/{yyyy}/{ddd}/",
) -> Tuple[List[str], List[int]]:

    ionex_archive_path = format_filepath(ionex_path_template, dt)
    url = cddis_url + ionex_archive_path
    url += "*?list"
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise Exception("Failure to retrieve list: {0}".format(url))
    items = r.content.decode("utf-8").splitlines()
    file_names, file_sizes = zip(*map(lambda x: x.split(), items))
    return list(file_names), list(file_sizes)


def download_ionex_file(
    dt: datetime,
    data_dir: str,
    auth: Optional[Tuple[str, str]] = None,
    cddis_url: str = "https://cddis.nasa.gov/",
    ionex_path_template: str = "archive/gnss/products/ionex/{yyyy}/{ddd}/",
    ionex_file_template: str = "IGS0OPSFIN_{yyyy}{ddd}0000_01D_02H_GIM.INX.gz",
) -> str:
    ionex_filepath = os.path.join(
        format_filepath(ionex_path_template, dt),
        format_filepath(ionex_file_template, dt),
    )
    url = cddis_url + ionex_filepath
    r = requests.get(url, auth=auth)
    if r.status_code != 200:
        raise Exception("Failure to retrieve file: {0}".format(url))
    downloaded_filepath = os.path.join(data_dir, ionex_filepath)
    os.makedirs(os.path.dirname(downloaded_filepath), exist_ok=True)
    with open(downloaded_filepath, "wb") as f:
        f.write(r.content)
    return downloaded_filepath


def download_and_decompress_ionex_file(
    dt: datetime,
    data_dir: str,
    auth: Optional[Tuple[str, str]] = None,
    cddis_url: str = "https://cddis.nasa.gov/",
    ionex_path_template: str = "archive/gnss/products/ionex/{yyyy}/{ddd}/",
    ionex_file_template: str = "IGS0OPSFIN_{yyyy}{ddd}0000_01D_02H_GIM.INX.gz",
    overwrite: bool = False,
) -> str:
    ionex_filepath = os.path.join(
        format_filepath(ionex_path_template, dt),
        format_filepath(ionex_file_template, dt),
    )
    url = cddis_url + ionex_filepath
    downloaded_filepath = os.path.join(data_dir, ionex_filepath)
    os.makedirs(os.path.dirname(downloaded_filepath), exist_ok=True)
    decompressed_filepath = downloaded_filepath[:-3]  # remove .gz
    if not os.path.exists(decompressed_filepath) or overwrite:
        r = requests.get(url, auth=auth)
        if r.status_code != 200:
            raise Exception("Failure to retrieve file: {0}".format(url))
        with open(downloaded_filepath, "wb") as f:
            f.write(r.content)

        with gzip.open(downloaded_filepath, "rb") as f_in:
            with open(decompressed_filepath, "wb") as f_out:
                f_out.write(f_in.read())
    return decompressed_filepath
    