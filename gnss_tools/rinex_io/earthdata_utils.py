# """
# Author Brian Breitsch
# Date: 2025-01-02
# """

# from gnss_tools.misc.data_utils import format_filepath, http_download
# import requests
# import os
# import logging
# from datetime import datetime
# from typing import List, Tuple, Optional
# import gnss_tools.rinex_io.rinex3 as rinex3
# import hatanaka


# CDDIS_ARCHIVE_URL = f"https://cddis.nasa.gov/archive/"
# RINEX3_ARCHIVE_PATH_TEMPLATE = "gnss/data/daily/{yyyy}/{ddd}/{yy}d/"
# # RINEX3_ARCHIVE_PATH_TEMPLATE = "gnss/data/daily/{yyyy}/{ddd}/{yy}d/"

# EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME")
# EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD")
# if EARTHDATA_USERNAME is None or EARTHDATA_PASSWORD is None:
#     logging.warning("EARTHDATA_USERNAME and EARTHDATA_PASSWORD not set")
#     EARTHDATA_AUTH = None
# else:
#     EARTHDATA_AUTH = (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)

# DATA_DIR = os.getenv("DATA_DIR")
# if DATA_DIR is None:
#     logging.warning("DATA_DIR not set; using working directory")
#     DATA_DIR = "./"


# def get_earthdata_file_list(
#     dt: datetime,
#     archive_path_template: str = RINEX3_ARCHIVE_PATH_TEMPLATE,
#     auth: Optional[Tuple[str, str]] = None
# ) -> Tuple[List[str], List[int]]:

#     rinex3_archive_path = format_filepath(archive_path_template, dt)
#     url = CDDIS_ARCHIVE_URL + rinex3_archive_path
#     url += "*?list"
#     r = requests.get(url, auth=auth)
#     if r.status_code != 200:
#         raise Exception("Failure to retrieve list: {0}".format(url))
#     items = r.content.decode("utf-8").splitlines()
#     file_names, file_sizes = zip(*map(lambda x: x.split(), items))
#     return list(file_names), list(file_sizes)


# def _download_station_data_for_day(
#     day: datetime,
#     station_ids: List[str],
#     overwrite: bool = False,
#     data_dir: str = DATA_DIR,
# ) -> dict[str, str]:
#     relevant_rinex3_filenames: dict[str, str] = {}  # station_id -> filename
#     file_names, file_sizes = get_cddis_rinex3_file_list(day, auth=EARTHDATA_AUTH)
#     for station_id in station_ids:
#         for fname in file_names:
#             if fname.startswith(station_id):
#                 relevant_rinex3_filenames[station_id] = fname

#     rinex3_zipped_filepaths: dict[str, str] = {}
#     for station_id, filename in relevant_rinex3_filenames.items():
#         rinex3_archive_path = format_filepath(RINEX3_ARCHIVE_PATH_TEMPLATE, day)
#         url = os.path.join(CDDIS_ARCHIVE_URL, rinex3_archive_path, filename)
#         print(f"Downloading {url}", end="... ")
#         output_filepath = os.path.join(data_dir, "cddis", rinex3_archive_path, filename)
#         if os.path.exists(output_filepath) and not overwrite:
#             print("Already downloaded")
#             rinex3_zipped_filepaths[station_id] = output_filepath
#             continue
#         success = http_download(url, output_filepath, auth=EARTHDATA_AUTH)
#         if success:
#             print("Success")
#             rinex3_zipped_filepaths[station_id] = output_filepath
#         else:
#             print("Failed")
#     return rinex3_zipped_filepaths


# def download_station_data(
#     days: list[datetime],
#     station_ids: List[str],
#     overwrite: bool = False,
#     data_dir: str = DATA_DIR,
# ) -> dict[str, list[str]]:
#     rinex3_zipped_filepaths: dict[str, list[str]] = {}
#     for day in days:
#         filepaths = _download_station_data_for_day(
#             day, station_ids, overwrite, data_dir
#         )
#         for station_id, filepath in filepaths.items():
#             if station_id not in rinex3_zipped_filepaths:
#                 rinex3_zipped_filepaths[station_id] = []
#             rinex3_zipped_filepaths[station_id].append(filepath)
#     return rinex3_zipped_filepaths


# def load_station_data(
#     rinex3_zipped_filepaths: dict[str, list[str]],  # station_id -> list of filepaths
#     strict: bool = False,
# ) -> dict[str, rinex3.Dataset]:
#     # Directly read RINEX3 files with Hatanaka decompression
#     rinex3_datasets = {}
#     for station_id, filepaths in rinex3_zipped_filepaths.items():
#         dataset_filepaths = []
#         for filepath in filepaths:
#             print(f"Reading {filepath}")
#             filepath = hatanaka.decompress_on_disk(filepath)
#             dataset_filepaths.append(filepath)
#         dataset = rinex3.Dataset()
#         dataset.load_files(dataset_filepaths, strict)
#         rinex3_datasets[station_id] = dataset
#     return rinex3_datasets
