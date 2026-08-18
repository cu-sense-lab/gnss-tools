"""
Dump PRN code sequences to hex files, one per signal.

Run as a script from the repository root: `python -m gnss_tools.signals.utils.generate_hex`.
"""

import os

import numpy as np

from ..gps_l1ca import generate_code_sequence_L1CA
from ..gps_l2c import generate_code_sequence_L2CL, generate_code_sequence_L2CM
from ..gps_l5 import generate_code_sequence_L5I, generate_code_sequence_L5Q

HEXFILES_DIR = os.path.join(os.path.dirname(__file__), "hexfiles")


def _code_to_hex(code_seq_01: np.ndarray) -> str:
    """
    Pack a 0/1 code sequence into a hex string, zero-padding the final byte.

    Cast to int first: L5's getters return float64 0/1 (unlike L1CA/L2C's int8),
    and `str(1.0)` is `"1.0"`, not `"1"`.
    """
    binstr = "".join(str(int(b)) for b in code_seq_01)
    n_full_bytes = len(binstr) // 8
    hexstr = "".join(
        f"{int(binstr[i * 8:(i + 1) * 8], 2):02x}" for i in range(n_full_bytes)
    )
    remainder = binstr[n_full_bytes * 8:]
    if remainder:
        hexstr += f"{int(remainder.ljust(8, '0'), 2):02x}"
    return hexstr


def write_hex_file(out_filepath: str, header: str, prn_list: list, get_code_sequence) -> None:
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    with open(out_filepath, "w") as f:
        f.write(header + "\n")
        f.write(" ".join(str(prn) for prn in prn_list) + "\n")
        for prn in prn_list:
            f.write(_code_to_hex(get_code_sequence(prn)) + "\n")


if __name__ == "__main__":
    prn_list = list(range(1, 33))

    write_hex_file(
        os.path.join(HEXFILES_DIR, "gps_l1ca.txt"),
        "GPS L1CA code sequences.  Second line contains space-separated PRN list.  "
        "Subsequent lines contain PRN codes in HEX format.",
        prn_list,
        generate_code_sequence_L1CA,
    )
    write_hex_file(
        os.path.join(HEXFILES_DIR, "gps_l2cm.txt"),
        "GPS L2CM code sequences.  Second line contains space-separated PRN list.  "
        "Subsequent lines contain PRN codes in HEX format.",
        prn_list,
        generate_code_sequence_L2CM,
    )
    write_hex_file(
        os.path.join(HEXFILES_DIR, "gps_l2cl.txt"),
        "GPS L2CL code sequences.  Second line contains space-separated PRN list.  "
        "Subsequent lines contain PRN codes in HEX format.",
        prn_list,
        generate_code_sequence_L2CL,
    )
    write_hex_file(
        os.path.join(HEXFILES_DIR, "gps_l5i.txt"),
        "GPS L5I code sequences.  Second line contains space-separated PRN list.  "
        "Subsequent lines contain PRN codes in HEX format.",
        prn_list,
        generate_code_sequence_L5I,
    )
    write_hex_file(
        os.path.join(HEXFILES_DIR, "gps_l5q.txt"),
        "GPS L5Q code sequences.  Second line contains space-separated PRN list.  "
        "Subsequent lines contain PRN codes in HEX format.",
        prn_list,
        generate_code_sequence_L5Q,
    )
    print(f"Wrote hex files to {HEXFILES_DIR}")
