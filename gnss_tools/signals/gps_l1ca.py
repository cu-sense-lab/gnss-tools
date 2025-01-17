from typing import Optional
import numpy as np
from .utils.mls import generate_mls

CARRIER_FREQ: int = 1575420000
CODE_RATE: int = 1023000
CODE_LENGTH: int = 1023
DATA_SYMBOL_RATE: int = 50

"""
(svid, prn, ca_phase_select, x2_phase_select, ca_code_delay, p_code_delay, first_10_chips_ca, first_12_chips_p)
Tuple struct to store data from Table 3-I of the IS-GPS 200 
specification, which contains code phase assignment information for GPS L1 signal.

`ca_phase_select` is a 2-tuple in this structure.

`first_12_chips_p`, `first_10_chips_ca` are represented in octal in the table,
but should just be integer types in this structure.

Note that SVID and PRN numbers differ only for SVIDs 65-69.
"""
CODE_PHASE_ASSIGNMENTS = {
    1: (1, 1, (2, 6), 1, 5, 1, 1440, 4444),
    2: (2, 2, (3, 7), 2, 6, 2, 1620, 4000),
    3: (3, 3, (4, 8), 3, 7, 3, 1710, 4333),
    4: (4, 4, (5, 9), 4, 8, 4, 1744, 4377),
    5: (5, 5, (1, 9), 5, 17, 5, 1133, 4355),
    6: (6, 6, (2, 10), 6, 18, 6, 1455, 4344),
    7: (7, 7, (1, 8), 7, 139, 7, 1131, 4340),
    8: (8, 8, (2, 9), 8, 140, 8, 1454, 4342),
    9: (9, 9, (3, 10), 9, 141, 9, 1626, 4343),
    10: (10, 10, (2, 3), 10, 251, 10, 1504, 4343),
    11: (11, 11, (3, 4), 11, 252, 11, 1642, 4343),
    12: (12, 12, (5, 6), 12, 254, 12, 1750, 4343),
    13: (13, 13, (6, 7), 13, 255, 13, 1764, 4343),
    14: (14, 14, (7, 8), 14, 256, 14, 1772, 4343),
    15: (15, 15, (8, 9), 15, 257, 15, 1775, 4343),
    16: (16, 16, (9, 10), 16, 258, 16, 1776, 4343),
    17: (17, 17, (1, 4), 17, 469, 17, 1156, 4343),
    18: (18, 18, (2, 5), 18, 470, 18, 1467, 4343),
    19: (19, 19, (3, 6), 19, 471, 19, 1633, 4343),
    20: (20, 20, (4, 7), 20, 472, 20, 1715, 4343),
    21: (21, 21, (5, 8), 21, 473, 21, 1746, 4343),
    22: (22, 22, (6, 9), 22, 474, 22, 1763, 4343),
    23: (23, 23, (1, 3), 23, 509, 23, 1063, 4343),
    24: (24, 24, (4, 6), 24, 512, 24, 1706, 4343),
    25: (25, 25, (5, 7), 25, 513, 25, 1743, 4343),
    26: (26, 26, (6, 8), 26, 514, 26, 1761, 4343),
    27: (27, 27, (7, 9), 27, 515, 27, 1770, 4343),
    28: (28, 28, (8, 10), 28, 516, 28, 1774, 4343),
    29: (29, 29, (1, 6), 29, 859, 29, 1127, 4343),
    30: (30, 30, (2, 7), 30, 860, 30, 1453, 4343),
    31: (31, 31, (3, 8), 31, 861, 31, 1625, 4343),
    32: (32, 32, (4, 9), 32, 862, 32, 1712, 4343),
    33: (65, 33, (5, 10), 33, 863, 33, 1745, 4343),
    34: (66, 34, (4, 10), 34, 950, 34, 1713, 4343),
    35: (67, 35, (1, 7), 35, 947, 35, 1134, 4343),
    36: (68, 36, (2, 8), 36, 948, 36, 1456, 4343),
    37: (69, 37, (4, 10), 37, 950, 37, 1713, 4343),
}


def generate_code_sequence_L1CA(prn: int) -> np.ndarray[np.int8]:
    """Generates L1CA PRN code for given PRN.

    Parameters
    ----------
    prn : int
        the PRN of the signal/satellite

    Returns
    -------
    output : ndarray of shape(1023,)
        the complete binary (0/1) code sequence
    """
    ps = CODE_PHASE_ASSIGNMENTS[prn][2]
    g1 = generate_mls(10, [2, 9], [9])
    g2 = generate_mls(10, [1, 2, 5, 7, 8, 9], [ps[0] - 1, ps[1] - 1])
    code_seq_01 = (g1 + g2) % 2
    return code_seq_01.astype(np.int8)


_CODE_SEQUENCES_GPS_L1CA = {}
def get_GPS_L1CA_code_sequence(prn: int) -> np.ndarray:
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _CODE_SEQUENCES_GPS_L1CA:
        _CODE_SEQUENCES_GPS_L1CA[prn] = generate_code_sequence_L1CA(prn)
    return _CODE_SEQUENCES_GPS_L1CA[prn]

