from typing import Dict, List

import numpy as np


CODE_LENGTH_L2CM = 10230
CODE_LENGTH_L2CL = 767250
CODE_RATE_L2CM = CODE_RATE_L2CL = 511500

CODE_LENGTH_L2CLM = 1534500  # 2 * CODE_LENGTH_L
CODE_RATE_L2CLM = 1023000

DATA_SYMBOL_RATE = 50

CARRIER_FREQ = 1.2276e9


"""
(svid, prn, cm_initial_state, cl_initial_state, cm_end_state, cl_end_state)
Tuple struct to store data from Tabel 3-II of the IS-GPS 200 
specification, which contains shift register state information for L2
signals.

`cm_initial_state`, `cm_end_state`, etc. are represented in octal in the table,
but should just be integer types in this structure.

Note that SVID and PRN numbers differ only for SVIDs 65-69.
"""
L2_CODE_PHASE_ASSIGNMENTS = {
    1: (1, 1, 0o742417664, 0o624145772, 0o552566002, 0o267724236),
    2: (2, 2, 0o756014035, 0o506610362, 0o034445034, 0o167516066),
    3: (3, 3, 0o002747144, 0o220360016, 0o723443711, 0o771756405),
    4: (4, 4, 0o066265724, 0o710406104, 0o511222013, 0o047202624),
    5: (5, 5, 0o601403471, 0o001143345, 0o463055213, 0o052770433),
    6: (6, 6, 0o703232733, 0o053023326, 0o667044524, 0o761743665),
    7: (7, 7, 0o124510070, 0o652521276, 0o652322653, 0o133015726),
    8: (8, 8, 0o617316361, 0o206124777, 0o505703344, 0o610611511),
    9: (9, 9, 0o047541621, 0o015563374, 0o520302775, 0o352150323),
    10: (10, 10, 0o733031046, 0o561522076, 0o244205506, 0o051266046),
    11: (11, 11, 0o713512145, 0o023163525, 0o236174002, 0o305611373),
    12: (12, 12, 0o024437606, 0o117776450, 0o654305531, 0o504676773),
    13: (13, 13, 0o021264003, 0o606516355, 0o435070571, 0o272572634),
    14: (14, 14, 0o230655351, 0o003037343, 0o630431251, 0o731320771),
    15: (15, 15, 0o001314400, 0o046515565, 0o234043417, 0o631326563),
    16: (16, 16, 0o222021506, 0o671511621, 0o535540745, 0o231516360),
    17: (17, 17, 0o540264026, 0o605402220, 0o043056734, 0o030367366),
    18: (18, 18, 0o205521705, 0o002576207, 0o731304103, 0o713543613),
    19: (19, 19, 0o064022144, 0o525163451, 0o412120105, 0o232674654),
    20: (20, 20, 0o120161274, 0o266527765, 0o365636111, 0o641733155),
    21: (21, 21, 0o044023533, 0o006760703, 0o143324657, 0o730125345),
    22: (22, 22, 0o724744327, 0o501474556, 0o110766462, 0o000316074),
    23: (23, 23, 0o045743577, 0o743747443, 0o602405203, 0o171313614),
    24: (24, 24, 0o741201660, 0o615534726, 0o177735650, 0o001523662),
    25: (25, 25, 0o700274134, 0o763621420, 0o630177560, 0o023457250),
    26: (26, 26, 0o010247261, 0o720727474, 0o653467107, 0o330733254),
    27: (27, 27, 0o713433445, 0o700521043, 0o406576630, 0o625055726),
    28: (28, 28, 0o737324162, 0o222567263, 0o221777100, 0o476524061),
    29: (29, 29, 0o311627434, 0o132765304, 0o773266673, 0o602066031),
    30: (30, 30, 0o710452007, 0o746332245, 0o100010710, 0o012412526),
    31: (31, 31, 0o722462133, 0o102300466, 0o431037132, 0o705144501),
    32: (32, 32, 0o050172213, 0o255231716, 0o624127475, 0o615373171),
    33: (65, 33, 0o500653703, 0o437661701, 0o154624012, 0o041637664),
    34: (66, 34, 0o755077436, 0o717047302, 0o275636742, 0o100107264),
    35: (67, 35, 0o136717361, 0o222614207, 0o644341556, 0o634251723),
    36: (68, 36, 0o756675453, 0o561123307, 0o514260662, 0o257012032),
    37: (69, 37, 0o435506112, 0o240713073, 0o133501670, 0o703702423),
}


def shift_state(state: int) -> int:
    """
    Applying the shift register polynomial involves an xor operation of the current state
    and the polynomial followed by a right shift. The MSb comes from the last bit of the current state.
    """
    poly = 0b01001001010010010101001111001
    next_state = (((state & 1) * poly) ^ state) >> 1
    return next_state


def generate_code_sequence_L2CM(prn: int) -> np.ndarray:
    """
    Generates the L2 CM code given the initial shift register state.
    """
    initial_state = L2_CODE_PHASE_ASSIGNMENTS[prn][2]
    state = initial_state
    code_seq_01 = np.zeros((CODE_LENGTH_L2CM,), dtype=np.int8)
    for i in range(CODE_LENGTH_L2CM):
        code_seq_01[i] = state & 0x1
        state = shift_state(state)
    return code_seq_01


def generate_code_sequence_L2CL(prn: int) -> np.ndarray:
    """
    Generates the L2 CL code given the initial shift register state.
    """
    initial_state = L2_CODE_PHASE_ASSIGNMENTS[prn][3]
    state = initial_state
    code_seq_01 = np.zeros((CODE_LENGTH_L2CL,), dtype=np.int8)
    for i in range(CODE_LENGTH_L2CL):
        code_seq_01[i] = state & 0x1
        state = shift_state(state)
    return code_seq_01

CODE_SEQUENCES_L2CL = {prn: 1 - 2 * generate_code_sequence_L2CL(prn).astype(np.int8) for prn in range(1, 33)}
CODE_SEQUENCES_L2CM = {prn: 1 - 2 * generate_code_sequence_L2CM(prn).astype(np.int8) for prn in range(1, 33)}


_CODE_SEQUENCES_GPS_L2CL = {}
def get_GPS_L2CL_code_sequence(prn: int) -> np.ndarray:
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _CODE_SEQUENCES_GPS_L2CL:
        _CODE_SEQUENCES_GPS_L2CL[prn] = generate_code_sequence_L2CL(prn)
    return _CODE_SEQUENCES_GPS_L2CL[prn]


_CODE_SEQUENCES_GPS_L2CM = {}
def get_GPS_L2CM_code_sequence(prn: int) -> np.ndarray:
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _CODE_SEQUENCES_GPS_L2CM:
        _CODE_SEQUENCES_GPS_L2CM[prn] = generate_code_sequence_L2CM(prn)
    return _CODE_SEQUENCES_GPS_L2CM[prn]
