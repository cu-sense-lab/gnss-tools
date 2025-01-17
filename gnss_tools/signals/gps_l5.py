
import numpy as np


PRIMARY_CODE_LENGTH = 10230
CODE_RATE = 10230000
NEUMAN_HOFFMAN_RATE = 1000

NEUMAN_HOFFMAN_SEQ_L5I = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
NEUMAN_HOFFMAN_SEQ_L5Q = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=np.int8)

CODE_LENGTH_L5I = PRIMARY_CODE_LENGTH * len(NEUMAN_HOFFMAN_SEQ_L5I)
CODE_LENGTH_L5Q = PRIMARY_CODE_LENGTH * len(NEUMAN_HOFFMAN_SEQ_L5Q)

DATA_SYMBOL_RATE = 100  # 50 Hz 1/2 convolutional encoding on the L5I channel

CARRIER_FREQ = 1.17645e9


XA_CODE_LENGTH = 8190
XB_CODE_LENGTH = 8191

'''
(svid, prn, xb_advance_i, xb_advance_q, xb_initial_state_i, xb_initial_state_q)
Tuple struct to store data from Tabel 3-I of the IS-GPS 705 
specification, which contains shift register state information for L5
signals.

`xb_advance_i`, `xb_advance_q`, are the code phase advance for I and Q signals respectively in chips
`xb_initial_state_i` and `xb_initial_state_q` are the initial shift register states
'''
L5_CODE_PHASE_ASSIGNMENTS = { 
    1 : (1, 1, 266, 1701, 0b0101011100100, 0b1001011001100),
    2 : (2, 2, 365, 323, 0b1100000110101, 0b0100011110110),
    3 : (3, 3, 804, 5292, 0b0100000001000, 0b1111000100011),
    4 : (4, 4, 1138, 2020, 0b1011000100110, 0b0011101101010),
    5 : (5, 5, 1509, 5429, 0b1110111010111, 0b0011110110010),
    6 : (6, 6, 1559, 7136, 0b0110011111010, 0b0101010101001),
    7 : (7, 7, 1756, 1041, 0b1010010011111, 0b1111110000001),
    8 : (8, 8, 2084, 5947, 0b1011110100100, 0b0110101101000),
    9 : (9, 9, 2170, 4315, 0b1111100101011, 0b1011101000011),
    10 : (10, 10, 2303, 148, 0b0111111011110, 0b0010010000110),
    11 : (11, 11, 2527, 535, 0b0000100111010, 0b0001000000101),
    12 : (12, 12, 2687, 1939, 0b1110011111001, 0b0101011000101),
    13 : (13, 13, 2930, 5206, 0b0001110011100, 0b0100110100101),
    14 : (14, 14, 3471, 5910, 0b0100000100111, 0b1010000111111),
    15 : (15, 15, 3940, 3595, 0b0110101011010, 0b1011110001111),
    16 : (16, 16, 4132, 5135, 0b0001111001001, 0b1101001011111),
    17 : (17, 17, 4332, 6082, 0b0100110001111, 0b1110011001000),
    18 : (18, 18, 4924, 6990, 0b1111000011110, 0b1011011100100),
    19 : (19, 19, 5343, 3546, 0b1100100011111, 0b0011001011011),
    20 : (20, 20, 5443, 1523, 0b0110101101101, 0b1100001110001),
    21 : (21, 21, 5641, 4548, 0b0010000001000, 0b0110110010000),
    22 : (22, 22, 5816, 4484, 0b1110111101111, 0b0010110001110),
    23 : (23, 23, 5898, 1893, 0b1000011111110, 0b1000101111101),
    24 : (24, 24, 5918, 3961, 0b1100010110100, 0b0110111110011),
    25 : (25, 25, 5955, 7106, 0b1101001101101, 0b0100010011011),
    26 : (26, 26, 6243, 5299, 0b1010110010110, 0b0101010111100),
    27 : (27, 27, 6345, 4660, 0b0101011011110, 0b1000011111010),
    28 : (28, 28, 6477, 276, 0b0111101010110, 0b1111101000010),
    29 : (29, 29, 6518, 4389, 0b0101111100001, 0b0101000100100),
    30 : (30, 30, 6875, 3783, 0b1000010110111, 0b1000001111001),
    31 : (31, 31, 7168, 1591, 0b0001010011110, 0b0101111100101),
    32 : (32, 32, 7187, 1601, 0b0000010111001, 0b1001000101010),
    33 : (65, 33, 7329, 749, 0b1101010000001, 0b1011001000100),
    34 : (66, 34, 7577, 1387, 0b1101111111001, 0b1111001000100),
    35 : (67, 35, 7720, 1661, 0b1111011011100, 0b0110010110011),
    36 : (68, 36, 7777, 3210, 0b1001011001000, 0b0011110101111),
    37 : (69, 37, 8057, 708, 0b0011010010000, 0b0010011010001),}


def generate_XA_code():
    '''
    ----------------------------------------------------------------------------
    Generates the XA codes used in generating the GPS L5 codes. The variable
    `state` represents the state of a 13-bit shift register. Shift amounts
    should be one less than the degree of the polynomial (since bit indexing
    starts at 1). Taps: 9, 10, 12, 13
    '''
    code_seq_01 = np.zeros((XA_CODE_LENGTH,))
    state = 0b1111111111111
    for i in range(XA_CODE_LENGTH):
        code_seq_01[i] = (state >> 12) & 1
        shift_in = ((state >> 12) ^ (state >> 11) ^ (state >> 9) ^ (state >> 8)) & 1
        state = (state << 1) | shift_in
    return code_seq_01


def generate_XB_code():
    '''
    ----------------------------------------------------------------------------
    Generates the XB codes used in generating the GPS L5 codes. The variable
    `state` represents the state of a 13-bit shift register. Shift amounts
    should be one less than the degree of the polynomial (since bit indexing
    starts at 1).  Taps: 1, 3, 4, 6, 7, 8, 12, 13
    '''
    code_seq_01 = np.zeros((XB_CODE_LENGTH,))
    state = 0b1111111111111
    for i in range(XB_CODE_LENGTH):
        code_seq_01[i] = (state >> 12) & 1
        shift_in = ((state >> 12) ^ (state >> 11) ^ (state >> 7) ^ (state >> 6)
                    ^ (state >> 5) ^ (state >> 3) ^ (state >> 2) ^ (state >> 0)) & 1
        state = (state << 1) | shift_in
    return code_seq_01


XA_CODE_SEQ = generate_XA_code()
XB_CODE_SEQ = generate_XB_code()
PRIMARY_CODE_LENGTH = 10230


def generate_code_sequence_L5I_L5Q(XB_advance):
    '''
    ----------------------------------------------------------------------------
    Generates the GPS L5 code (either I or Q) given the initial state of the XB
    shift register.
    '''
    indices = np.arange(PRIMARY_CODE_LENGTH)
    code_seq = (XA_CODE_SEQ[indices % XA_CODE_LENGTH] + XB_CODE_SEQ[(XB_advance + indices) % XB_CODE_LENGTH]) % 2  # <- initial state used in index
    return code_seq


def generate_code_sequence_L5I(prn):
    '''
    ----------------------------------------------------------------------------
    Generates the L5I code for the desired PRN.
    '''
    XB_advance_I = L5_CODE_PHASE_ASSIGNMENTS[prn][2]
    return generate_code_sequence_L5I_L5Q(XB_advance_I)

def generate_code_sequence_L5Q(prn):
    '''
    ----------------------------------------------------------------------------
    Generates the L5Q code for the desired PRN.
    '''
    XB_advance_Q = L5_CODE_PHASE_ASSIGNMENTS[prn][3]
    return generate_code_sequence_L5I_L5Q(XB_advance_Q)



CODE_SEQUENCES_L5I = {prn: (1 - 2 * generate_code_sequence_L5I(prn)).astype(np.int8) for prn in range(1, 33)}
CODE_SEQUENCES_L5Q = {prn: (1 - 2 * generate_code_sequence_L5Q(prn)).astype(np.int8) for prn in range(1, 33)}



_CODE_SEQUENCES_GPS_L5I = {}
def get_GPS_L5I_code_sequence(prn: int) -> np.ndarray:
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _CODE_SEQUENCES_GPS_L5I:
        _CODE_SEQUENCES_GPS_L5I[prn] = generate_code_sequence_L5I(prn)
    return _CODE_SEQUENCES_GPS_L5I[prn]


_CODE_SEQUENCES_GPS_L5Q = {}
def get_GPS_L5Q_code_sequence(prn: int) -> np.ndarray:
    '''
    Returns the code sequence corresponding to the given PRN
    '''
    if prn not in _CODE_SEQUENCES_GPS_L5Q:
        _CODE_SEQUENCES_GPS_L5Q[prn] = generate_code_sequence_L5Q(prn)
    return _CODE_SEQUENCES_GPS_L5Q[prn]
