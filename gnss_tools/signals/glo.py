"""
Author Brian Breitsch
Date: 2025-01-02
"""

import numpy as np

CODE_RATE = 0.511E6
CODE_LENGTH = 511

# GLONASS C/A code (GLONASS ICD)
def gencode_G1G2() -> np.ndarray:
    code = np.zeros(CODE_LENGTH, dtype=np.int8)
    reg = -np.ones(9, dtype=np.int8)
    for i in range(CODE_LENGTH):
        code[i] = -reg[6]
        newbit = reg[4] * reg[8]
        for j in range(8, 0):
            reg[j] = reg[j - 1]
        reg[0] = newbit
    return code