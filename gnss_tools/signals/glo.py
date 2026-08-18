"""
GLONASS L1/L2 C/A code generation.

Author Brian Breitsch
Date: 2025-01-02

INCOMPLETE / KNOWN BUG: `gencode_G1G2`'s register-shift loop
(`for j in range(8, 0)`) is an empty range and never executes, so the shift
register never advances -- the function as written does not produce a correct
GLONASS C/A code. There is also no carrier-frequency table (GLONASS is FDMA:
each satellite's L1/L2 frequency depends on its frequency channel number, not
a single constant like GPS), no L2 C/A or L3 (CDMA) signal, and no nav message
support. Not wired into `catalog.py` or used anywhere in this project.
Left as-is rather than fixed opportunistically: fixing it without a reference
GLONASS ICD test vector to validate against would risk a plausible-looking but
still-wrong implementation. See the project's TODO.md.
"""

import numpy as np

CODE_RATE = 0.511E6
CODE_LENGTH = 511

# GLONASS C/A code (GLONASS ICD) -- see module docstring: currently broken.
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