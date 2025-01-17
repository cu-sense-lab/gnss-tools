from aenum import Enum, extend_enum
from dataclasses import dataclass
from typing import Final, List, Tuple
import numpy as np
import h5py


class SystemType(Enum):
    GPS = "GPS"
    GLONASS = "GLO"
    GALILEO = "GAL"
    BEIDOU = "BEI"
    QZSS = "QZS"


MAX_SYSTEM_TYPE_ENUM_STR_LENGTH = 3 + 1

SYSTEM_LETTERS = {
    SystemType.GPS: "G",
    SystemType.GLONASS: "R",
    SystemType.GALILEO: "E",
    SystemType.BEIDOU: "C",
    SystemType.QZSS: "Q",
}
SYSTEM_LETTERS_REVERSED = {v: k for k, v in SYSTEM_LETTERS.items()}


SYSTEM_NAMES = {
    SystemType.GPS: "GPS",
    SystemType.GLONASS: "GLONASS",
    SystemType.GALILEO: "Galileo",
    SystemType.BEIDOU: "Beidou",
    SystemType.QZSS: "QZSS",
}
SYSTEM_NAMES_REVERSED = {v: k for k, v in SYSTEM_NAMES.items()}


SYSTEM_ABREV = {
    SystemType.GPS: "GPS",
    SystemType.GLONASS: "GLO",
    SystemType.GALILEO: "GAL",
    SystemType.BEIDOU: "BEI",
    SystemType.QZSS: "QZS",
}
SYSTEM_ABREV_REVERSED = {v: k for k, v in SYSTEM_ABREV.items()}


class SignalType(Enum):
    L1CA = "L1CA"  # GPS/QZSS L1C/A
    L1CP = "L1CP"  # GPS/QZSS L1C Pilot
    L1CD = "L1CD"  # GPS/QZSS L1C Data
    L2CM = "L2CM"  # GPS/QZSS L2CM
    L2CL = "L2CL"  # GPS/QZSS L2CL
    L2CLM = "L2CLM"  # GPS combined L2CL and L2CM   # TODO: LM or ML?
    L5I = "L5I"  # GPS/QZSS L5I
    L5Q = "L5Q"  # GPS/QZSS L5Q
    L5IQ = "L5IQ"  # GPS combined L5 I/Q
    E1B = "E1B"  # Galileo E1B (Data)
    E1C = "E1C"  # Galileo E1C (Pilot)
    E5AI = "E5AI"  # Galileo E5aI (Data)
    E5AQ = "E5AQ"  # Galileo E5aQ (Pilot)
    E5BI = "E5BI"  # Galileo E5bI (Data)
    E5BQ = "E5BQ"  # Galileo E5bQ (Pilot)
    G1 = "G1"  # GLONASS G1
    G2 = "G2"  # GLONASS G2
    B1I = "B1I"  # BeiDou B1I
    B2I = "B2I"  # BeiDou B2I
    LEXS = "LEXS"  # QZSS LEX short
    LEXL = "LEXL"  # QZSS LEX long


MAX_SIGNAL_TYPE_ENUM_STR_LENGTH = 6 + 1


def add_signal(
    system: str,
    signal_types: List[str],
    system_letter: str,
    system_name: str,
    system_abrev: str,
) -> None:
    extend_enum(SystemType, system, system)
    for signal_type in signal_types:
        extend_enum(SignalType, signal_type, signal_type)
    SYSTEM_LETTERS[SystemType[system]] = system_letter
    SYSTEM_LETTERS_REVERSED[system_letter] = SystemType[system]
    SYSTEM_NAMES[SystemType[system]] = system_name
    SYSTEM_NAMES_REVERSED[system_name] = SystemType[system]
    SYSTEM_ABREV[SystemType[system]] = system_abrev
    SYSTEM_ABREV_REVERSED[system_abrev] = SystemType[system]


@dataclass
class SignalId:
    system: SystemType
    signal_type: SignalType
    prn: int

    def __lt__(self, other: "SignalId"):
        if self.system.value < other.system.value:
            return True
        elif self.system.value == other.system.value:
            if self.signal_type.value < other.signal_type.value:
                return True
            elif self.signal_type.value == other.signal_type.value:
                if self.prn < other.prn:
                    return True
        return False

    def __repr__(self):
        return f"{SYSTEM_LETTERS[self.system]}{self.prn:02}_{self.signal_type.value}"
        # return f"{CONSTELLATION_ABREV[self.constellation]}_{self.signal_type}_PRN{self.prn:02}"

    @staticmethod
    def from_string(s: str) -> "SignalId":
        system_prn, signal_type = s.split("_")
        system = SYSTEM_LETTERS_REVERSED[system_prn[0]]
        signal_type = SignalType[signal_type]
        prn = int(system_prn[1:])
        return SignalId(system, signal_type, prn)

    # we can use has for python version, but C or other versions might need a full enumeration
    def __hash__(self):
        return hash((self.system, self.signal_type, self.prn))

    def to_dtype_tuple(self) -> Tuple[str, str, int]:
        return (
            self.system.value.encode("ascii"),
            self.signal_type.value.encode("ascii"),
            self.prn,
        )


# h5py.string_dtype()
SIGNAL_ID_DTYPE = np.dtype(
    [
        ("system", np.byte, MAX_SYSTEM_TYPE_ENUM_STR_LENGTH),
        ("signal", np.byte, MAX_SIGNAL_TYPE_ENUM_STR_LENGTH),
        ("prn", np.int32),
    ]
)
