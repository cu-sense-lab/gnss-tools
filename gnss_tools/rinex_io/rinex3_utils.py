"""
Author Brian Breitsch
Date: 2025-01-02
"""

from typing import Dict, Any, Iterable, List, Tuple, Optional

# RINEX 3.03

OBSERVATION_LETTERS = {
    "C": "pseudorange",
    "L": "carrier",
    "D": "doppler",
    "S": "cnr",
}

SYSTEM_ID_TO_NAME_MAP = {
    "GPS": "GPS",
    "GLO": "GLONASS",
    "GAL": "Galileo",
    "QZS": "QZSS",
    "BDS": "Beidou", # Beidou
    "IRS": "IRNSS",
    "SBS": "SBAS",
}

SYSTEM_LETTER_TO_ID_MAP = {
    "G": "GPS",
    "R": "GLO",
    "E": "GAL",
    "J": "QZS",
    "C": "BDS", # Beidou
    "I": "IRS",
    "S": "SBS",
}

BAND_AND_CHANNEL_INFO = {
    "GPS": {
        "1": {
            "band_name": "L1",
            "frequency": 1575.42e6,
            "channel_names": {
                "C": "C/A",
                "S": "L1C(D)",
                "L": "L1C(P)",
                "X": "L1C (D+P)",
                "P": "P (AS off)",
                "W": "Z-tracking and similar(AS on)",
                "Y": "Y",
                "M": "M",
                "N": "codeless",
            },
        },
        "2": {
            "band_name": "L2",
            "frequency": 1227.60e6,
            "channel_names": {
                "C": "C/A",
                "D": "L1(C/A)+(P2-P1)(semi-codeless)",
                "S": "L2C (M)",
                "L": "L2C (L)",
                "X": "L2C (M+L)",
                "P": "P (AS off)",
                "W": "Z-tracking and similar(AS on)",
                "Y": "Y",
                "M": "M",
                "N": "codeless",
            },
        },
        "5": {
            "band_name": "L5",
            "frequency": 1176.45e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
    },
    "GLO": {
        "1": {
            "band_name": "G1",
            "frequency": lambda k: 1602e6 + k * (9 / 16) * 1e3,
            "channel_names": {"C": "C/A", "P": "P"},
        },
        "2": {
            "band_name": "G2",
            "frequency": lambda k: 1246e6 + k * 716 * 1e3,
            "channel_names": {"C": "C/A(GLONASS M)", "P": "P"},
        },
        "3": {
            "band_name": "G3",
            "frequency": 1202.025e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
    },
    "GAL": {
        "1": {
            "band_name": "E1",
            "frequency": 1575.42e6,
            "channel_names": {
                "A": "A PRS",
                "B": "B I/NAV OS/CS/SoL",
                "C": "C no data",
                "X": "B+C",
                "Z": "A+B+C",
            },
        },
        "5": {
            "band_name": "E5a",
            "frequency": 1176.45e6,
            "channel_names": {"I": "I F/NAV OS", "Q": "Q no data", "X": "I+Q"},
        },
        "7": {
            "band_name": "E5b",
            "frequency": 1207.140e6,
            "channel_names": {"I": "I F/NAV OS/CS/SoL", "Q": "Q no data", "X": "I+Q"},
        },
        "8": {
            "band_name": "E5",
            "frequency": 1191.795e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
        "6": {
            "band_name": "E6",
            "frequency": 1278.75e6,
            "channel_names": {
                "A": "A PRS",
                "B": "B C/NAV CS",
                "C": "C no data",
                "X": "B+C",
                "Z": "A+B+C",
            },
        },
    },
    "BDS": {
        "2": {
            "band_name": "B1",
            "frequency": 1561.098e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
        "7": {
            "band_name": "B2",
            "frequency": 1207.14e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
        "6": {
            "band_name": "B3",
            "frequency": 1268.52e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
        # redundancy to support older RINEX 3.02 versions:
        "1": {
            "band_name": "B1",
            "frequency": 1561.098e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
    },
    "SBS": {
        "1": {
            "band_name": "L1",
            "frequency": 1575.42e6,
            "channel_names": {
                "C": "C/A",
            },
        },
        "5": {
            "band_name": "L5",
            "frequency": 1176.45e6,
            "channel_names": {"I": "I", "Q": "Q", "X": "I+Q"},
        },
    },
    "QZS": {
        "1": {
            "band_name": "L1",
            "frequency": 1575.42e6,
            "channel_names": {
                "C": "C/A",
                "S": "L1C (D)",
                "L": "L1C (P)",
                "X": "L1C (D+P)",
                "Z": "L1-SAIF",
            },
        },
        "2": {
            "band_name": "L2",
            "frequency": 1227.60e6,
            "channel_names": {
                "S": "L2C (M)",
                "L": "L2C (L)",
                "X": "L2C (M+L)",
            },
        },
        "5": {
            "band_name": "L5",
            "frequency": 1176.45e6,
            "channel_names": {
                "I": "I",
                "Q": "Q",
                "X": "I+Q",
            },
        },
        "6": {
            "band_name": "LEX",
            "frequency": 1278.75e6,
            "channel_names": {
                "S": "S",
                "L": "L",
                "X": "S+L",
            },
        },
    },
    "IRS": {
        "5": {
            "band_name": "L5",
            "frequency": 1176.45e6,
            "channel_names": {
                "A": "A SPS",
                "B": "B RS (D)",
                "C": "C RS (P)",
                "X": "B+C",
            },
        },
        "9": {
            "band_name": "S",
            "frequency": 2492.028e6,
            "channel_names": {
                "A": "A SPS",
                "B": "B RS (D)",
                "C": "C RS (P)",
            },
        },
    },
}

DUAL_FREQ_BAND_PREFERENCES = {
    "GPS": [(1, 2)],
    "GLO": [],
    "GAL": [(1, 5)],
    "BDS": [(1, 5), (2, 5), (1, 6), (2, 6)],
    "SBS": [],
    "QZS": [],
    "IRS": [],
}

CHANNEL_PREFERENCES = {
    "GPS": {
        "1": ["X", "L", "S", "C", "P", "W", "Y", "M", "N"],
        "2": ["X", "L", "S", "C", "Y", "M", "D", "W", "N"],
        "5": ["X", "Q", "I"],
    },
    "GLO": {
        "1": ["C", "P"],
        "4": ["X", "A", "B"],
        "2": ["C", "P"],
        "6": ["X", "A", "B"],
        "3": ["X", "Q", "I"],
    },
    "GAL": {
        "1": ["X", "Z", "A", "B", "C"],
        "5": ["X", "Q", "I"],
        "7": ["X", "Q", "I"],
        "8": ["X", "Q", "I"],
        "6": ["X", "Z", "A", "B", "C"],
    },
    "SBS": {"1": ["C"], "5": ["X", "Q", "I"]},
    "QZS": {
        "L1": ["X", "L", "S", "C", "Z"],
        "L2": ["X", "L", "S"],
        "L5": ["X", "Q", "I"],
        "LEX": ["X", "L", "S"],
    },
    "BDS": {
        "2": ["X", "Q", "I"],
        "1": ["X", "D", "P"],
        "5": ["X", "Q", "I"],
        "7": ["X", "Q", "I"],
        "8": ["X", "Q", "I"],
        "6": ["X", "Q", "I"],
    },
    "IRS": {"5": ["X", "A", "B", "C"], "9": ["A", "B", "C"]},
}

def create_preferred_obs_id_map(
    rinex_obs_data: Dict[str, Any], n_bands: int = 2  # sat_id -> {obs_id: np.ndarray}
) -> Dict[str, List[Tuple[str, str]]]:
    results: Dict[str, List[Optional[str]]] = {}  # sat_id -> ["{band_id}{channel_id}"]
    for sat_id, sat_data in rinex_obs_data.items():

        num_bands_found: int = 0
        band_numbers: List[Optional[str]] = [None for i in range(n_bands)]
        channel_letters: List[Optional[str]] = [None for i in range(n_bands)]

        system_letter = sat_id[0]
        system_id = SYSTEM_LETTER_TO_ID_MAP[system_letter]
        band_preferences = CHANNEL_PREFERENCES[system_id]
        for band_number, channel_preferences in band_preferences.items():
            # Check if band already found
            if band_number in band_numbers:
                continue
            # check that code and carrier meas present for given band / signal
            for channel_letter in channel_preferences:
                # print(f"\r {sat_id} {band_number} {channel_letter}", end="")
                carrier_obs_id = f"L{band_number}{channel_letter}"
                code_obs_id = f"C{band_number}{channel_letter}"
                if carrier_obs_id in sat_data and code_obs_id in sat_data:
                    band_numbers[num_bands_found] = band_number
                    channel_letters[num_bands_found] = channel_letter
                    num_bands_found += 1
                    break
            if num_bands_found >= n_bands:
                break
        # We found IDs for dual-frequency obs
        results[sat_id] = list(
            map(
                lambda x: "".join(x) if x[0] is not None else None,
                zip(band_numbers, channel_letters),
            )
        )
    return results


def get_band_frequencies(
    sat_id: str,
    obs_ids: Iterable[str],
    glonass_slot_number: Optional[Dict[str, int]] = None,
) -> List[float]:
    # Note: `obs_ids` could be either f"{band_number}{channel_letter}" or just f"{band_number}"
    #  Its entries just need to start with the appropriate band number
    freqs = []
    system_id = SYSTEM_LETTER_TO_ID_MAP[sat_id[0]]
    system_info = BAND_AND_CHANNEL_INFO[system_id]
    for obs_id in obs_ids:
        band_number = obs_id[0]
        if band_number not in system_info:
            freqs.append(None)
        else:
            freq = system_info[band_number]["frequency"]
            if system_id == "GLO":
                freq = freq(glonass_slot_number[sat_id])
            freqs.append(freq)
    return freqs