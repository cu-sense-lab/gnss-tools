"""
A minimal, common shape for "link" (RF carrier) and "signal" (a modulation on
that link) that every constellation's code generators can be registered under.

Scope note: this catalog currently only wraps the GPS signals that already have
working generators here (L1 C/A, L2C's CM/CL, L5's I/Q). Galileo, GLONASS,
BeiDou, GPS L1C's data channel, and GPS L2P are NOT yet represented -- their
code generators don't exist yet (see the per-module docstrings and the
project's TODO.md for what's missing). Adding a constellation later means
adding `Link`/`Signal` entries here plus the generator functions they wrap; it
does not require touching this file's shape.

`Signal.generate_code_sequence(prn)` intentionally keeps each constellation
module's own return convention (0/1, not +/-1) rather than normalizing it here,
so this catalog stays a thin index over existing functions instead of a second
implementation. Callers needing +/-1 int8 convert with
`(1 - 2 * seq).astype(np.int8)`, same as `gps-tracking-example`'s
`utils.signal_interfaces` already does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import gps_l1ca, gps_l2c, gps_l5


@dataclass(frozen=True)
class Link:
    """One RF carrier, shared by every signal modulated onto it."""

    id: str
    carrier_freq_hz: float
    description: str = ""


@dataclass(frozen=True)
class Signal:
    """
    One modulation on a `Link`: a spreading code plus the chip-rate/length that
    describe it, and (optionally) a secondary/overlay code and a data channel.
    """

    id: str
    link: Link
    chip_rate_hz: float
    code_length_chips: int
    generate_code_sequence: Callable[[int], np.ndarray]
    data_symbol_rate_hz: Optional[float] = None
    secondary_code: Optional[np.ndarray] = None
    description: str = ""


LINKS: dict[str, Link] = {
    "GPS_L1": Link(
        id="GPS_L1",
        carrier_freq_hz=gps_l1ca.CARRIER_FREQ,
        description="GPS L1 (1575.42 MHz): carries L1 C/A (implemented) and L1C (partial -- pilot code only, see gps_l1c.py)",
    ),
    "GPS_L2": Link(
        id="GPS_L2",
        carrier_freq_hz=gps_l2c.CARRIER_FREQ,
        description="GPS L2 (1227.6 MHz): carries L2C (implemented) and L2P(Y) (not implemented)",
    ),
    "GPS_L5": Link(
        id="GPS_L5",
        carrier_freq_hz=gps_l5.CARRIER_FREQ,
        description="GPS L5 (1176.45 MHz): carries L5 I/Q (implemented)",
    ),
}


SIGNALS: dict[str, Signal] = {
    "GPS_L1CA": Signal(
        id="GPS_L1CA",
        link=LINKS["GPS_L1"],
        chip_rate_hz=gps_l1ca.CODE_RATE,
        code_length_chips=gps_l1ca.CODE_LENGTH,
        generate_code_sequence=gps_l1ca.generate_code_sequence_L1CA,
        data_symbol_rate_hz=gps_l1ca.DATA_SYMBOL_RATE,
        description="GPS L1 C/A: 1.023 Mcps Gold code, 50 sps LNAV data",
    ),
    "GPS_L2CM": Signal(
        id="GPS_L2CM",
        link=LINKS["GPS_L2"],
        chip_rate_hz=gps_l2c.CODE_RATE_L2CM,
        code_length_chips=gps_l2c.CODE_LENGTH_L2CM,
        generate_code_sequence=gps_l2c.generate_code_sequence_L2CM,
        data_symbol_rate_hz=gps_l2c.DATA_SYMBOL_RATE,
        description="GPS L2C moderate-length code (data channel), 511.5 kcps, 50 sps CNAV data",
    ),
    "GPS_L2CL": Signal(
        id="GPS_L2CL",
        link=LINKS["GPS_L2"],
        chip_rate_hz=gps_l2c.CODE_RATE_L2CL,
        code_length_chips=gps_l2c.CODE_LENGTH_L2CL,
        generate_code_sequence=gps_l2c.generate_code_sequence_L2CL,
        description="GPS L2C long-length code (dataless pilot), 511.5 kcps",
    ),
    "GPS_L5I": Signal(
        id="GPS_L5I",
        link=LINKS["GPS_L5"],
        chip_rate_hz=gps_l5.CODE_RATE,
        code_length_chips=gps_l5.PRIMARY_CODE_LENGTH,
        generate_code_sequence=gps_l5.generate_code_sequence_L5I,
        data_symbol_rate_hz=gps_l5.DATA_SYMBOL_RATE,
        secondary_code=gps_l5.NEUMAN_HOFFMAN_SEQ_L5I,
        description="GPS L5 in-phase (data) component, 10.23 Mcps + NH10 overlay, 100 sps CNAV data",
    ),
    "GPS_L5Q": Signal(
        id="GPS_L5Q",
        link=LINKS["GPS_L5"],
        chip_rate_hz=gps_l5.CODE_RATE,
        code_length_chips=gps_l5.PRIMARY_CODE_LENGTH,
        generate_code_sequence=gps_l5.generate_code_sequence_L5Q,
        secondary_code=gps_l5.NEUMAN_HOFFMAN_SEQ_L5Q,
        description="GPS L5 quadrature (dataless pilot) component, 10.23 Mcps + NH20 overlay",
    ),
}


def get_link(link_id: str) -> Link:
    try:
        return LINKS[link_id]
    except KeyError:
        raise KeyError(f"no link {link_id!r}; have {sorted(LINKS)}") from None


def get_signal(signal_id: str) -> Signal:
    try:
        return SIGNALS[signal_id]
    except KeyError:
        raise KeyError(f"no signal {signal_id!r}; have {sorted(SIGNALS)}") from None
