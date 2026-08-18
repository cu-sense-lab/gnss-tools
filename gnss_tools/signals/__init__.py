"""
GNSS signal code generators.

Currently implemented: GPS L1 C/A (`gps_l1ca`), GPS L2C (`gps_l2c`), GPS L5
(`gps_l5`), and a partial GPS L1C pilot code (`gps_l1c`, data channel missing).
GLONASS has only a code-generator stub (`glo`, and it has a known bug -- see
its docstring). Galileo and BeiDou have no code here yet.

`catalog` wraps the implemented GPS signals in a common `Link`/`Signal` shape
so future constellations have somewhere to register into rather than adding
another ad hoc module of loose constants and functions.
"""

from . import gps_l1ca, gps_l1c, gps_l2c, gps_l5, glo
from .catalog import LINKS, SIGNALS, Link, Signal, get_link, get_signal
