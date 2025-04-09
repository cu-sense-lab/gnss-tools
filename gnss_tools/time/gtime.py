"""
Author: Brian Breitsch
Date: 2025-01-02
"""

from dataclasses import dataclass
from typing import Any, Tuple, Type, TypeVar, Union

import numpy as np
import numpy.typing as npt
import numba as nb

GTIME_DTYPE = np.dtype(
    [
        ("whole_seconds", np.int32),
        ("frac_seconds", np.float64),
    ]
)


@dataclass(slots=True)
# @numba.experimental.jitclass
class GTime:
    whole_seconds: int
    frac_seconds: float

    def __init__(self, whole_seconds: int, frac_seconds: float):
        self.whole_seconds = whole_seconds
        self.frac_seconds = frac_seconds
        # assert(0 <= self.frac_seconds and self.frac_seconds < 1)

    def __post_init__(self):
        if self.frac_seconds < 0 or self.frac_seconds >= 1:
            raise ValueError("frac_seconds must be in range [0, 1)")

    def __repr__(self) -> str:
        return f"GTime({self.whole_seconds}, {self.frac_seconds:6.5f})"

    def __str__(self) -> str:
        frac_str = f"{self.frac_seconds:5.5f}"[1:]  # remove leading 0
        return f"T({self.whole_seconds}{frac_str})"

    def __add__(self, other: "GTime") -> "GTime":
        whole_seconds = self.whole_seconds + other.whole_seconds
        frac_seconds = self.frac_seconds + other.frac_seconds
        if frac_seconds >= 1:
            return GTime(whole_seconds + 1, frac_seconds - 1)
        else:
            return GTime(whole_seconds, frac_seconds)

    def __sub__(self, other: "GTime") -> "GTime":
        frac_seconds = self.frac_seconds - other.frac_seconds
        whole_seconds = self.whole_seconds - other.whole_seconds
        # GTime validity constraint guarantees that self.frac_seconds - other.frac_seconds
        # is in range (-1, 1)
        if frac_seconds < 0:
            return GTime(whole_seconds - 1, frac_seconds + 1)
        else:
            return GTime(whole_seconds, frac_seconds)

    def __neg__(self) -> "GTime":
        # NOTE: We want to keep frac_seconds in range [0, 1)
        if self.frac_seconds == 0.0:
            return GTime(-self.whole_seconds, 0.0)
        return GTime(-self.whole_seconds - 1, -self.frac_seconds + 1)

    def __eq__(self, other: "GTime") -> bool:
        return (
            self.whole_seconds == other.whole_seconds
            and self.frac_seconds == other.frac_seconds
        )

    def __lt__(self, other: "GTime") -> bool:
        return self.whole_seconds < other.whole_seconds or (
            self.whole_seconds == other.whole_seconds
            and self.frac_seconds < other.frac_seconds
        )

    def __le__(self, other: "GTime") -> bool:
        return self.whole_seconds < other.whole_seconds or (
            self.whole_seconds == other.whole_seconds
            and self.frac_seconds <= other.frac_seconds
        )

    def __gt__(self, other: "GTime") -> bool:
        return self.whole_seconds > other.whole_seconds or (
            self.whole_seconds == other.whole_seconds
            and self.frac_seconds > other.frac_seconds
        )

    def __ge__(self, other: "GTime") -> bool:
        return self.whole_seconds > other.whole_seconds or (
            self.whole_seconds == other.whole_seconds
            and self.frac_seconds >= other.frac_seconds
        )

    def add_integer_seconds(self, seconds: int) -> "GTime":
        return GTime(self.whole_seconds + seconds, self.frac_seconds)

    def add_float_seconds(self, seconds: float) -> "GTime":
        integer_seconds = int(seconds)
        fractional_seconds = seconds - integer_seconds
        whole_seconds = self.whole_seconds + integer_seconds
        fractional_seconds = self.frac_seconds + fractional_seconds
        if fractional_seconds >= 1:
            return GTime(whole_seconds + 1, fractional_seconds - 1)
        elif fractional_seconds < 0:
            return GTime(whole_seconds - 1, fractional_seconds + 1)
        else:
            return GTime(whole_seconds, fractional_seconds)

    def to_float(self) -> float:
        return float(self.whole_seconds) + self.frac_seconds

    def to_tuple(self) -> Tuple[int, float]:
        return (self.whole_seconds, self.frac_seconds)

    def to_tick_count_with_float_rate(self, tick_rate: float) -> Tuple[int, float]:
        """
        Computes the number of ticks at the given rate that this time represents.
        Returns a tuple of (whole_ticks, frac_ticks) where frac_ticks is in range [0, 1)
        """
        rate_i: int = int(tick_rate)
        rate_f: float = tick_rate - rate_i
        whole_ticks = self.whole_seconds * rate_i
        delta_ticks = self.whole_seconds * rate_f + self.frac_seconds * tick_rate
        delta_whole_ticks = int(delta_ticks)
        frac_ticks = delta_ticks - delta_whole_ticks
        whole_ticks += delta_whole_ticks
        return (whole_ticks, frac_ticks)

    def to_tick_count_with_integer_rate(self, tick_rate: int) -> Tuple[int, float]:
        """
        Computes the number of ticks at the given rate that this time represents.
        Returns a tuple of (whole_ticks, frac_ticks) where frac_ticks is in range [0, 1)
        """
        delta_ticks = self.frac_seconds * tick_rate
        delta_whole_ticks = int(delta_ticks)
        whole_ticks = self.whole_seconds * tick_rate + delta_whole_ticks
        frac_ticks = delta_ticks - delta_whole_ticks
        return (int(whole_ticks), frac_ticks)

    
    @classmethod
    def from_tick_count_with_float_rate(cls, tick_count: int, ticks_per_second: float) -> "GTime":
        """
        Computes the time that the given tick count represents at the given rate.
        Returns a GTime object representing the time.

        NOTE: this function is derived from the UHD time_spec_t::from_ticks function.
        """
        integer_rate: int = int(ticks_per_second)
        if integer_rate == 0:
            seconds_per_tick = 1.0 / ticks_per_second
            whole_seconds_per_tick = int(seconds_per_tick)
            fractional_seconds_per_tick = seconds_per_tick - whole_seconds_per_tick
            whole_seconds = tick_count * whole_seconds_per_tick
            error_seconds = tick_count * fractional_seconds_per_tick
            integer_error_seconds = int(error_seconds)
            whole_seconds += integer_error_seconds
            fractional_seconds = error_seconds - integer_error_seconds
            if fractional_seconds < 0:
                return cls(whole_seconds - 1, fractional_seconds + 1.0)
            else:
                return cls(whole_seconds, fractional_seconds)
        else:
            fractional_rate: float = ticks_per_second - integer_rate
            whole_seconds: int = int(tick_count // integer_rate)
            integer_tick_error = tick_count - whole_seconds * integer_rate
            tick_error = integer_tick_error - whole_seconds * fractional_rate
            error_seconds = tick_error / ticks_per_second
            integer_error_seconds = int(error_seconds)
            whole_seconds += integer_error_seconds
            fractional_seconds = error_seconds - integer_error_seconds
            if fractional_seconds < 0:
                return cls(whole_seconds - 1, fractional_seconds + 1.0)
            else:
                return cls(whole_seconds, fractional_seconds)
    
    
    @classmethod
    def from_tick_count_with_integer_rate(cls, tick_count: int, ticks_per_second: int) -> "GTime":
        """
        Computes the time that the given tick count represents at the given rate.
        Returns a GTime object representing the time.
        """
        whole_seconds: int = int(tick_count // ticks_per_second)
        tick_error = tick_count - (whole_seconds * ticks_per_second)
        frac_seconds = tick_error / ticks_per_second
        if frac_seconds < 0:
            return cls(whole_seconds - 1, frac_seconds + 1.0)
        else:
            return cls(whole_seconds, frac_seconds)


    T = TypeVar("T", bound="GTime")
    @classmethod
    def from_float_seconds(cls: Type[T], seconds: float) -> T:
        whole_seconds = int(seconds)
        frac_seconds = seconds - whole_seconds
        if frac_seconds < 0.0:
            return cls(whole_seconds - 1, frac_seconds + 1.0)
        else:
            return cls(whole_seconds, frac_seconds)
    
    
    @classmethod
    def from_integer_milliseconds(cls: Type[T], milliseconds: int) -> T:
        whole_seconds = int(milliseconds // 1000)
        frac_seconds = (milliseconds - 1000 * whole_seconds) * 0.001
        if frac_seconds < 0.0:
            return cls(whole_seconds - 1, frac_seconds + 1.0)
        else:
            return cls(whole_seconds, frac_seconds)
    


    ## Array methods
    @classmethod
    def from_float_seconds_array(cls: Type[T], seconds: npt.NDArray) -> npt.NDArray:
        """
        Converts an array of floats representing seconds into a numpy array with GTIME_DTYPE.
        """
        outputs = np.zeros(seconds.shape, dtype=GTIME_DTYPE)
        outputs["whole_seconds"] = seconds.astype(np.int32)
        outputs["frac_seconds"] = seconds - outputs["whole_seconds"]
        return outputs

    @classmethod
    def from_tick_count_array_with_integer_rate(cls: Type[T], tick_counts: npt.NDArray, ticks_per_second: int) -> npt.NDArray:
        """
        Computes the time that the given tick count represents at the given rate.
        Returns a numpy array of GTIME dtype representing the times.
        """
        output = np.zeros(tick_counts.shape, dtype=GTIME_DTYPE)
        output['whole_seconds'] = tick_counts // ticks_per_second
        tick_error = tick_counts - (output['whole_seconds'] * ticks_per_second)
        output['frac_seconds'] = tick_error / ticks_per_second
        ind = output['frac_seconds'] < 0
        output['frac_seconds'][ind] += 1.0
        output['whole_seconds'][ind] -= 1
        return output


@nb.njit
def numba_gtime_to_tick_count_array_with_integer_rate(
    x_whole_seconds: nb.int32[:],
    x_frac_seconds: nb.float64[:],
    rate: int,
) -> Tuple[nb.int32[:], nb.float64[:]]:
    tick_count = x_whole_seconds * rate
    residual = x_frac_seconds * rate
    delta_tick_count = np.floor(residual)
    tick_count = tick_count + delta_tick_count
    frac_tick_count = residual - delta_tick_count
    return tick_count, frac_tick_count


@nb.njit
def numba_gtime_interpolate_float64(
    x_whole_seconds: nb.int32[:],
    x_frac_seconds: nb.float64[:],
    xp_whole_seconds: nb.int32[:],
    xp_frac_seconds: nb.float64[:],
    fp: nb.float64[:],
) -> nb.float64[:]:
    num_inputs = len(xp_whole_seconds)
    num_outputs = len(x_whole_seconds)
    assert num_inputs > 1
    xp_deltas = (xp_whole_seconds[1:] - xp_whole_seconds[:-1]) + (
        xp_frac_seconds[1:] - xp_frac_seconds[:-1]
    )

    f = np.zeros(num_outputs, dtype=np.float64)
    fp_deltas = fp[1:] - fp[:-1]

    i = 0
    while (i < num_outputs) and (
        (x_whole_seconds[i] < xp_whole_seconds[0])
        or (
            (x_whole_seconds[i] == xp_whole_seconds[0])
            and (x_frac_seconds[i] < xp_frac_seconds[0])
        )
    ):
        f[i] = fp[0]

    head = 0
    while (i < num_outputs) and (head < num_inputs - 1):
        if (x_whole_seconds[i] < xp_whole_seconds[head + 1]) or (
            (x_whole_seconds[i] == xp_whole_seconds[head + 1])
            and (x_frac_seconds[i] < xp_frac_seconds[head + 1])
        ):
            delta = (
                x_whole_seconds[i]
                - xp_whole_seconds[head]
                + x_frac_seconds[i]
                - xp_frac_seconds[head]
            )
            norm_delta = delta / xp_deltas[head]
            f[i] = fp[head] + norm_delta * fp_deltas[head]
            i += 1
        else:
            head += 1

    while i < num_outputs:
        f[i] = fp[-1]
        i += 1

    return f


def gtime_interpolate_float64(
    x: npt.NDArray[GTIME_DTYPE],
    xp: npt.NDArray[GTIME_DTYPE],
    fp: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return numba_gtime_interpolate_float64(
        x["whole_seconds"],
        x["frac_seconds"],
        xp["whole_seconds"],
        xp["frac_seconds"],
        fp,
    )


@nb.njit
def numba_gtime_interpolate_gtime(
    x_whole_seconds: nb.int32[:],
    x_frac_seconds: nb.float64[:],
    xp_whole_seconds: nb.int32[:],
    xp_frac_seconds: nb.float64[:],
    fp_whole_seconds: nb.int32[:],
    fp_frac_seconds: nb.float64[:],
) -> Tuple[nb.int32[:], nb.float64[:]]:
    num_inputs = len(xp_whole_seconds)
    num_outputs = len(x_whole_seconds)
    assert num_inputs > 1
    xp_deltas = (xp_whole_seconds[1:] - xp_whole_seconds[:-1]) + (
        xp_frac_seconds[1:] - xp_frac_seconds[:-1]
    )

    f_whole_seconds = np.zeros(num_outputs, dtype=np.int32)
    f_frac_seconds = np.zeros(num_outputs, dtype=np.float64)
    fp_deltas = (fp_whole_seconds[1:] - fp_whole_seconds[:-1]) + (
        fp_frac_seconds[1:] - fp_frac_seconds[:-1]
    )

    i = 0
    while (i < num_outputs) and (
        (x_whole_seconds[i] < xp_whole_seconds[0])
        or (
            (x_whole_seconds[i] == xp_whole_seconds[0])
            and (x_frac_seconds[i] < xp_frac_seconds[0])
        )
    ):
        f_whole_seconds[i] = fp_whole_seconds[0]
        f_frac_seconds[i] = fp_frac_seconds[0]
        i += 1

    head = 0
    while (i < num_outputs) and (head < num_inputs - 1):
        if (x_whole_seconds[i] < xp_whole_seconds[head + 1]) or (
            (x_whole_seconds[i] == xp_whole_seconds[head + 1])
            and (x_frac_seconds[i] < xp_frac_seconds[head + 1])
        ):
            delta = (
                x_whole_seconds[i]
                - xp_whole_seconds[head]
                + x_frac_seconds[i]
                - xp_frac_seconds[head]
            )
            norm_delta = delta / xp_deltas[head]
            f_frac_seconds[i] = fp_frac_seconds[head] + norm_delta * fp_deltas[head]
            delta_whole_seconds = int(f_frac_seconds[i])
            f_frac_seconds[i] -= delta_whole_seconds
            f_whole_seconds[i] = fp_whole_seconds[head] + delta_whole_seconds
            i += 1
        else:
            head += 1

    while i < num_outputs:
        f_whole_seconds[i] = fp_whole_seconds[-1]
        f_frac_seconds[i] = fp_frac_seconds[-1]
        i += 1

    return f_whole_seconds, f_frac_seconds


def gtime_interpolate_gtime(
    x: npt.NDArray[GTIME_DTYPE],
    xp: npt.NDArray[GTIME_DTYPE],
    fp: npt.NDArray[GTIME_DTYPE],
) -> npt.NDArray[GTIME_DTYPE]:
    f = np.zeros(len(x), dtype=GTIME_DTYPE)
    f["whole_seconds"][:], f["frac_seconds"][:] = numba_gtime_interpolate_gtime(
        x["whole_seconds"],
        x["frac_seconds"],
        xp["whole_seconds"],
        xp["frac_seconds"],
        fp["whole_seconds"],
        fp["frac_seconds"],
    )
    return f


nb.njit


def numba_interpolate_gtime(
    x: nb.float64[:],
    xp: nb.float64[:],
    fp_whole_seconds: nb.int32[:],
    fp_frac_seconds: nb.float64[:],
) -> Tuple[npt.NDArray, npt.NDArray]:
    num_inputs = len(xp)
    num_outputs = len(x)
    assert num_inputs > 1
    xp_deltas = xp[1:] - xp[:-1]

    f_whole_seconds = np.zeros(num_outputs, dtype=np.int32)
    f_frac_seconds = np.zeros(num_outputs, dtype=np.float64)
    fp_deltas = (fp_whole_seconds[1:] - fp_whole_seconds[:-1]) + (
        fp_frac_seconds[1:] - fp_frac_seconds[:-1]
    )

    i = 0
    while (i < num_outputs) and (x[i] < xp[0]):
        f_whole_seconds[i] = fp_whole_seconds[0]
        f_frac_seconds[i] = fp_frac_seconds[0]
        i += 1

    head = 0
    while (i < num_outputs) and (head < num_inputs - 1):
        if x[i] < xp[head + 1]:
            delta = x[i] - xp[head]
            norm_delta = delta / xp_deltas[head]
            f_frac_seconds[i] = fp_frac_seconds[head] + norm_delta * fp_deltas[head]
            delta_whole_seconds = int(f_frac_seconds[i])
            f_frac_seconds[i] -= delta_whole_seconds
            f_whole_seconds[i] = fp_whole_seconds[head] + delta_whole_seconds
            i += 1
        else:
            head += 1

    while i < num_outputs:
        f_whole_seconds[i] = fp_whole_seconds[-1]
        f_frac_seconds[i] = fp_frac_seconds[-1]
        i += 1

    return f_whole_seconds, f_frac_seconds


def interpolate_gtime(
    x: np.ndarray[Any, np.float64],
    xp: npt.NDArray[np.float64],
    fp: np.ndarray[Any, GTIME_DTYPE],
) ->  np.ndarray[Any, GTIME_DTYPE]:
    f = np.zeros(len(x), dtype=GTIME_DTYPE)
    f["whole_seconds"][:], f["frac_seconds"][:] = numba_interpolate_gtime(
        x,
        xp,
        fp["whole_seconds"],
        fp["frac_seconds"],
    )
    return f
