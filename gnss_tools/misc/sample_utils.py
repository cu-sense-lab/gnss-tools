from dataclasses import dataclass
import io
import enum
from typing import Optional, Tuple
import numpy as np
import numba as nb


@dataclass
class SampleParameters:
    """
    ------------------------------------------------------------------------------------------------
    `bit_depth` -- int, number of bits per numeric value (so an entire complex sample require `2*bit_depth` bits)
    `is_complex` -- bool, whether the samples are complex-valued (versus real-valued)
    `is_integer` -- bool, whether the numeric types are integer (versus float)
    `is_signed` -- bool (default True), whether the numeric types are signed (versus unsigned)
    `is_i_lsb` -- bool (default True), whether the I-component occupies the least significant bits of the sample stream
        E.g. for 4-bit samples, specifies whether the I component is in the least significant nibble
    `bytes_per_sample` -- int, number of bytes per sample
    """

    bit_depth: int
    is_complex: bool
    is_integer: bool
    is_signed: bool = True
    is_i_lsb: bool = True

    @property
    def bytes_per_sample(self) -> float:
        if self.is_complex:
            return (self.bit_depth * 2) / 8
        return self.bit_depth / 8


def get_numpy_dtype(
    is_integer: bool, is_signed: bool, bit_depth: int
) -> Optional[np.dtype]:

    match (is_integer, is_signed, bit_depth):
        case (True, True, 8):
            return np.int8
        case (True, True, 16):
            return np.int16
        case (True, True, 32):
            return np.int32
        # case (True, True, 64):
        #     return np.int64
        case (True, False, 8):
            return np.uint8
        case (True, False, 16):
            return np.uint16
        case (True, False, 32):
            return np.uint32
        case (True, False, 64):
            return np.uint64
        case (False, True, 16):
            return np.float16
        case (False, True, 32):
            return np.float32
        # case (False, True, 64):  # Note: we cannot deal with 64-bit sample components since we restrict ourselves to 32-bit sample buffers
        #     return np.float64
        case (_, _, _):
            return None


def compute_sample_array_size_bytes(
    num_samples: int,
    component_bit_depth: int,
    is_complex: bool,
) -> int:
    """
    Compute the number of bytes needes to store `num_samples` samples.
    This function works even for samples with bit depths less than 8 bits.
    """
    bits_per_sample = component_bit_depth * (2 if is_complex else 1)
    buffer_size_bytes = (num_samples * bits_per_sample) // 8
    if buffer_size_bytes * 8 != (num_samples * bits_per_sample):
        buffer_size_bytes += 1
    return buffer_size_bytes


def convert_to_complex64_samples(
    input_bytes: memoryview,
    output_bytes: memoryview,
    bit_depth: int,
    is_signed: bool,
    is_integer: bool,
    is_complex: bool,
    is_i_lsb: bool,
):

    # Determine sample component dtype
    input_component_dtype = get_numpy_dtype(is_integer, is_signed, bit_depth)
    bytes_per_input_component = bit_depth // 8
    bytes_per_input_sample = (
        2 * bytes_per_input_component if is_complex else bytes_per_input_component
    )
    assert int(len(output_bytes) // 8 * bytes_per_input_sample) == len(input_bytes)

    # Need to view as float32 (instead of complex64) so that we can handle I/Q ordering and real samples
    output_sample_array = np.frombuffer(output_bytes, dtype=np.float32)
    if input_component_dtype is not None:
        raw_sample_array = np.frombuffer(input_bytes, dtype=input_component_dtype)
        if is_complex:
            if is_i_lsb:
                output_sample_array[0::2] = raw_sample_array[1::2]
                output_sample_array[1::2] = raw_sample_array[0::2]
            else:
                output_sample_array[:] = raw_sample_array
        else:
            output_sample_array[::2] = np.frombuffer(
                input_bytes, dtype=input_component_dtype
            )
            output_sample_array[1::2] = 0
    elif bit_depth == 4:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        if is_complex:
            if is_i_lsb:
                if is_signed:
                    output_sample_array[0::2] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[1::2] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[0::2] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[1::2] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
            else:
                if is_signed:
                    output_sample_array[1::2] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[0::2] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4  # TODO is mask necessary?
                else:
                    output_sample_array[1::2] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[0::2] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
        else:
            if is_i_lsb:  # this means first sample is in least significant nibble
                if is_signed:
                    output_sample_array[0::4] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[2::4] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[0::4] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[2::4] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
            else:
                if is_signed:
                    output_sample_array[2::4] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[0::4] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[2::4] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[0::4] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
    elif bit_depth == 2:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        # 0xC0 0x30 0x0C 0x03
        if is_complex:
            if is_signed:
                if is_i_lsb:
                    output_sample_array[0::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.int8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.int8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.int8
                    ) >> 6
                    output_sample_array[3::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.int8
                    ) >> 6
                else:
                    output_sample_array[3::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.int8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.int8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.int8
                    ) >> 6
                    output_sample_array[0::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.int8
                    ) >> 6
            else:
                if is_i_lsb:
                    output_sample_array[0::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[3::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.uint8
                    ) >> 6
                else:
                    output_sample_array[3::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[0::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.uint8
                    ) >> 6
        else:
            raise NotImplemented()
    else:
        raise NotImplemented()


def convert_to_complex_int8_samples(
    input_bytes: memoryview,
    output_bytes: memoryview,
    bit_depth: int,
    is_signed: bool,
    is_integer: bool,
    is_complex: bool,
    is_i_lsb: bool,
):

    # Determine sample component dtype
    input_component_dtype = get_numpy_dtype(is_integer, is_signed, bit_depth)
    bytes_per_input_component = bit_depth / 8
    bytes_per_input_sample = (
        2 * bytes_per_input_component if is_complex else bytes_per_input_component
    )
    assert int(len(output_bytes) // 8 * bytes_per_input_sample) == len(input_bytes)

    # Need to view as float32 (instead of complex64) so that we can handle I/Q ordering and real samples
    output_sample_array = np.frombuffer(output_bytes, dtype=np.int8)
    if input_component_dtype is not None:
        # Note: we only support I, Q interleaving
        # `i_lsn` field is for 4-bit samples only
        raw_sample_array = np.frombuffer(input_bytes, dtype=input_component_dtype)
        if is_complex:
            if is_i_lsb:
                output_sample_array[0::2] = raw_sample_array[1::2]
                output_sample_array[1::2] = raw_sample_array[0::2]
            else:
                output_sample_array[:] = raw_sample_array
        else:
            output_sample_array[::2] = np.frombuffer(
                input_bytes, dtype=input_component_dtype
            )
            output_sample_array[1::2] = 0
    elif bit_depth == 4:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        if is_complex:
            if is_i_lsb:
                if is_signed:
                    output_sample_array[0::2] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[1::2] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[0::2] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[1::2] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
            else:
                if is_signed:
                    output_sample_array[1::2] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[0::2] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4  # TODO is mask necessary?
                else:
                    output_sample_array[1::2] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[0::2] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
        else:
            if is_i_lsb:  # this means first sample is in least significant nibble
                if is_signed:
                    output_sample_array[0::4] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[2::4] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[0::4] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[2::4] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
            else:
                if is_signed:
                    output_sample_array[2::4] = (sample_byte_array << 4).view(
                        np.int8
                    ) >> 4
                    output_sample_array[0::4] = (sample_byte_array << 0).view(
                        np.int8
                    ) >> 4
                else:
                    output_sample_array[2::4] = (sample_byte_array << 4).view(
                        np.uint8
                    ) >> 4
                    output_sample_array[0::4] = (sample_byte_array << 0).view(
                        np.uint8
                    ) >> 4
    elif bit_depth == 2:
        sample_byte_array = np.frombuffer(input_bytes, dtype=np.byte)
        # 0xC0 0x30 0x0C 0x03
        if is_complex:
            if is_signed:
                if is_i_lsb:
                    output_sample_array[0::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.int8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.int8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.int8
                    ) >> 6
                    output_sample_array[3::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.int8
                    ) >> 6
                else:
                    output_sample_array[3::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.int8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.int8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.int8
                    ) >> 6
                    output_sample_array[0::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.int8
                    ) >> 6
            else:
                if is_i_lsb:
                    output_sample_array[0::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[3::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.uint8
                    ) >> 6
                else:
                    output_sample_array[3::4] = ((sample_byte_array & 0x03) << 6).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[2::4] = ((sample_byte_array & 0x0C) << 4).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[1::4] = ((sample_byte_array & 0x30) << 2).view(
                        np.uint8
                    ) >> 6
                    output_sample_array[0::4] = ((sample_byte_array & 0xC0) << 0).view(
                        np.uint8
                    ) >> 6
        else:
            raise NotImplemented()
    else:
        raise NotImplemented()


### Sample loading
class SampleBufferDataType(enum.Enum):
    COMPLEX_FLOAT32 = 0
    INTERLEAVED_COMPLEX_INT8 = 1


SAMPLE_SIZE_BYTES = {
    SampleBufferDataType.COMPLEX_FLOAT32: 8,
    SampleBufferDataType.INTERLEAVED_COMPLEX_INT8: 2,
}

class SampleLoader:
    # The loader stores different buffers
    # When we load samples, we can copy them to the specified buffers
    # We load the max number of available samples that fit the largest buffer
    def __init__(
        self,
        raw_sample_params: SampleParameters,
        samp_rate: int,
        output_sample_dtype: SampleBufferDataType = SampleBufferDataType.COMPLEX_FLOAT32,
        max_buffer_duration_ms: int = 100,
    ) -> None:
        self.raw_sample_params = raw_sample_params
        self.samp_rate = samp_rate
        self.output_sample_dtype = output_sample_dtype

        self.max_buffer_duration_ms = max_buffer_duration_ms
        self.max_buffer_length_samples = int(
            self.max_buffer_duration_ms * self.samp_rate / 1000
        )
        max_input_buffer_size_bytes = compute_sample_array_size_bytes(
            self.max_buffer_length_samples,
            raw_sample_params.bit_depth,
            raw_sample_params.is_complex,
        )
        self._byte_buffer = bytearray(max_input_buffer_size_bytes)
        self._byte_buffer_view = memoryview(self._byte_buffer)

        self.sample_size_bytes = SAMPLE_SIZE_BYTES[output_sample_dtype]
        max_sample_buffer_size_bytes = self.max_buffer_length_samples * self.sample_size_bytes
        self._sample_buffer = bytearray(max_sample_buffer_size_bytes)
        self._sample_buffer_view = memoryview(self._sample_buffer)    

    def load_samples(self, file: io.BufferedReader, sample_index: int, num_samples: int) -> Optional[np.ndarray]:
        # Load samples from file into buffer
        if num_samples > self.max_buffer_length_samples:
            raise ValueError(f"Requested too many samples: {num_samples} > {self.max_buffer_length_samples}")
        file.seek(int(sample_index * self.raw_sample_params.bytes_per_sample))
        num_bytes_to_read = compute_sample_array_size_bytes(
            num_samples, self.raw_sample_params.bit_depth, self.raw_sample_params.is_complex
        )
        if num_bytes_to_read > len(self._byte_buffer):
            raise ValueError(f"Requested too many bytes: {num_bytes_to_read} > {len(self._byte_buffer)}")
        num_bytes_read = file.readinto(self._byte_buffer_view[:num_bytes_to_read])
        if num_bytes_read < num_bytes_to_read:
            # logging.warning("Failed to read enough bytes from file")
            return None
        num_samples_read = int(num_bytes_read // self.raw_sample_params.bytes_per_sample)
        if num_samples_read < num_samples:
            raise ValueError(f"Failed to read enough samples: {num_samples_read} < {num_samples}")
        
        if self.output_sample_dtype == SampleBufferDataType.COMPLEX_FLOAT32:
            convert_to_complex64_samples(
                memoryview(self._byte_buffer[:num_bytes_read]),
                self._sample_buffer_view[:num_samples_read * 8],
                self.raw_sample_params.bit_depth,
                self.raw_sample_params.is_signed,
                self.raw_sample_params.is_integer,
                self.raw_sample_params.is_complex,
                self.raw_sample_params.is_i_lsb,
            )
            return np.frombuffer(self._sample_buffer[:num_samples_read * 8], dtype=np.complex64)
        elif self.output_sample_dtype == SampleBufferDataType.INTERLEAVED_COMPLEX_INT8:
            convert_to_complex_int8_samples(
                memoryview(self._byte_buffer[:num_bytes_read]),
                self._sample_buffer_view[:num_samples_read * 2],
                self.raw_sample_params.bit_depth,
                self.raw_sample_params.is_signed,
                self.raw_sample_params.is_integer,
                self.raw_sample_params.is_complex,
                self.raw_sample_params.is_i_lsb,
            )
            return np.frombuffer(self._sample_buffer[:num_samples_read * 2], dtype=np.int8)
        else:
            raise ValueError("Invalid buffer data type")


def compute_cycles_from_sample_count(
    sample_count: int, samp_rate: int, freq_hz: float
) -> Tuple[int, float]:
    integer_seconds, excess_samples = divmod(sample_count, samp_rate)
    fractional_seconds = excess_samples / samp_rate
    integer_freq_hz, fractional_freq_hz = divmod(freq_hz, 1)
    integer_phase_cycles = integer_seconds * integer_freq_hz
    fractional_phase_cycles = (
        fractional_seconds * integer_freq_hz
        + fractional_seconds * fractional_freq_hz
        + integer_seconds * fractional_freq_hz
    )
    delta_integer_phase_cycles, fractional_phase_cycles = divmod(
        fractional_phase_cycles, 1
    )
    integer_phase_cycles += int(delta_integer_phase_cycles)
    return integer_phase_cycles, fractional_phase_cycles


@nb.njit
def mixdown_samples(
    input_samples: nb.complex64[:],
    output_samples: nb.complex64[:],
    mixdown_phase_cycles: nb.float64,
    mixdown_freq_hz: nb.float64,
    samp_rate: nb.float64,
) -> None:
    num_samples = len(input_samples)
    assert len(output_samples) >= num_samples
    mixdown_mult = np.exp(-2j * np.pi * mixdown_freq_hz / samp_rate)
    mixdown_carrier = np.exp(-2j * np.pi * mixdown_phase_cycles)
    for i in range(num_samples):
        output_samples[i] = input_samples[i] * mixdown_carrier
        mixdown_carrier = mixdown_carrier * mixdown_mult


def epl_correlate_bpsk(
        samples: np.ndarray[np.complex64],
        samp_rate: float,
        code_seq: np.ndarray[np.int8],
        code_length: int,
        code_phase_chips: float,
        code_rate_chips_per_sec: float,
        carr_phase_cycles: float,
        carr_rate_hz: float,
        chip_bin_spacing: float,
) -> np.ndarray[np.complex64]:
    corr_values = np.zeros(3, dtype=np.complex64)
    center_chip = code_phase_chips % code_length
    chip_delta = code_rate_chips_per_sec / samp_rate
    conj_carr_sample = np.exp(-2j * np.pi * carr_phase_cycles)
    conj_carr_rotation = np.exp(-2j * np.pi * carr_rate_hz / samp_rate)
    numba_correlate__bpsk__complex64(
        samples,
        code_seq,
        code_length,
        center_chip,
        chip_delta,
        3,
        -chip_bin_spacing,
        chip_bin_spacing,
        conj_carr_sample,
        conj_carr_rotation,
        corr_values,
    )
    return corr_values

@nb.jit(nopython=True, parallel=False)
def numba_correlate__bpsk__complex64(
        samples: nb.complex64[:],
        code_seq: nb.int8[:],
        code_length: nb.int32,
        chip_start: nb.float32,
        chip_delta: nb.float32,
        num_bins: nb.int32,
        chip_bin_offset: nb.float32,
        chip_bin_spacing: nb.float32,
        conj_carr_sample: nb.complex64,
        conj_carr_rotation: nb.complex64,
        corr_values: nb.complex64[:]
    ):  
    
    num_samples = len(samples)
    center_chip = chip_start
    for i in range(num_samples):
        carrierless = samples[i] * conj_carr_sample
        for j in range(num_bins):
            symbol = code_seq[int(center_chip + chip_bin_offset + j * chip_bin_spacing) % code_length]
            if symbol == 1:
                corr_values[j] += carrierless
            elif symbol == -1: 
                corr_values[j] -= carrierless
            elif symbol != 0:
                corr_values[j] += carrierless * symbol
        conj_carr_sample *= conj_carr_rotation
        center_chip += chip_delta

