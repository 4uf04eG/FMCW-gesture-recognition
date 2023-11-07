# ===========================================================================
# Copyright (C) 2021-2022 Infineon Technologies AG
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===========================================================================

"""Python wrapper for Infineon Radar CW (Constant wave)

The package expects the library (radar_sdk.dll on Windows, libradar_sdk.so on
Linux) either in the same directory as this file (ifxRadarSDK.py) or in a
subdirectory ../../libs/ARCH/ relative to this file where ARCH is depending on
the platform either win32_x86, win32_x64, raspi, or linux_x64.
"""
import sys
from pathlib import Path
_cur_dir = str(Path(__file__).parent)
if _cur_dir not in sys.path:
    sys.path.append(_cur_dir)

import types
from enum import IntEnum
from ctypes import *
from ifxError import *
from ifxRadarSDK import *
from ifxRadarSDK import dll
from ifxRadarSDK import check_rc
import numpy as np

__all__ = ["DeviceConstantWave", "DeviceADCTracking", "DeviceADCSampleTime", "DeviceADCOversampling",
           "DeviceBasebandVgaGain", "DeviceBasebandHpGain",
           "DeviceTestSignalGeneratorMode", "create_adc_configs_struct", "create_baseband_configs_struct",
           "create_test_signal_configs_struct"]

#types

class DeviceADCTracking(IntEnum):
    ADC_NO_SUBCONVERSIONS = 0
    ADC_1_SUBCONVERSIONS  = 1
    ADC_3_SUBCONVERSIONS  = 2
    ADC_7_SUBCONVERSIONS  = 3

class DeviceADCSampleTime(IntEnum):
    ADC_SAMPLETIME_50NS  = 0
    ADC_SAMPLETIME_100NS = 1
    ADC_SAMPLETIME_200NS = 2
    ADC_SAMPLETIME_400NS = 3

class DeviceADCOversampling(IntEnum):
    ADC_OVERSAMPLING_OFF = 0
    ADC_OVERSAMPLING_2x  = 1
    ADC_OVERSAMPLING_4x  = 2
    ADC_OVERSAMPLING_8x  = 3

class DeviceADCConfigStruct(Structure):
    """Wrapper for structure ifx_Avian_ADC_Config_t"""
    _fields_ = (("samplerate_Hz", c_uint32),
                ("tracking", c_uint32),
                ("sample_time", c_uint32),
                ("double_msb_time", c_uint8),
                ("oversampling", c_uint32))

class IntInRangeValueStruct(Structure):
    _fields_ = (("value", c_uint32),
                ("min", c_uint32),
                ("max", c_uint32))

class Int64InRangeValueStruct(Structure):
    _fields_ = (("value", c_uint64),
                ("min", c_uint64),
                ("max", c_uint64))

class IntRangeStruct(Structure):
    _fields_ = (("min", c_uint32),
                ("max", c_uint32))

class FloatRangeStruct(Structure):
    _fields_ = (("min", c_float),
                ("max", c_float))

class DeviceBasebandVgaGain(IntEnum):
    VGA_GAIN_0dB = 0
    VGA_GAIN_5dB = 1
    VGA_GAIN_10dB = 2
    VGA_GAIN_15dB = 3
    VGA_GAIN_20dB = 4
    VGA_GAIN_25dB = 5
    VGA_GAIN_30dB = 6

class DeviceBasebandHpGain(IntEnum):
    HP_GAIN_18dB = 0
    HP_GAIN_30dB = 1

class DeviceBasebandConfigStruct(Structure):
    """Wrapper for structure ifx_Avian_Baseband_Config_t"""
    _fields_ = (("vga_gain", c_uint32),
                ("hp_gain", c_uint32),
                ("hp_cutoff", c_uint16),
                ("aaf_cutoff", c_uint16))

class DeviceTestSignalGeneratorMode(IntEnum):
    TEST_SIGNAL_MODE_OFF = 0
    TEST_SIGNAL_MODE_BASEBAND_TEST = 1
    TEST_SIGNAL_MODE_TOGGLE_TX_ENABLE = 2
    TEST_SIGNAL_MODE_TOGGLE_DAC_VALUE = 3
    TEST_SIGNAL_MODE_TOGGLE_RX_SELF_TEST = 4

class DeviceTestSignalGeneratorStruct(Structure):
    """Wrapper for structure ifx_Avian_Test_Signal_Generator_t"""
    _fields_ = (("mode", c_uint32),
                ("frequency_Hz", c_float))

class DeviceCWConfigStruct(Structure):
    """Wrapper for structure ifx_Avian_CW_Config_t"""
    _fields_ = (("tx_mask", c_uint32),
                ("rx_mask", c_uint32),
                ("duty_cycle", c_uint8),
                ("num_of_samples", IntInRangeValueStruct),
                ("tx_dac_value", IntInRangeValueStruct),
                ("rf_freq_Hz", Int64InRangeValueStruct),
                ("sample_rate_range_Hz", IntRangeStruct),
                ("test_signal_freq_range_Hz", FloatRangeStruct),
                ("adc_configs", DeviceADCConfigStruct),
                ("baseband_configs", DeviceBasebandConfigStruct),
                ("test_signal_configs", DeviceTestSignalGeneratorStruct))

class MatrixRStruct(Structure):
    _fields_ = (('d', POINTER(c_float)),
                ('rows', c_uint32),
                ('cols', c_uint32),
                ('stride', c_uint32*2),
                ('owns_d', c_uint8, 1))

    @classmethod
    def from_numpy(cls, np_frame):
        """Create a real matrix structure from a numpy 2-D array"""
        rows, cols = np_frame.shape

        if np_frame.dtype != np.float32:
            # If necessary convert matrix to float matrix
            np_frame = np.array(np_frame, dtype=np.float32)

        # We do not copy data but create a view. It is crucial that the memory
        # of the matrix np_frame is not released as long as this object
        # exists. For this reason we assign np_frame to this object. This way
        # the memory of np_frame is not released before our current object is
        # released.
        d = np_frame.ctypes.data_as(POINTER(c_float))
        mat = MatrixRStruct(d, rows, cols, cols, 0)
        mat.np_frame = np_frame # avoid that memory of np_frame is freed
        return mat

    def to_numpy(self):
        """Convert matrix structure to a numpy 2-D array"""
        shape = (self.rows, self.cols)
        data = np.ctypeslib.as_array(self.d, shape)
        return np.array(data, order="C", copy=True)


DeviceCWConfigStructPointer = POINTER(DeviceCWConfigStruct)
DeviceBasebandConfigStructPointer = POINTER(DeviceBasebandConfigStruct)
DeviceADCConfigStructPointer = POINTER(DeviceADCConfigStruct)
DeviceTestSignalGeneratorStructPointer = POINTER(DeviceTestSignalGeneratorStruct)
MatrixRStructPointer = POINTER(MatrixRStruct)

def to_dict(dict_instance, obj, parent_name=None):
    if not hasattr(obj, '_fields_'):
        return dict_instance

    sub_dict = dict()

    for field in obj._fields_:
        name = field[0]

        if not hasattr(getattr(obj, name), '_fields_'):
            if(parent_name):
                sub_dict[name] = getattr(obj, name)
            else:
                dict_instance[name] = getattr(obj, name)
        else:
            to_dict(dict_instance, getattr(obj, name), name)

    if(parent_name):
        dict_instance[parent_name] = sub_dict

    return dict_instance

def convert_cw_config_dict_enums(cw_config_as_dict):
    cw_config_as_dict["adc_configs"]["tracking"] = DeviceADCTracking(cw_config_as_dict["adc_configs"]["tracking"])
    cw_config_as_dict["adc_configs"]["sample_time"] = DeviceADCSampleTime(cw_config_as_dict["adc_configs"]["sample_time"])
    cw_config_as_dict["adc_configs"]["oversampling"] = DeviceADCOversampling(cw_config_as_dict["adc_configs"]["oversampling"])

    cw_config_as_dict["baseband_configs"]["vga_gain"] = DeviceBasebandVgaGain(cw_config_as_dict["baseband_configs"]["vga_gain"])
    cw_config_as_dict["baseband_configs"]["hp_gain"] = DeviceBasebandHpGain(cw_config_as_dict["baseband_configs"]["hp_gain"])
    cw_config_as_dict["baseband_configs"]["hp_cutoff"] = cw_config_as_dict["baseband_configs"]["hp_cutoff"]
    cw_config_as_dict["baseband_configs"]["aaf_cutoff"] = cw_config_as_dict["baseband_configs"]["aaf_cutoff"]

    cw_config_as_dict["test_signal_configs"]["mode"] = DeviceTestSignalGeneratorMode(cw_config_as_dict["test_signal_configs"]["mode"])

def create_adc_configs_struct(samplerate_Hz, tracking, sample_time, double_msb_time, oversampling):
    """Utility function for the creation of a DeviceADCConfigStruct type"""
    configs = DeviceADCConfigStruct()
    configs.samplerate_Hz = samplerate_Hz
    configs.tracking = tracking
    configs.sample_time = sample_time
    configs.double_msb_time = double_msb_time
    configs.oversampling = oversampling
    return configs

def create_baseband_configs_struct(vga_gain, hp_gain, hp_cutoff, aaf_cutoff):
    """Utility function for the creation of a DeviceBasebandConfigStruct type"""
    configs = DeviceBasebandConfigStruct()
    configs.vga_gain = vga_gain
    configs.hp_gain = hp_gain
    configs.hp_cutoff = hp_cutoff
    configs.aaf_cutoff = aaf_cutoff
    return configs

def create_test_signal_configs_struct(mode, frequency_Hz):
    """Utility function for the creation of a DeviceTestSignalGeneratorStruct type"""
    configs = DeviceTestSignalGeneratorStruct()
    configs.mode = mode
    configs.frequency_Hz = frequency_Hz
    return configs

def initialize_module_cw():
    """Further initialize the module and return ctypes handle"""

    # device (constant wave)
    dll.ifx_avian_cw_create.restype = c_void_p
    dll.ifx_avian_cw_create.argtypes = [c_void_p]

    dll.ifx_avian_cw_get_default_config.restype = None
    dll.ifx_avian_cw_get_default_config.argtypes = [c_void_p, DeviceCWConfigStructPointer]

    dll.ifx_avian_cw_enabled.restype = c_bool
    dll.ifx_avian_cw_enabled.argtypes = [c_void_p]

    dll.ifx_avian_cw_start_emission.restype = None
    dll.ifx_avian_cw_start_emission.argstype = [c_void_p]

    dll.ifx_avian_cw_stop_emission.restype = None
    dll.ifx_avian_cw_stop_emission.argstype = [c_void_p]

    dll.ifx_avian_cw_set_rf_frequency.restype = None
    dll.ifx_avian_cw_set_rf_frequency.argstype = [c_void_p, c_uint64]

    dll.ifx_avian_cw_get_rf_frequency.restype = c_uint64
    dll.ifx_avian_cw_get_rf_frequency.argstype = [c_void_p]

    dll.ifx_avian_cw_set_tx_dac_value.restype = None
    dll.ifx_avian_cw_set_tx_dac_value.argstype = [c_void_p, c_uint32]

    dll.ifx_avian_cw_get_tx_dac_value.restype = c_uint32
    dll.ifx_avian_cw_get_tx_dac_value.argstype = [c_void_p]

    dll.ifx_avian_cw_enable_tx_antenna.restype = None
    dll.ifx_avian_cw_enable_tx_antenna.argstype = [c_void_p, c_uint32, c_bool]

    dll.ifx_avian_cw_tx_antenna_enabled.restype = c_bool
    dll.ifx_avian_cw_tx_antenna_enabled.argstype = [c_void_p, c_uint32]

    dll.ifx_avian_cw_enable_rx_antenna.restype = None
    dll.ifx_avian_cw_enable_rx_antenna.argstype = [c_void_p, c_uint32, c_bool]

    dll.ifx_avian_cw_rx_antenna_enabled.restype = c_bool
    dll.ifx_avian_cw_rx_antenna_enabled.argstype = [c_void_p, c_uint32]

    dll.ifx_avian_cw_set_num_of_samples_per_antenna.restype = None
    dll.ifx_avian_cw_set_num_of_samples_per_antenna.argstype = [c_void_p, c_uint32]

    dll.ifx_avian_cw_get_num_of_samples_per_antenna.restype = c_uint32
    dll.ifx_avian_cw_get_num_of_samples_per_antenna.argstype = [c_void_p]

    dll.ifx_avian_cw_set_baseband_params.restype = None
    dll.ifx_avian_cw_set_baseband_params.argstype = [c_void_p, DeviceBasebandConfigStructPointer]

    dll.ifx_avian_cw_get_baseband_params.restype = DeviceBasebandConfigStructPointer
    dll.ifx_avian_cw_get_baseband_params.argstype = [c_void_p]

    dll.ifx_avian_cw_set_adc_params.restype = None
    dll.ifx_avian_cw_set_adc_params.argstype = [c_void_p, DeviceADCConfigStructPointer]

    dll.ifx_avian_cw_get_adc_params.restype = DeviceADCConfigStructPointer
    dll.ifx_avian_cw_get_adc_params.argstype = [c_void_p]

    dll.ifx_avian_cw_get_sampling_rate_limits.restype = c_uint32
    dll.ifx_avian_cw_get_sampling_rate_limits.argstype = [c_void_p, DeviceADCConfigStructPointer, POINTER(c_float), POINTER(c_float)]

    dll.ifx_avian_cw_get_sampling_rate.restype = c_float
    dll.ifx_avian_cw_get_sampling_rate.argstype = [c_void_p]

    dll.ifx_avian_cw_set_test_signal_generator_config.restype = None
    dll.ifx_avian_cw_set_test_signal_generator_config.argstype = [c_void_p, DeviceTestSignalGeneratorStructPointer]

    dll.ifx_avian_cw_get_test_signal_generator_config.restype = DeviceTestSignalGeneratorStructPointer
    dll.ifx_avian_cw_get_test_signal_generator_config.argstype = [c_void_p]

    dll.ifx_avian_cw_measure_temperature.restype = c_float
    dll.ifx_avian_cw_measure_temperature.argstype = [c_void_p]

    dll.ifx_avian_cw_measure_tx_power.restype = c_float
    dll.ifx_avian_cw_measure_tx_power.argstype = [c_void_p, c_uint32]

    dll.ifx_avian_cw_capture_frame.restype = MatrixRStructPointer
    dll.ifx_avian_cw_capture_frame.argstype = [c_void_p, MatrixRStructPointer]

    dll.ifx_avian_cw_destroy.restype = None
    dll.ifx_avian_cw_destroy.argtypes = [c_void_p]

initialize_module_cw()
dll_cw = dll


class DeviceConstantWave:
    def __init__(self, device):
        """Create and initialize constant wave controller

        Parameter:
        - device: instance of ifxRadarSDK.Device
        """
        h = dll_cw.ifx_avian_cw_create(device.handle)
        self.handle = c_void_p(h) # Reason of that cast HMI-2896
        check_rc()

        # Save the device object as part of this object. This is crucial
        # because the CW handle is only valid as long as the device object is
        # valid. By saving the device object as part of this object we prevent
        # that the device handle gets destroyed. In other words, we prevent
        # that the connection to the device is closed while we are still
        # accessing CW functions.
        self.device = device

    def get_default_config(self):
        """Get the default configuration from the device"""
        config = DeviceCWConfigStruct()
        dll_cw.ifx_avian_cw_get_default_config(self.handle, byref(config))
        check_rc()
        return config

    def get_default_config_as_dict(self):
        """Get the default configuration from the device (formatted as dictionary)"""
        config = DeviceCWConfigStruct()
        dll_cw.ifx_avian_cw_get_default_config(self.handle, byref(config))
        check_rc()
        # return struct as dictionary
        d = dict()
        d = to_dict(d, config)
        convert_cw_config_dict_enums(d)
        return d

    def enabled(self):
        """ Checks whether the constant wave mode is enabled or not (returns boolean) """
        cw_enabled = dll_cw.ifx_avian_cw_enabled(self.handle)
        check_rc()
        return cw_enabled

    def start_emission(self):
        """ Starts the constant wave emission """
        dll_cw.ifx_avian_cw_start_emission(self.handle)
        check_rc()

    def stop_emission(self):
        """ Stops the constant wave emission """
        dll_cw.ifx_avian_cw_stop_emission(self.handle)
        check_rc()

    def set_rf_frequency(self, frequency_Hz):
        """ Sets the rf frequency

        Parameter:
        - frequency_Hz: the rf frequency to be set
        """
        dll_cw.ifx_avian_cw_set_rf_frequency(self.handle, c_uint64(frequency_Hz))
        check_rc()

    def get_rf_frequency(self):
        """ Gets the rf frequency """
        frequency_Hz = dll_cw.ifx_avian_cw_get_rf_frequency(self.handle)
        check_rc()
        return int(frequency_Hz)

    def set_tx_dac_value(self, dac_value):
        """ Sets the dac value

        Parameter:
        - dac_value: the dac value to be set
        """
        dll_cw.ifx_avian_cw_set_tx_dac_value(self.handle, c_uint32(dac_value))
        check_rc()

    def get_tx_dac_value(self):
        """ Gets the dac value """
        dac_value = dll_cw.ifx_avian_cw_get_tx_dac_value(self.handle)
        check_rc()
        return int(dac_value)

    def enable_tx_antenna(self, antenna, enable):
        """ Enables a specific tx antenna

        Parameter:
        - antenna: the tx antenna id (zero based) to be enabled or disabled
        - enable: the enabled flag
        """
        dll_cw.ifx_avian_cw_enable_tx_antenna(self.handle, antenna, enable)
        check_rc()

    def antenna_tx_enabled(self, antenna):
        """ Check whether a specific tx antenna is enabled

        Parameter:
        - antenna: the tx antenna id (zero based) to be checked if enabled
        """
        enabled = dll_cw.ifx_avian_cw_tx_antenna_enabled(self.handle, antenna)
        check_rc()
        return enabled

    def enable_rx_antenna(self, antenna, enable):
        """ Enables a specific rx antenna

        Parameter:
        - antenna: the rx antenna id (zero based) to be enabled or disabled
        - enable: the enabled flag
        """
        dll_cw.ifx_avian_cw_enable_rx_antenna(self.handle, antenna, enable)
        check_rc()

    def antenna_rx_enabled(self, antenna):
        """ Check whether a specific rx antenna is enabled

        Parameter:
        - antenna: the rx antenna id (zero based) to be checked if enabled
        """
        enabled = dll_cw.ifx_avian_cw_rx_antenna_enabled(self.handle, antenna)

        check_rc()
        return enabled

    def set_num_of_samples_per_antenna(self, num_samples):
        """ Sets the number of samples per antenna

        Parameter:
        - num_samples: the number of samples (per rx antenna)
        """
        dll_cw.ifx_avian_cw_set_num_of_samples_per_antenna(self.handle, c_uint32(num_samples))
        check_rc()

    def get_num_of_samples_per_antenna(self):
        """ Gets the number of samples per antenna """
        num_samples = dll_cw.ifx_avian_cw_get_num_of_samples_per_antenna(self.handle)
        check_rc()
        return num_samples

    def set_baseband_params(self, baseband_params):
        """ Sets the baseband parameters

        Parameter:
        - baseband_params: the baseband parameters (DeviceBasebandConfigStruct) to be set
        """
        dll_cw.ifx_avian_cw_set_baseband_params(self.handle, byref(baseband_params))
        check_rc()

    def get_baseband_params(self):
        """ Gets the baseband parameters """
        baseband_params = dll_cw.ifx_avian_cw_get_baseband_params(self.handle)
        check_rc()
        return baseband_params.contents

    def get_baseband_params_as_dict(self):
        """ Gets the baseband parameters (dictionary formatted) """
        baseband_params = self.get_baseband_params()
        check_rc()
        # return struct as dictionary
        d = dict((field, getattr(baseband_params, field)) for field, _ in baseband_params._fields_)
        d["vga_gain"] = DeviceBasebandVgaGain(d["vga_gain"])
        d["hp_gain"] = DeviceBasebandHpGain(d["hp_gain"])
        d["hp_cutoff"] = d["hp_cutoff"]
        d["aaf_cutoff"] = d["aaf_cutoff"]
        return d

    def set_adc_params(self, adc_params):
        """ Sets the adc params

        Parameter:
        - ads_params: the adc parameters (DeviceADCConfigStruct) to be set
        """
        dll_cw.ifx_avian_cw_set_adc_params(self.handle, byref(adc_params))
        check_rc()

    def get_adc_params(self):
        # Gets the adc params
        adc_params = dll_cw.ifx_avian_cw_get_adc_params(self.handle)
        check_rc()
        return adc_params.contents

    def get_adc_params_as_dict(self):
        # Gets the adc params
        adc_params = self.get_adc_params()
        # return struct as dictionary
        d = dict((field, getattr(adc_params, field)) for field, _ in adc_params._fields_)
        d["tracking"] = DeviceADCTracking(d["tracking"])
        d["sample_time"] = DeviceADCSampleTime(d["sample_time"])
        d["oversampling"] = DeviceADCOversampling(d["oversampling"])
        return d

    def set_test_signal_generator_config(self, signal_generator_config):
        """ Sets the test signal generator configuration

        Parameter:
        - signal_generator_config: the signal generator configuration (DeviceTestSignalGeneratorStruct) to be set
        """
        dll_cw.ifx_avian_cw_set_test_signal_generator_config(self.handle, byref(signal_generator_config))
        check_rc()

    def get_test_signal_generator_config(self):
        """ Gets the test signal generator configuration """
        signal_generator_config = dll_cw.ifx_avian_cw_get_test_signal_generator_config(self.handle)
        check_rc()
        return signal_generator_config.contents

    def get_test_signal_generator_config_as_dict(self):
        """ Gets the test signal generator configuration """
        signal_generator_config = self.get_test_signal_generator_config()
        # return struct as dictionary
        d = dict((field, getattr(signal_generator_config, field)) for field, _ in signal_generator_config._fields_)
        d["mode"] = DeviceTestSignalGeneratorMode(d["mode"])
        return d

    def measure_temperature(self):
        """ Gets the temperature value """
        temperature = dll_cw.ifx_avian_cw_measure_temperature(self.handle)
        check_rc()
        return float(temperature)

    def measure_tx_power(self, antenna):
        """ Gets the tx power level
            Return power is equal to -1 if CW is not in emission
            Parameter:
           - antenna: index of the antenna to be measured. The value
                      is 0 based and must be less than the value returned
                      by \ref get_number_of_tx_antennas. If the value is
                      not in the allowed range, an exception is thrown."""
        tx_power = dll_cw.ifx_avian_cw_measure_tx_power(self.handle, antenna)
        check_rc()
        return float(tx_power)

    def get_sampling_rate_limits(self, adc_config):
        """ Gets the sampling rate limits """
        min = c_float(0)
        max = c_float(0)
        error = dll_cw.ifx_avian_cw_get_sampling_rate_limits(self.handle, byref(adc_config), byref(min), byref(max))
        check_rc(error)
        return [float(min.value), float(max.value)]

    def get_sampling_rate(self):
        """ Gets the sampling rate """
        sampling_rate = dll_cw.ifx_avian_cw_get_sampling_rate(self.handle)
        check_rc()
        return float(sampling_rate)

    def capture_frame(self):
        """ Captures a single frame in cw mode and returns it as a 2D numpy array"""
        frame = dll_cw.ifx_avian_cw_capture_frame(self.handle, None)
        check_rc()
        frame_numpy = frame.contents.to_numpy()
        dll.ifx_mat_destroy_r(frame)
        return frame_numpy

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __del__(self):
        """Destroy device CW handle - just in case due to (handle and device) references being in the constructor"""
        if hasattr(self, "handle") and (self.handle is not None):
            dll_cw.ifx_avian_cw_destroy(self.handle)
            self.handle = None
