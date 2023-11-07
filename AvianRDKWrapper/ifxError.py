
# ===========================================================================
# Copyright (C) 2021 Infineon Technologies AG
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
        
error_class_list = ['Error', 'ErrorArgumentNull', 'ErrorArgumentInvalid', 'ErrorArgumentOutOfBounds', 'ErrorArgumentInvalidExpectedReal', 'ErrorArgumentInvalidExpectedComplex', 'ErrorIndexOutOfBounds', 'ErrorDimensionMismatch', 'ErrorMemoryAllocationFailed', 'ErrorInPlaceCalculationNotSupported', 'ErrorMatrixSingular', 'ErrorMatrixNotPositiveDefinite', 'ErrorNotSupported', 'ErrorInternal', 'ErrorNoDevice', 'ErrorDeviceBusy', 'ErrorCommunicationError', 'ErrorNumSamplesOutOfRange', 'ErrorRxAntennaCombinationNotAllowed', 'ErrorIfGainOutOfRange', 'ErrorSamplerateOutOfRange', 'ErrorRfOutOfRange', 'ErrorTxPowerOutOfRange', 'ErrorChirpRateOutOfRange', 'ErrorFrameRateOutOfRange', 'ErrorNumChirpsNotAllowed', 'ErrorFrameSizeNotSupported', 'ErrorTimeout', 'ErrorFifoOverflow', 'ErrorTxAntennaModeNotAllowed', 'ErrorFirmwareVersionNotSupported', 'ErrorDeviceNotSupported', 'ErrorBasebandConfigNotAllowed', 'ErrorAdcConfigNotAllowed', 'ErrorTestSignalModeNotAllowed', 'ErrorFrameAcquisitionFailed', 'ErrorTemperatureMeasurementFailed', 'ErrorPowerMeasurementFailed', 'ErrorTxAntennaCombinationNotAllowed', 'ErrorSequencerError', 'ErrorEeprom', 'ErrorHost', 'ErrorHostFileDoesNotExist', 'ErrorHostFileInvalid', 'ErrorApp']      
error_mapping_exception = {65536: 'Error', 65537: 'ErrorArgumentNull', 65538: 'ErrorArgumentInvalid', 65539: 'ErrorArgumentOutOfBounds', 65540: 'ErrorArgumentInvalidExpectedReal', 65541: 'ErrorArgumentInvalidExpectedComplex', 65542: 'ErrorIndexOutOfBounds', 65543: 'ErrorDimensionMismatch', 65544: 'ErrorMemoryAllocationFailed', 65545: 'ErrorInPlaceCalculationNotSupported', 65546: 'ErrorMatrixSingular', 65547: 'ErrorMatrixNotPositiveDefinite', 65548: 'ErrorNotSupported', 65549: 'ErrorInternal', 69632: 'ErrorNoDevice', 69633: 'ErrorDeviceBusy', 69634: 'ErrorCommunicationError', 69635: 'ErrorNumSamplesOutOfRange', 69636: 'ErrorRxAntennaCombinationNotAllowed', 69637: 'ErrorIfGainOutOfRange', 69638: 'ErrorSamplerateOutOfRange', 69639: 'ErrorRfOutOfRange', 69640: 'ErrorTxPowerOutOfRange', 69641: 'ErrorChirpRateOutOfRange', 69642: 'ErrorFrameRateOutOfRange', 69643: 'ErrorNumChirpsNotAllowed', 69644: 'ErrorFrameSizeNotSupported', 69645: 'ErrorTimeout', 69646: 'ErrorFifoOverflow', 69647: 'ErrorTxAntennaModeNotAllowed', 69648: 'ErrorFirmwareVersionNotSupported', 69649: 'ErrorDeviceNotSupported', 69650: 'ErrorBasebandConfigNotAllowed', 69651: 'ErrorAdcConfigNotAllowed', 69652: 'ErrorTestSignalModeNotAllowed', 69653: 'ErrorFrameAcquisitionFailed', 69654: 'ErrorTemperatureMeasurementFailed', 69655: 'ErrorPowerMeasurementFailed', 69656: 'ErrorTxAntennaCombinationNotAllowed', 69657: 'ErrorSequencerError', 69664: 'ErrorEeprom', 196608: 'ErrorHost', 196609: 'ErrorHostFileDoesNotExist', 196610: 'ErrorHostFileInvalid', 2147483648: 'ErrorApp'}      
  
ifx_error_api_base = 0x00010000        
ifx_error_dev_base = 0x00011000         
ifx_error_host_base = 0x00030000        
ifx_error_app_base = 0x80000000     

def raise_exception_for_error_code(error_code,dll):

    if error_code in error_mapping_exception:
        raise eval(error_mapping_exception[error_code])(dll)
    else:
        raise GeneralError(error_code,dll)  

class GeneralError(Exception):
    def __init__(self, error,dll):
        '''Create new RadarSDKException with error code given by error'''
        self.error = error
        self.dll = dll

    def __str__(self):
        '''Exception message'''
        
        return self.dll.ifx_error_to_string(self.error).decode("ascii")
        
class ErrorApiBase(GeneralError):
        def __init__(self,error,dll):
            super().__init__(error, dll)
        
class ErrorDevBase(GeneralError):
        def __init__(self,error,dll):
            super().__init__(error, dll)
            
class ErrorHostBase(GeneralError):
        def __init__(self,error,dll):
            super().__init__(error, dll)
        
class ErrorAppBase(GeneralError):
        def __init__(self,error, dll):
            super().__init__(error, dll)
    

class Error(ErrorApiBase):
    ''' A generic error occurred in radar SDK API.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base,dll)

class ErrorArgumentNull(ErrorApiBase):
    ''' Argument Null error.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x01,dll)

class ErrorArgumentInvalid(ErrorApiBase):
    ''' Argument invalid error.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x02,dll)

class ErrorArgumentOutOfBounds(ErrorApiBase):
    ''' Argument out of bounds.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x03,dll)

class ErrorArgumentInvalidExpectedReal(ErrorApiBase):
    ''' Argument invalid expected real.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x04,dll)

class ErrorArgumentInvalidExpectedComplex(ErrorApiBase):
    ''' Argument invalid expected complex.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x05,dll)

class ErrorIndexOutOfBounds(ErrorApiBase):
    ''' Index out of bounds.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x06,dll)

class ErrorDimensionMismatch(ErrorApiBase):
    ''' Dimension mismatch.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x07,dll)

class ErrorMemoryAllocationFailed(ErrorApiBase):
    ''' Memory allocation failed.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x08,dll)

class ErrorInPlaceCalculationNotSupported(ErrorApiBase):
    ''' In place calculation not supported.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x09,dll)

class ErrorMatrixSingular(ErrorApiBase):
    ''' Matrix is singular.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x0a,dll)

class ErrorMatrixNotPositiveDefinite(ErrorApiBase):
    ''' Matrix is not positive definite.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x0b,dll)

class ErrorNotSupported(ErrorApiBase):
    ''' Generic error for unsupported API.'''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x0c,dll)

class ErrorInternal(ErrorApiBase):
    ''' Generic internal logic error '''
    def __init__(self,dll):
        super().__init__(ifx_error_api_base + 0x0d,dll)

class ErrorNoDevice(ErrorDevBase):
    ''' No device compatible to Radar SDK was found.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base,dll)

class ErrorDeviceBusy(ErrorDevBase):
    ''' The connected device is busy and cannot
 perform the requested action. This can happen
 during device handle creation when the device
 is in an undefined state. It is recommended to
 unplug and replug the device.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x01,dll)

class ErrorCommunicationError(ErrorDevBase):
    ''' The communication between host computer and
device is disturbed. This error is also
returned when the device sends an unexpected
error code.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x02,dll)

class ErrorNumSamplesOutOfRange(ErrorDevBase):
    ''' The device does not support the requested
 number of samples, because the requested
 number is too high.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x03,dll)

class ErrorRxAntennaCombinationNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
 combination of RX antennas to be enabled.
 This error typically occurs when a
 non-existing antenna is requested to be
 enabled.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x04,dll)

class ErrorIfGainOutOfRange(ErrorDevBase):
    ''' The device does not support the requested IF
 gain, because the requested gain is either too
 high or too low.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x05,dll)

class ErrorSamplerateOutOfRange(ErrorDevBase):
    ''' The device does not support the requested
sampling rate, because the requested rate is
either too high or too low.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x06,dll)

class ErrorRfOutOfRange(ErrorDevBase):
    ''' The requested FMCW start and end frequency are
not in the supported RF range of the device.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x07,dll)

class ErrorTxPowerOutOfRange(ErrorDevBase):
    ''' The device does not support the requested TX
power, because the requested value is
too high.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x08,dll)

class ErrorChirpRateOutOfRange(ErrorDevBase):
    ''' The requested chirp-to-chirp time cannot be
applied. This typically happens when the
requested time is shorter than the chirp
duration resulting from the specified sampling
rate and number of samples.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x09,dll)

class ErrorFrameRateOutOfRange(ErrorDevBase):
    ''' The requested frame period cannot be applied.
This typically happens when the requested
period is shorter than the frame duration
resulting from the specified sampling
rate, number of samples and chirp-to-chirp
time.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0a,dll)

class ErrorNumChirpsNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
number of chirps per frame, because the
number is too high.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0b,dll)

class ErrorFrameSizeNotSupported(ErrorDevBase):
    ''' The device does not support the frame size
resulting from specified number of chirps,
number of samples and number of antennas.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0c,dll)

class ErrorTimeout(ErrorDevBase):
    ''' The device did not acquire a complete time
domain data frame within the expected time.
 '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0d,dll)

class ErrorFifoOverflow(ErrorDevBase):
    ''' The device stopped acquisition of time domain
data due to an internal buffer overflow. This
happens when time domain data is acquired
faster than it is read from the device.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0e,dll)

class ErrorTxAntennaModeNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
mode of TX antennas to be used.
This error typically occurs when a
the requested tx_mode is not supported by the
device due to non availability of TX antennas
for that mode.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x0f,dll)

class ErrorFirmwareVersionNotSupported(ErrorDevBase):
    ''' The firmware version is no longer supported.
Please update the firmware to the latest version. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x10,dll)

class ErrorDeviceNotSupported(ErrorDevBase):
    ''' The device is not supported. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x11,dll)

class ErrorBasebandConfigNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
baseband configurations ifx_Avian_Baseband_Config_t. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x12,dll)

class ErrorAdcConfigNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
ADC configurations ifx_Avian_ADC_Config_t. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x13,dll)

class ErrorTestSignalModeNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
mode for test signal generator ifx_Avian_Test_Signal_Generator_t. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x14,dll)

class ErrorFrameAcquisitionFailed(ErrorDevBase):
    ''' The device does not succeed to capture ADC 
raw data. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x15,dll)

class ErrorTemperatureMeasurementFailed(ErrorDevBase):
    ''' The device does not succeed to measure the 
temperature value. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x16,dll)

class ErrorPowerMeasurementFailed(ErrorDevBase):
    ''' The device does not succeed to measure the
power value. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x17,dll)

class ErrorTxAntennaCombinationNotAllowed(ErrorDevBase):
    ''' The device does not support the requested
combination of TX antennas to be enabled.
This error typically occurs when a
non-existing antenna is requested to be
enabled.'''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x18,dll)

class ErrorSequencerError(ErrorDevBase):
    ''' The device reports a sequencer error. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x19,dll)

class ErrorEeprom(ErrorDevBase):
    ''' An error occured while reading or writing the EEPROM.
This error might occur if an RF shield does not have an EEPROM,
the EEPROM is broker or not correctly inizialized. '''
    def __init__(self,dll):
        super().__init__(ifx_error_dev_base + 0x20,dll)

class ErrorHost(ErrorHostBase):
    ''' A generic error occurred on Host side '''
    def __init__(self,dll):
        super().__init__(ifx_error_host_base,dll)

class ErrorHostFileDoesNotExist(ErrorHostBase):
    ''' Host file does not exist. '''
    def __init__(self,dll):
        super().__init__(ifx_error_host_base + 0x01,dll)

class ErrorHostFileInvalid(ErrorHostBase):
    ''' Invalid host file. '''
    def __init__(self,dll):
        super().__init__(ifx_error_host_base + 0x02,dll)

class ErrorApp(ErrorAppBase):
    ''' A generic error occurred on Application side '''
    def __init__(self,dll):
        super().__init__(ifx_error_app_base             ,dll)
