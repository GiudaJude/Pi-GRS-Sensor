"""Sensor wrapper for Grove GSR / ADC with desktop fallback simulation.

This module provides a GroveGSRSensor class and an adc_to_us helper that
match the interface used by `device_main.py`. If the `grove.adc` package is
available the wrapper will attempt to use the real ADC; otherwise it falls
back to a simple simulator useful for desktop testing.
"""

import time
import math
import random
try:
    from grove.adc import ADC
    _HAS_ADC = True
except Exception:
    ADC = None
    _HAS_ADC = False


class GroveGSRSensor:
    """Wrapper for a GSR sensor read.

    If the grove.adc package is available this reads from the ADC hardware.
    Otherwise it produces a simulated GSR-like signal for testing.
    """

    def __init__(self, channel=0, fs=4.0, simulate=None):
        """Initialize the sensor wrapper.

        channel: ADC channel index
        fs: sampling frequency used for simulation timing (Hz)
        simulate: if True/False forces simulation on/off; if None auto-detects
        """
        self.channel = channel
        self.fs = float(fs)
        self._t = 0.0
        self._dt = 1.0 / self.fs
        # Decide whether to simulate (default: simulate if ADC not present)
        self.simulate = (_HAS_ADC is False) if simulate is None else bool(simulate)

        if not self.simulate and _HAS_ADC:
            try:
                self._adc = ADC()
            except Exception:
                # If ADC instantiation fails, fall back to simulated mode
                self.simulate = True
                self._adc = None
        else:
            self._adc = None

    def read_raw(self):
        """Return a raw ADC reading (float). In simulation returns ADC-like counts."""
        if self.simulate:
            # simulated GSR: slow tonic + occasional phasic + noise
            self._t += self._dt
            tonic = 200 + 15 * math.sin(0.05 * self._t)        # slow baseline drift
            phasic = 40 * math.sin(2.0 * self._t) * (random.random() > 0.9)
            noise = random.gauss(0, 3.0)
            return max(0.0, tonic + phasic + noise)
        else:
            try:
                if hasattr(self._adc, 'read'):
                    return float(self._adc.read(self.channel))
                if hasattr(self._adc, 'read_adc'):
                    return float(self._adc.read_adc(self.channel))
                if hasattr(self._adc, 'value'):
                    return float(self._adc.value)
                # Last resort: try constructing ADC with channel param
                adc_obj = ADC(self.channel)
                if hasattr(adc_obj, 'read'):
                    return float(adc_obj.read())
                return float(adc_obj.value)
            except Exception:
                # On any hardware error, switch to simulation to keep program running
                self.simulate = True
                return self.read_raw()

    @staticmethod
    def adc_to_us(adc_value, adc_max=1023.0, vref=3.3):
        """Convert ADC reading to microsiemens (µS).

        This function is a simple placeholder and should be calibrated for your
        specific hardware and circuit. It maps ADC counts -> voltage -> µS
        via a linear scale.
        """
        try:
            v = float(adc_value) / float(adc_max) * float(vref)
        except Exception:
            return 0.0
        # placeholder linear scale to micro-Siemens
        scale = 50.0
        return (v / vref) * scale


if __name__ == '__main__':
    # Quick smoke test for the wrapper
    s = GroveGSRSensor(channel=0, fs=4.0)
    for _ in range(16):
        raw = s.read_raw()
        us = GroveGSRSensor.adc_to_us(raw)
        print(f"raw={raw:.2f}, µS≈{us:.2f}, simulate={s.simulate}")
        time.sleep(1.0 / s.fs)