import sys
from grove.adc import ADC

class GroveGSRSensor:

	def __init__(self, channel):
		self.channel = channel
		self.adc = ADC()

		# Adjust hardware setup
		self.ADC_MAX = 1023       # 10-bit ADC
		self.V_REF = 3.3          # Reference voltage (V)
		self.R_FIXED = 100000.0   # Fixed resistor in ohms (100 kΩ typical)
		# MAX_GSR = 40.0       # Maximum allowed GSR value (μS)

	@property
	def read_GSR(self):
		return self.adc.read(self.channel)

	def adc_to_us(self, adc_value):
		"""Convert ADC reading to micro-Siemens (μS)."""
		v_out = (adc_value / self.ADC_MAX) * self.V_REF

		if v_out >= self.V_REF:
			return None 

		try:
			r_skin = (self.R_FIXED * v_out) / (self.V_REF - v_out)
			g_siemens = 1.0 / r_skin
			g_microsiemens = g_siemens * 1e6

			"""# Cap the value at MAX_GSR
				if g_microsiemens > MAX_GSR:
					g_microsiemens = MAX_GSR"""

		return g_microsiemens

		except ZeroDivisionError:
			return None
