import sys
import time
import math
# from grove.adc import ADC

## Packet = streamed data formatting and functions

# csv file format? for packet info

# ex. could sample at 10 Hz, and average/downsample to 5 Hz

# really want it every 1 minute, 30 minutes, etc.
# notification timeout -- 1/60 min or 1/30 min.
# 		in a test -- maybe 1/ 5-10 min.

## !! !! ##
# how often does the ML want data?
# how much data at a time does the ML want?
# what does the data look like/ how is it formatted?
# what is the interplay b/w the Baseline Period vs. Real Time / Batching Period?


# Real Time vs. Batching

class Packet:

	def __init__(self):
		data = 0


	def assemble(self):
		pass
