import sys
import math
import time
#from grove.adc import ADC

## GLOBAL VARIABLES ##
poll_sensor_window = 4 # get reading from sensor every x units -- placeholder value


def init_wifi():
	return 1
def init_bt():
	return 0
def init_radio():
	return 0

def init_connect(cxn_type=0):

	ready = 0

	if (cxn_type == 0): # WiFi
		if(init_wifi() == 1):
			ready = 1

	elif (cxn_type == 1): # Bluetooth
		if(init_bt() == 1):
			ready = 1

	elif (cxn_type == 2): # Radio
		if(init_radio() == 1):
			ready = 1

	return ready

def choose_cxn_device(cxn_type=0):

	## Try and Connect to a Device ##

	# Could design a spin wait / polling to try and connect forever, or a time window
	if(init_connect(cxn_type) == 0): # Device tried did not connect
		if(init_connect(0) == 0):    # Try connecting to all devices once
			if(init_connect(1) == 0):
				if(init_connect(2) == 0):
					cxn_type = -1 # Could not connect to a device
				else:
					cxn_type = 2 # Cxd to Radio
			else:
				cxn_type = 1 # Cxd to BT
		else:
			cxn_type = 0 # Cxd to Wifi

	return cxn_type # return here instead of later?


def main():

	# initialize sensors

	# intialize connection 
	connection_type = 0 # 0 W, 1 B, 2 R
	connection_type = choose_cxn_device(connection_type) # Default to WiFi or have some sort of decision var?

	if(connection_type == 0):
		print("wifi")
	elif(connection_type == 1):
		print("bt")
	elif(connection_type == 2):
		print("radio")
	else:
		print("Error: Could not connect to a device")
		exit(1)

	#

	return 


# make driver code general enough to think about > 1 sensor
# make communication easy to “switch out” if possible
# have an initial window of specific demographic and sampling baseline data before streaming data
#	I forget data rate but sensor might sample at like 4hz or something similar
# leave a “denoising” chunk between receiving from sensor and sending it to model
# collect & send data every x time
# packets – fixed sized packets
# check connection periodically, or if message send or receive fails enough, attempt to switch to a new comms
# account for not knowing demographics [ data noise / "attack"? ] 
# ringbuffer?


if __name__ == '__main__':
	main()
