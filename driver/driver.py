import sys
import math
import time
from packet import Packet
#from grove.adc import ADC

#### GLOBAL VARIABLES ####

cxn_timeout 		 = 120000		# Connection Timeout -- try and connect to any device for 2 min -- placeholder value 
poll_sensor_window  = 4 			# get reading from sensor every x units -- placeholder value
check_cxn_timer 	 = 60000 		# check connection every 60 s, 60000 ms -- placeholder value

ringbuffer = 0

#### FUNCTIONS ####

def init_wifi():  # Initialize Wifi Comms Protocol
	return 1
def init_bt(): 	 # Initialize Bluetooth Comms Protocol
	return 0
def init_radio(): # Initialize Radio Comms Protocol
	return 0
def init_plug():  # Initialize Wired Pi Cxn Protocol
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

	elif(cxn_type == 3): # Plug
		if(init_plug() == 1):
			ready = 1

	return ready

def choose_cxn_device(cxn_type=0):

	## Try and Connect to a Device ##

	# Could design a spin wait / polling to try and connect within a time window
	if(init_connect(cxn_type) == 0): # Device tried did not connect -- Try connecting to all devices if first attempt fails
		if(init_connect(0) == 0):     # Try Wifi
			if(init_connect(1) == 0):	# Try BT
				if(init_connect(2) == 0):		# Try Radio
					if(init_connect(3) == 0):	# Try Pi w/ Wired Cxn
						cxn_type = -1 				# Could not connect to any device
					else:
						cxn_type = 3 # Cxd to Pi w/ Wired Cxn
				else:
					cxn_type = 2 # Cxd to Radio
			else:
				cxn_type = 1 # Cxd to BT
		else:
			cxn_type = 0 # Cxd to Wifi

	return cxn_type # return which device is connected 

#### MAIN FUNCTION ####

def main():

	## Initialize Sensors ##

	## Intialize Connection ## 
	connection_type = 0 # 0 W, 1 B, 2 R, 3 P
	connection_type = choose_cxn_device(connection_type) # Default to WiFi or have some sort of decision var?

	if(connection_type == 0):
		print("wifi")
	elif(connection_type == 1):
		print("bt")
	elif(connection_type == 2):
		print("radio")
	elif(connection_type == 3):
		print("plug")
	else:
		print("Error: Could not connect to a device")
		exit(1)

	state = 'Sample' # Define State

	## Begin Sampling Period ##

	# get and store data from sensor
	# format / possibly denoise
	# perioidcally send to Pi

	state = 'Recv'

	## Stream Data ##
	
	# while True: #

	#     # Check Connection Periodically #

	#     if state == 'Recv':
	#			# receive data from the sensor
	#			# accumulate into a data structure
	#			# format

	#		elif state == 'Send':
	#			# send data to Pi
	#			# Receive ACK from Pi, else Try to Send Again [check cxn, etc.]
	#			# state = 'Recv'

	#		else:
	#			state = 'Recv'
	#			poll_sensor_window = 0

	return 


# make driver code general enough to think about > 1 sensor
# make communication easy to “switch out” if possible
# have an initial window of specific demographic and sampling baseline data before streaming data
#	  I forget data rate but sensor might sample at like 4hz or something similar
# leave a “denoising” chunk between receiving from sensor and sending it to model possibly
# collect & send data every x time
# packets – fixed sized packets
# check connection periodically, or if message send or receive fails enough, attempt to switch to a new comms
# account for not knowing demographics [ data noise / "attack"? ] 
# ringbuffer

# data preprocessing is probably best left to the Pi, so minimal processing should be done I think


if __name__ == '__main__':
	main()
