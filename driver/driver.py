import sys
import math
import time
from packet import Packet
from groveGSRsensor import GroveGSRSensor
from grove.adc import ADC

## General Skeleton meant to be robust and easy to build on. ##

##########################

#### GLOBAL VARIABLES ####

cxn_timeout 		 = 120   		# Connection Timeout -- try and connect to any device for 2 min -- placeholder value 
check_cxn_timer 	 = 300    		# check connection every 300 s  -- placeholder value
baseline_timer		 = 240			# Timer to measure baseline

poll_gsr_window = 1 / 4.0 			# get reading from sensor every x units
send_gsr_window = 30					# send reading to Pi every 30 sec

ringbuffer = 0

##########################

#### COMMS FUNCTIONS ####

def init_wifi():  # Initialize Wifi Comms Protocol
	return 0
def init_bt(): 	  # Initialize Bluetooth Comms Protocol
	return 0
def init_radio(): # Initialize Radio Comms Protocol
	return 0
def init_plug():  # Initialize Wired Pi Cxn Protocol
	# Defaul:
	return 1

def init_connect(cxn_type=3):

	ready = 0

	if (cxn_type == 0): 		# WiFi
		if(init_wifi() == 1):
			ready = 1

	elif (cxn_type == 1): 		# Bluetooth
		if(init_bt() == 1):
			ready = 1

	elif (cxn_type == 2): 		# Radio
		if(init_radio() == 1):
			ready = 1

	elif(cxn_type == 3): 		# Plug
		if(init_plug() == 1):
			ready = 1

	return ready

def choose_cxn_device(cxn_type=3):

	## Try and Connect to a Device ##

	# Could design a spin wait / polling to try and connect within a time window
	if(init_connect(cxn_type) == 0):    # Device tried did not connect -- Try connecting to all devices if first attempt fails
		if(init_connect(0) == 0):       # Try Wifi
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


def chck_wifi():  # Check Wifi Cxn
	return 0
def chck_bt(): 	  # Check BT Cxn
	return 0
def chck_radio(): # Check Radio Cxn
	return 0
def chck_plug():  # Check Wired Pi Cxn
	return 1

def check_connect(cxn_type=3):

	cxd = 0

	# Send a Msg & Get ACK Back

	if (cxn_type == 0): 		# WiFi
		if(init_wifi() == 1):
			cxd = 1

	elif (cxn_type == 1): 		# Bluetooth
		if(init_bt() == 1):
			cxd = 1

	elif (cxn_type == 2): 		# Radio
		if(init_radio() == 1):
			cxd = 1

	elif(cxn_type == 3): 		# Plug
		if(init_plug() == 1):
			cxd = 1

	return cxd


##########################

#### SENSOR FUNCTIONS ####

def init_gsr(channel=0):  	# Connect to GSR Sensor
	sensor = GroveGSRSensor(channel)
	# Ensure sensor is connected

	return sensor

##########################
#### MAIN FUNCTION ####

def main():

	## Initialize Sensors ##

	channel = 0 # ADC Channel
	gsr_sensor = init_gsr(channel)

	## Intialize Connection ## 
	connection_type = 3 # 0 W, 1 B, 2 R, 3 P
	connection_type = choose_cxn_device(connection_type) # Default to Plug or have some sort of decision var?

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

	start_gsr_recv = time.time()
	start_gsr_send = time.time()

	state = 'Sample' # Define State

	## Begin Sampling Period ##

	while( (time.time()-start_gsr_recv) < baseline_time ):

		# Recv
		if( (time.time() - start_gsr_recv) >= poll_gsr_window): # Accumlate baseline data
			gsr_reading = gsr_sensor.read_GSR()
			gsr_reading = gsr_sensor.adc_to_us(gsr_reading)
			if not gsr_reading: 
				pass # try again?
			# accumulate into a data structure
			# format Packet
			start_gsr_time = time.time()

		# Send
		if( (time.time()-start_gsr_send) >= send_gsr_window): # Send accumulated data
			# Send data to Pi
			start_gsr_send = time.time()


	## Stream Data ##

	start_gsr_recv = time.time()
	start_gsr_send = time.time()
	start_cxn_chck = time.time()

	state = 'Recv'

	# Init Packet

	while True:

		# Check Connection Periodically #
		if( (time.time() - start_cxn_chck) >= check_cxn_timer ):
			# Check cxn to device
			connected = check_connect(connection_type)
			if not connected: choose_connect(connection_type)
			start_cxn_chck = time.time()

		if state == 'Recv':
			# Receive data from the sensor
			if( (time.time() - start_gsr_recv) >= poll_gsr_window):
				gsr_reading = gsr_sensor.read_GSR()
				gsr_reading = gsr_sensor.adc_to_us(gsr_reading)
				if not gsr_reading: 
					pass # try again?
				# accumulate into a data structure
				# format Packet
				start_gsr_time = time.time()

			if( (time.time() - start_gsr_send) >= send_gsr_window): # Send accumulated data
				state = 'Send'
				start_gsr_send = time.time()

		elif state == 'Send':
			# Send data to Pi
			# Receive ACK from Pi, else Try to Send Again [check cxn, etc.]
			# Init a new Packet
			state = 'Recv'

		else:
			state = 'Recv'
			poll_gsr_window = time.time()

	return 

########################

# If connect to sensor but can't connect to device -> batch results to a max space?
# phone notification?
# make driver code general enough to think about > 1 sensor
# make communication easy to “switch out” if possible
# have an initial window of specific demographic and sampling baseline data before streaming data
# leave a “denoising” chunk between receiving from sensor and sending it to model possibly
# collect & send data every x time
# packets – fixed sized packets
# check connection periodically, or if message send or receive fails enough, attempt to switch to a new comms
# account for not knowing demographics [ data noise / "attack"? ] 
# ringbuffer
# ** multithread capabilities to send and recevie data at same time? **

# data preprocessing is probably best left to the Pi, so minimal processing should be done I think

if __name__ == '__main__':
	main()
