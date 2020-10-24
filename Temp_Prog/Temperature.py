

import smbus
import sys
import getopt
import time 
import pigpio

   
def temp():

   i2c_bus = smbus.SMBus(1)
   OMRON_1=0x0a 					# 7 bit I2C address of Omron MEMS Temp Sensor D6T-44L
   OMRON_BUFFER_LENGTH=11			# Omron data buffer size
   temperature_data=[0]*OMRON_BUFFER_LENGTH 	# initialize the temperature data list

   # intialize the pigpio library and socket connection to the daemon (pigpiod)
   pi = pigpio.pi()              # use defaults
   version = pi.get_pigpio_version()
   handle = pi.i2c_open(1, 0x0a) # open Omron D6T device at address 0x0a on bus 1


   
   # initialize the device based on Omron's appnote 1
   result=i2c_bus.write_byte(OMRON_1,0x4c);
   #print 'write result = '+str(result)

   #for x in range(0, len(temperature_data)):
      #print x
      # Read all data  tem
      #temperature_data[x]=i2c_bus.read_byte(OMRON_1)
   (bytes_read, temperature_data) = pi.i2c_read_device(handle, len(temperature_data))

   # Display data 
   
   a=(temperature_data[2]+temperature_data[3]*256)/10
   a = a*9
   a = a/5
   a = a+32
   a = a+15
   print(a)
   #print 'done'
   pi.i2c_close(handle)
   pi.stop()

   
while True:
   temp()
   time.sleep(1)
