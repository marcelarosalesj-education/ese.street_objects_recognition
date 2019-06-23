import time
import serial
import pynmea2

port = "/dev/ttyAMA0"
 
ser = serial.Serial(port, baudrate = 9600, timeout = 0.5)
 
while 1:
    try:
        data = ser.readline()
        data = str(data, 'utf-8')
    except Exception as e:
        print("Exception: {}".format(e))
    
    if 'GGA' in data:
        msg = pynmea2.parse(data)
        print("msg: {} type: {}".format(msg, type(msg)))
        latval = msg.latitude
        print("lat: {}".format(latval))
        longval = msg.longitude
        print("long: {}".format(longval))
        time.sleep(0.5) 
