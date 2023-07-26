import numpy as np
import serial
import time
import h5py
# import datetime
from datetime import datetime
from datetime import date

def getUnixTimestamp():
    # now = datetime.now()
    # total_seconds = now.microsecond/1000000 + now.second + now.minute*60 + now.hour*3600 
    # return total_seconds
    # dt = date.today()  
    # print (date.today(), datetime.now())
    # seconds = datetime.now().timestamp()
    # return (seconds - dt.timestamp()) 
    return np.datetime64(datetime.now()).astype(np.int64) / 1e6  # unix TS in secs and microsecs

def read_scale(serial):
    ser.reset_input_buffer()
    input_string = ser.read(8)
    x = np.frombuffer(input_string, dtype=np.uint8)
    try:
        return int(''.join([chr(i) for i in x]))
    except:
        return 0

def append_data(f, init, block_size, fc, ts, reading):
    if not init:
        sz = [block_size,]
        maxShape = sz.copy()
        maxShape[0] = None
        f.create_dataset('frame_count', (1,), dtype=np.uint32)
        f.create_dataset('ts', tuple(sz), maxshape = tuple(maxShape), dtype = ts.dtype, chunks=True)
        f.create_dataset('scale', tuple(sz), maxshape = tuple(maxShape), dtype = np.int32, chunks=True)
    
    # Check size
    oldSize = f['ts'].shape[0]
    if oldSize == fc:            
        newSize = oldSize + block_size
        f['ts'].resize(newSize, axis=0)
        f['scale'].resize(newSize, axis=0)

    f['frame_count'][0] = fc
    f['ts'][fc] = ts
    f['scale'][fc] = reading

    f.flush()



path = './scale_' + str(time.time()) + '.hdf5'
f = h5py.File(path, 'w')
block_size = 1024
init = False
fc = 0

port = 'COM5'
ser = serial.Serial(port, baudrate=57600, timeout=3.0)
print (ser)
assert ser.is_open, 'Failed to open COM port!'


while ser.is_open:
    fc += 1
    # ts = 1
    # reading = 10
    ts = getUnixTimestamp()
    reading = read_scale(ser)
    append_data(f, init, block_size, fc, ts, reading)
    init = True
    # print (ts, reading)


