

import time
import sys
import binascii
import struct
from machine import UART
import TMC_2209_reg as reg


#-----------------------------------------------------------------------
# TMC_UART
#
# this class is used to communicate with the TMC via UART
# it can be used to change the settings of the TMC.
# like the current or the microsteppingmode
#-----------------------------------------------------------------------
class TMC_UART:

    mtr_id = 0
    ser = None
    rFrame  = [0x55, 0, 0, 0  ]
    wFrame  = [0x55, 0, 0, 0 , 0, 0, 0, 0 ]
    communication_pause = 500
    error_handler_running = False
    

#-----------------------------------------------------------------------
# constructor
#-----------------------------------------------------------------------
    def __init__(self, serialport=0, baudrate=57600, mtr_id = 0):
        try:
            self.ser = UART(serialport, baudrate, bits=8, parity=None, stop=1) ## adjust per baud and hardware. Sequential reads without some delay fail. default pinout for the uare, for uart 0 it's pin 1 for tx and 2 for rx, don't forget the 1k resistor
        except Exception as e:
            errnum = e.args[0]
            print("TMC2209: SERIAL ERROR: "+str(e))
            if(errnum == 2):
                print("TMC2209: "+str(serialport)+" does not exist. You need to activate the serial port with \"sudo raspi-config\"")
            if(errnum == 13):
                print("TMC2209: you have no permission to use the serial port. You may need to add your user to the dialout group with \"sudo usermod -a -G dialout pi\"")

        self.mtr_id = mtr_id
   
        self.communication_pause = 500/baudrate     # adjust per baud and hardware. Sequential reads without some delay fail.

        self.ser.write("test") 
        self.ser.read(12) #just to clear the buffer because there seems to be no other way
        
    
        

#-----------------------------------------------------------------------
# this function calculates the crc8 parity bit
#-----------------------------------------------------------------------
    def compute_crc8_atm(self, datagram, initial_value=0):
        crc = initial_value
        # Iterate bytes in data
        for byte in datagram:
            # Iterate bits in byte
            for _ in range(0, 8):
                if (crc >> 7) ^ (byte & 0x01):
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
                # Shift to next bit
                byte = byte >> 1
        return crc
    

#-----------------------------------------------------------------------
# reads the registry on the TMC with a given address.
# returns the binary value of that register
#-----------------------------------------------------------------------
    def read_reg(self, register):
        
        self.ser.read(12) #just to clear the buffer because there seems to be no other way
        
        
        self.rFrame[1] = self.mtr_id
        self.rFrame[2] = register
        self.rFrame[3] = self.compute_crc8_atm(self.rFrame[:-1])

        rtn = self.ser.write(bytearray(self.rFrame))
        if rtn != len(self.rFrame):
            print("TMC2209: Err in write {}".format(__), file=sys.stderr)
            return False

        time.sleep(self.communication_pause)  # adjust per baud and hardware. Sequential reads without some delay fail.
        
        rtn = self.ser.read(12) #aparently the only way to clear the input buffer is to read it and discard the contents, it gets a copy of whatever was just sent so you have to do this.
        #print("received "+str(len(rtn))+" bytes; "+str(len(rtn)*8)+" bits")
        #print(rtn.hex())

        time.sleep(self.communication_pause)
        
        return(rtn)
        #return(rtn)


#-----------------------------------------------------------------------
# this function tries to read the registry of the TMC 10 times
# if a valid answer is returned, this function returns it as an integer
#-----------------------------------------------------------------------
    def read_int(self, register, tries=10):
        while(True):
            tries -= 1
            rtn = self.read_reg(register)
            rtn_data = rtn[7:11]
            not_zero_count = len([elem for elem in rtn if elem != 0])
            
            if(len(rtn)<12 or not_zero_count == 0):
                print("TMC2209: UART Communication Error: "+str(len(rtn_data))+" data bytes | "+str(len(rtn))+" total bytes")
            elif(rtn[11] != self.compute_crc8_atm(rtn[4:11])):
                print("TMC2209: UART Communication Error: CRC MISMATCH")
            else:
                break
                        
            if(tries<=0):
                print("TMC2209: after 10 tries not valid answer")
                print("TMC2209: snd:\t"+str(bytes(self.rFrame)))
                print("TMC2209: rtn:\t"+str(rtn))
                self.handle_error()
                return -1
        
        val = struct.unpack(">i",rtn_data)[0]
        return(val)


#-----------------------------------------------------------------------
# this function can write a value to the register of the tmc
# 1. use read_int to get the current setting of the TMC
# 2. then modify the settings as wished
# 3. write them back to the driver with this function
#-----------------------------------------------------------------------
    def write_reg(self, register, val):
        time.sleep_ms(100) #not clear if this helps any
        rtn = self.ser.read(12)# it would be good to clear the input buffer too, but I have no clue how to do that except to wait for the timeout to expire
        
        self.wFrame[1] = self.mtr_id
        self.wFrame[2] =  register | 0x80;  # set write bit
        
        self.wFrame[3] = 0xFF & (val>>24)
        self.wFrame[4] = 0xFF & (val>>16)
        self.wFrame[5] = 0xFF & (val>>8)
        self.wFrame[6] = 0xFF & val
        
        self.wFrame[7] = self.compute_crc8_atm(self.wFrame[:-1])


        rtn = self.ser.write(bytearray(self.wFrame))
        if rtn != len(self.wFrame):
            print("TMC2209: Err in write {}".format(__), file=sys.stderr)
            return False

        time.sleep(self.communication_pause)

        return True


#-----------------------------------------------------------------------
# this function als writes a value to the register of the TMC
# but it also checks if the writing process was successfully by checking
# the InterfaceTransmissionCounter before and after writing
#-----------------------------------------------------------------------
    def write_reg_check(self, register, val, tries=10):
        ifcnt1 = self.read_int(reg.IFCNT)
        
        while(True):
            self.write_reg(register, val)
            tries -= 1
            ifcnt2 = self.read_int(reg.IFCNT) # for some reason IFCNT seems to get reset sometimes
            if(ifcnt1 >= ifcnt2):
                print("TMC2209: writing not successful!")
                print("TMC2209: ifcnt:",ifcnt1,ifcnt2)
            else:
                return True
            if(tries<=0):
                print("TMC2209: after 10 tries not valid write access")
                self.handle_error()
                return -1


#-----------------------------------------------------------------------
# this function clear the communication buffers of the Raspberry Pi
#-----------------------------------------------------------------------
    def flushSerialBuffer(self):
        self.ser.read(12) #just to clear the buffer because there seems to be no other way
        #there is no way to clear the output buffer apparently, still not ready


#-----------------------------------------------------------------------
# this sets a specific bit to 1
#-----------------------------------------------------------------------
    def set_bit(self, value, bit):
        return value | (bit)


#-----------------------------------------------------------------------
# this sets a specific bit to 0
#-----------------------------------------------------------------------
    def clear_bit(self, value, bit):
        return value & ~(bit)


#-----------------------------------------------------------------------
# error handling
#-----------------------------------------------------------------------
    def handle_error(self):
        if(self.error_handler_running):
            return
        self.error_handler_running = True
        gstat = self.read_int(reg.GSTAT)
        print("TMC2209: GSTAT Error check:")
        if(gstat == -1):
            print("TMC2209: No answer from Driver")
        elif(gstat == 0):
            print("TMC2209: Everything looks fine in GSTAT")
        else:
            if(gstat & reg.reset):
                print("TMC2209: The Driver has been reset since the last read access to GSTAT")
            if(gstat & reg.drv_err):
                print("TMC2209: The driver has been shut down due to overtemperature or short circuit detection since the last read access")
            if(gstat & reg.uv_cp):
                print("TMC2209: Undervoltage on the charge pump. The driver is disabled in this case")
        print("EXITING!")
        raise SystemExit


#-----------------------------------------------------------------------
# test UART connection
#-----------------------------------------------------------------------
    def test_uart(self, register):
        
        self.ser.read(12) #just to clear the buffer because there seems to be no other way
        
        
        self.rFrame[1] = self.mtr_id
        self.rFrame[2] = register
        self.rFrame[3] = self.compute_crc8_atm(self.rFrame[:-1])

        rtn = self.ser.write(bytearray(self.rFrame))
        if rtn != len(self.rFrame):
            print("TMC2209: Err in write {}".format(__), file=sys.stderr)
            return False

        time.sleep(self.communication_pause)  # adjust per baud and hardware. Sequential reads without some delay fail.
        
        rtn = self.ser.read(12)
        print("TMC2209: received "+str(len(rtn))+" bytes; "+str(len(rtn)*8)+" bits")
        print("TMC2209: hex: ", binascii.hexlify(rtn))
       

        time.sleep(self.communication_pause)
        
        return bytes(self.rFrame), rtn