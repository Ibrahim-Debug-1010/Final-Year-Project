from sympy import integer_log
import os,time
import subprocess
 
 
AS726X_ADDR = 0x49 #7-bit unshifted default I2C Address
 
#AS7263 Registers
AS726x_DEVICE_TYPE = 0x00
AS726x_HW_VERSION = 0x01
AS726x_CONTROL_SETUP = 0x04
AS726x_INT_T = 0x05
AS726x_DEVICE_TEMP = 0x06
AS726x_LED_CONTROL = 0x07
 
AS72XX_SLAVE_STATUS_REG = 0x00
 
AS7263_R = 0x08
AS7263_S = 0x0A
AS7263_T = 0x0C
AS7263_U = 0x0E
AS7263_V = 0x10
AS7263_W = 0x12
 
AS7263_R_CAL = 0x14
AS7263_S_CAL = 0x18
AS7263_T_CAL = 0x1C
AS7263_U_CAL = 0x20
AS7263_V_CAL = 0x24
AS7263_W_CA =0x28
 
AS72XX_SLAVE_TX_VALID = 0x02
AS72XX_SLAVE_RX_VALID = 0x01
 
SENSORTYPE_AS726 = 0x3F
 
POLLING_DELAY = 5 #Amount of ms to wait between checking for virtual register changes
 
# ------------------------------------------------------------------------
# Global variables
# ------------------------------------------------------------------------
 
# None
 
# ------------------------------------------------------------------------
# Functions / Classes
# ------------------------------------------------------------------------
 
class IR_SENSOR():
    """ IR Sensor """
    i2c_address = None
    mode = None
    gain = None
    integration_time = None
 
    def __init__(self, bus=2, address=AS726X_ADDR, mode=3, gain=3,
                integration_time=50):
        if mode not in [0, 1, 2, 3]:
            raise ValueError('Invalid Mode')
        self.mode = mode
        self.gain = gain
        self.integration_time = integration_time
        self.write_i2c_command = "i2cset -y {0} {1}".format(bus, address)
        self.read_i2c_command = "i2cget -y {0} {1}".format(bus, address)
 
    # End def
 
    def read_byte(self, data_address):
        """ Reads byte at specified data address of IR sensor """
 
        command = self.read_i2c_command + " {0}".format(data_address)
        proc=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, )
        output=proc.communicate()[0]
 
        return output
 
    # End def
 
    def write_byte(self, data_address, value):
        """ Write byte to specified data address of IR sensor """
 
        command = self.write_i2c_command + " {0} {1}".format(data_address, value)
        proc=subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, )
        output=proc.communicate()[0]
 
        return output
 
    # End def
 
    #### WIP ####
    def init_device(self):
        self._sensor_version = self.virtual_read_register(AS726x_HW_VERSION) # How to read virtual register in Linux?
        if (self._sensor_version != 0x3E) & (self._sensor_version != 0x3F):
            raise ValueError("Wrong sensor version {}. Should be 0x3E or 0x3F".format(self._sensor_version))
 
        self.set_bulb_current(0)
        self.disable_bulb()
        self.set_indicator_current(0b11)
        self.disable_indicator_led()
        self.set_integration_time(self._integration_time)
        self.set_gain(self._gain)
        self.set_measurement_mode(self._mode)
 
    # End def
 
    
    #### WIP ####
    def set_measurement_mode(self, mode):
    # Sets the measurement mode
    # Mode 0: Continuous reading of VBGY (7262) / STUV (7263)
    # Mode 1: Continuous reading of GYOR (7262) / RTUX (7263)
    # Mode 2: Continuous reading of all channels (power-on default)
    # Mode 3: One-shot reading of all channels
        if (mode > 0b11):
            mode = 0b11
        value = self.virtual_read_register(AS726x_CONTROL_SETUP)
        value = value & 0b11110011
        value = value | (mode << 2) #Set BANK bits with user's choice
        self.virtual_write_register(AS726x_CONTROL_SETUP, value) # How to write virtual register in Linux
        self._mode = mode
 
    # End def
 
    #### WIP ####
    def take_measurements(self):
        # Clear DATA_RDY flag when using Mode 3
        self.clear_data_available()
 
        # Goto mode 3 for one shot measurement of all channels
        self.set_measurement_mode(3);
 
        #Wait for data to be ready
        while self.data_available() == False:
            time.sleep_ms(POLLING_DELAY)
 
        r_val = self.read_byte(AS7263_R)
        s_val = self.read_byte(AS7263_S)
        t_val = self.read_byte(AS7263_T)
        u_val = self.read_byte(AS7263_U)
        v_val = self.read_byte(AS7263_V)
        w_val = self.read_byte(AS7263_W)
 
        return [r_val, s_val, t_val, u_val, v_val, w_val]
 
    # End def