import TMC_2209_uart
import time
import binascii
import TMC_2209_reg as reg
driver = TMC_2209_uart.TMC_UART()


print(driver.read_int(reg.IFCNT))
driver.test_uart(reg.VACTUAL)
time.sleep_ms(100)
driver.write_reg_check(reg.VACTUAL, 60000)
print(driver.read_int(reg.VACTUAL))

#driver.test_uart(reg.VACTUAL)
#driver.write_reg_check(reg.VACTUAL, 500) # the checking part of this doesn't seem to work for some reason