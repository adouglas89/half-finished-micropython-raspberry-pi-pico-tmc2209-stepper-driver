# half-finished-micropython-raspberry-pi-pico-tmc2209-stepper-driver
This appears to be able to read, but not write registers. The counter that indicates a register was written increments, but when you read it back it's unchanged.  It's possible my board was borked or something.I'm putting it up to help the next person.
See the TMC2209 datasheet for mor information on how things work.
Read the code to determine the names of the different registers etc.
I halted the work on this when it became clear the TMC2209 was not capable of driving the motor quietly anyway no matter what settings I used.  This was determined by using an arduino and the libraries for it.
It is not very flexible and probably full of bugs.
A better way to flush the serial buffers is in order, for instance.
Things were very erratic in my testing, sometimes it worked and sometimes it didn't.  The device appeared to reset randomly, too.
Be very careful not to disconnect power or anything, these boards appear to be extremely fragile, I destroyed many of them accidentally for reasons I am still not clear on.  It's probably a good idea to solder all connections that carry much current, and you have to use a capacitor of 100 microfarads at least across the vmot and ground.
Also the driver doesn't work unless it has it's vmot voltage.  Some boards have solder pads that need a solder bridge to connect the uart pin to the actual chip, like a jumper.
Read the description carefully of the other github project I link to below, it has many important warnings and points.
This is derived from this library, which is in turn in development : https://github.com/Chr157i4n/TMC2209_Raspberry_Pi
