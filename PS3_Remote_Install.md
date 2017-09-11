Install PS3 controller on a RPi3
References for this information was found here:

https://github.com/RetroPie/RetroPie-Setup/wiki/PS3-Controller

https://github.com/RetroPie/RetroPie-Setup/issues/1128



Install all the dependencies:
```bash
sudo apt-get install bluetooth blueman bluez-hcidump checkinstall libusb-dev libbluetooth-dev joystick pkg-config
```

Connect the PS3 remote to the RPI3 with the USB cable

Check to see if we can see the PS3
```bash
hciconfig
hci0: Type: BR/EDR Bus: USB
 BD Address: 00:1F:81:00:06:20 ACL MTU: 1021:4 SCO MTU: 180:1
UP RUNNING PSCAN
RX bytes:1260 acl:0 sco:0 events:46 errors:0
TX bytes:452 acl:0 sco:0 commands:45 errors:0
```

Pair the PS3 controller to the RPI3
```bash
bluetoothctrl
```

This will start the application and give you a menu.
List the Bluetooth devices available.
```bash
devices
```

Write down the device MAC address
Now turn on agent and
let it trust the mac address
```bash
agent on
trust "mac address"
quit
```

You only have to do this once.  It will remeber the PS3 remote from now on.
Disconnect the USB cable from the PS3 remote.
Press the connect button on the PS3.  You should see it blink all 4 LEDs, then 1 LED which means it is connected. 

To test that the Bluetooth PS3 remote is working, verify that /dev/input/js0 exist.

You can also run this application to view the output.
```bash
sudo jstest /dev/input/js0
```