# Controller Parts

## Local Web Controller

The default controller to drive the car with your phone or browser. This has a web live preview of camera. Control options include:

Install all the dependencies:
```bash
sudo apt-get install bluetooth blueman bluez-hcidump checkinstall libusb-dev libbluetooth-dev joystick pkg-config
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


### Charging PS3 Sixaxis Joystick

For some reason, they don't like to charge in a powered usb port that doesn't have an active bluetooth control and os driver. So a phone type usb charger won't work. Try a powered linux or mac laptop usb port. You should see the lights blink after plugging in and hitting center PS logo.

After charging, you will need to plug-in the controller again to the Pi, hit the PS logo, then unplug to pair again.

### New Battery for PS3 Sixaxis Joystick

Sometimes these controllers can be quite old. Here's a link to a [new battery](http://a.co/5k1lbns). Be careful when taking off the cover. Remove 5 screws. There's a tab on the top half between the hand grips. You'll want to split/open it from the front and try pulling the bottom forward as you do. Or you'll break the tab off as I did.