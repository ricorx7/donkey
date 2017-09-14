# donkeycar: a python self driving library 

![build status](https://travis-ci.org/wroscoe/donkey.svg?branch=master)

Donkeycar is minimalist and modular self driving library written in Python. It is 
developed for hobbiests and students with a focus on allowing fast experimentation and easy 
community contributions.  

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Vehicle Build Instructions](http://www.donkeycar.com)
* [Software documentation](http://docs.donkeycar.com)
* [Slack / Chat](https://donkey-slackin.herokuapp.com/)

#### Use Donkey if you want to:
* Make an RC car drive its self.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com)
* Experiment with different driving methods.
* Add parts and sensors to your car.
* Log sensor data. (images, user inputs, sensor readings) 
* Drive yoru car via a web or game controler.
* Leverage community contributed driving data.
* Use existing hardware CAD designs for upgrades.

### Getting started. 
After building a Donkey2, here are the steps to start driving.

####Install donkey
```
pip install donkeycar
```
 
 

####Create a car folder.
```
donkey createcar --path ~/d2
```
 
 
 
####Start your car.
```
python ~/d2/manage.py drive --throttle=0.25
```
 
This will start recording data when the throttle is pressed from the controls.
Now you can control your car by going to `<ip_address_of_your_pi>:8887/drive`
You can use the Web controls or a PS3 controller (joystick).
Run around 20 - 30 laps 
The data will be recorded to the ~/d2/data folder.

I find easiest to drive with the throttle maxed.  Set a max throttle to a value that allows you drive always at full throttle.
This is because the timing of when you take a turn and the angle of the turn will vary with speed.  So if your speed fluctuates, your steering will also.

 
 

####Train based off the recorded data
```bash
python ~/d2/manage.py train --model=any_model_name --/full/path/to/data/tub_XXX... 
```

The tub must be the whole path. You can also load multiple tubs separated by a comma.  There cannot be any spaces between each tub path.
The model can be any name.
This will create a model with the given name in the ~/d2/models folder.
 
 
 
 
####Drive Autonomous
```bash
python ~/d2/manage.py drive --model=any_model_name --throttle=0.25
```

This will load the trained model for your drive.  When you first start this, it will be in 'User' mode.  
You will need to change the mode to 'Local_Angle'.  This will cause the steering to be autonomous but the throttle is still controled by the user.
Set the same throttle value as when you trained and max the throttle and let it steer autonomously.
 
 
####View Tensorboard
```bash
python ~/d2/manage.py train --tensorboard=True --model=any_model_name --tubs=/full/path/to/tub_XX_XX-XX-XX 
```
```bash
tensorboard --logdir Graph/
```
While training, files will be generated in ~/d2/Graph, these files can get large.  
After running this command, a URL will be given to view the Tensorflow process.
Refresh the page to see updates in the training.
Clear the ~/d2/Graph folder when starting a new training.
angle_out_loss, throttle_out_loss and loss are based off the training data.
val_angle_out_loss, val_throttle_out_loss and val_loss are based off the validation training set data.



####Modifying Source Code/Reinstall
If you modify the source code, you will need to reinstall the donkeycar.  Run these 
commands to reinstall donkeycar with your modified source code.
```bash
pip uninstall donkeycar
cd /path/to/donkeycar/source/code
python setup.py build
pip install -e /path/to/donkeycar/source/code
```

I created a script file "install_on_desktop.sh" to do all these commands for you.