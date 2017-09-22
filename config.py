"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/d2/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')

#VEHICLE
DRIVE_LOOP_HZ = 20
MAX_LOOPS = 100000

#CAMERA
CAMERA_RESOLUTION = (160, 120)
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

#STEERING
STEERING_CHANNEL = 1
STEERING_LEFT_PWM = 420
STEERING_RIGHT_PWM = 360

#THROTTLE
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 400
THROTTLE_STOPPED_PWM = 360
THROTTLE_REVERSE_PWM = 310

#JOYSTICK
JOYSTICK_MAX_THROTTLE=0.25
JOYSTICK_STEERING_SCALE=1

# Web Control
IS_WEB_CONTROL = False

# Model types
MODEL_TYPE_NVIDIA = "NVIDIA"
MODLE_TYPE_CATEGORICAL = "CATEGORICAL"
MODEL_TYPE_LINEAR = "LINEAR"

# Keras settings
TRAINING_MODEL = MODEL_TYPE_NVIDIA          # The type of Neural Network model to use for training.
IS_TENSORBOARD = True                       # Create a Graph directory and record the Tensorboard data to review results
IS_EARLY_STOP = False                       # If the data does not improve, stop training early
LEARNING_RATE = 1.0e-5                      # Learn rate.  Decrease to fix bias
EPOCHS = 20000                              # Number of epochs to run.  The higher the number, the more training
