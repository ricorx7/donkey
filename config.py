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
USE_JOYSTICK_AS_DEFAULT = True
JOYSTICK_MAX_THROTTLE = 0.25
JOYSTICK_STEERING_SCALE = 1
AUTO_RECORD_ON_THROTTLE = True

# Web Control
IS_WEB_CONTROL = False

#TRAINING
BATCH_SIZE = 128
TRAIN_TEST_SPLIT = 0.8

# Model types
MODEL_TYPE_NVIDIA = "NVIDIA"
MODEL_TYPE_CATEGORICAL = "CATEGORICAL"
MODEL_TYPE_LINEAR = "LINEAR"

# Keras settings
TRAINING_MODEL = MODEL_TYPE_CATEGORICAL     # The type of Neural Network model to use for training.
IS_TENSORBOARD = True                       # Create a Graph directory and record the Tensorboard data to review results
IS_EARLY_STOP = False                       # If the data does not improve, stop training early
LEARNING_RATE = 1.0e-4                      # Learn rate.  Decrease to fix bias
EPOCHS = 20000                              # Number of epochs to run.  The higher the number, the more training

# Nvidia End-To-End Paper
INVERT_STEERING_ANGLE = False                # It is suggest to invert the steering angle when running through the CNN.  1/r smoothly transitions through zero from left turns (negative values) to right turns (positive values).
FPS = 20                                    # DEFAULT: 20.  A higher sampling rate would result in including images that are highly similar and thus not provide much useful information.

