#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car and train a model for it. 

Usage:
    manage.py drive [--model=<model>] [--web=<True/False>] [--throttle=<Throttle 0.0-1.0>]
    manage.py train (--tub=<tub>) (--model=<model>) [--tensorboard] [--epochs=<number>] [--lr=<Learning Rate>]
    manage.py calibrate
"""


import os
from docopt import docopt
import donkeycar as dk

CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')


def drive(model_path=None, web_control=False, max_throttle=0.40):
    #Initialized car
    V = dk.vehicle.Vehicle()

    # Setup camera
    cam = dk.parts.PiCamera()
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    # Select if only use bluetooth PS3 controller
    # Or web controller
    # Also set the max throttle
    if web_control:
        ctr = dk.parts.LocalWebController()
    else:
        ctr = dk.parts.JoystickPilot(max_throttle=float(max_throttle))
    V.add(ctr,
          inputs=['cam/image_array'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)
    
    # See if we should even run the pilot module.
    # This is only needed because the part run_contion only accepts boolean
    def pilot_condition(mode):
        if mode == 'user':
            return False
        else:
            return True
        
    pilot_condition_part = dk.parts.Lambda(pilot_condition)
    V.add(pilot_condition_part, inputs=['user/mode'], outputs=['run_pilot'])
    
    # Run the pilot if the mode is not user
    # There is also KereasLinear()
    # kl = dk.parts.KerasCategorical()
    kl = dk.parts.KerasNvidaEndToEnd()
    if model_path:
        print(model_path)
        kl.load(model_path)
    
    V.add(kl, inputs=['cam/image_array'], 
          outputs=['pilot/angle', 'pilot/throttle'],
          run_condition='run_pilot')

    # Choose what inputs should change the car.
    def drive_mode(mode, 
                   user_angle, user_throttle,
                   pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        
        else: 
            return pilot_angle, pilot_throttle
        
    drive_mode_part = dk.parts.Lambda(drive_mode)
    V.add(drive_mode_part, 
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'], 
          outputs=['angle', 'throttle'])

    # Configure the throttle and angle control hardware
    # Calibrate min/max for steering angle
    # Calibrate min/max/zero for throttle
    steering_controller = dk.parts.PCA9685(1)
    steering = dk.parts.PWMSteering(controller=steering_controller,
                                    left_pulse=460, right_pulse=260)
    
    throttle_controller = dk.parts.PCA9685(0)
    throttle = dk.parts.PWMThrottle(controller=throttle_controller,
                                    max_pulse=500, zero_pulse=370, min_pulse=220)
    
    V.add(steering, inputs=['angle'])
    V.add(throttle, inputs=['throttle'])
    
    # Add tub to save data
    inputs = ['cam/image_array',
              'user/angle', 'user/throttle',
              'pilot/angle', 'pilot/throttle',
              'user/mode']
    types = ['image_array',
             'float', 'float',
             'float', 'float',
             'str']
    
    th = dk.parts.TubHandler(path=DATA_PATH)
    tub_writer = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub_writer, inputs=inputs, run_condition='recording')
    
    # Run the vehicle for 20 seconds
    V.start(rate_hz=20, max_loop_count=100000)
    
    print("You can now go to <your pi ip address>:8887 to drive your car.")


def train(tub_names, model_name, tensorboard=False, num_epochs=500, lr=1.0e-4):

    # Set the Neural Network type (Categorical or Linear)
    # kl = dk.parts.KerasCategorical()
    kl = dk.parts.KerasNvidaEndToEnd(learning_rate=lr)

    # Set the model name with model path
    model_path = os.path.join(MODELS_PATH, model_name)

    # Set the complete path for each tub listed
    if tub_names:
        tub_paths = [os.path.join(DATA_PATH, n.strip()) for n in tub_names.split(',')]
    else:
        tub_paths = [os.path.join(DATA_PATH, n.strip()) for n in os.listdir(DATA_PATH)]
    tubs = [dk.parts.Tub(p) for p in tub_paths]

    def rt(record):
        record['user/angle'] = dk.utils.linear_bin(record['user/angle'])
        #record['user/throttle'] = dk.utils.linear_bin(record['user/throttle'])      # !!! Possible where to fix throttle
        return record

    # Combine the generators from multiple tubs
    def combined_gen(gens):
        import itertools
        combined_gen = itertools.chain()
        for gen in gens:
            combined_gen = itertools.chain(combined_gen, gen)
        return combined_gen

    X_keys = ['cam/image_array']
    y_keys = ['user/angle', 'user/throttle']

    # Accumulate a generator for each tub
    # X_keys = Training values (Images)
    # y_keys = Labels for Training values (Steering Angle and Throttle)
    # record_transform = Record the results to the record dict
    gens = [tub.train_val_gen(X_keys, y_keys, record_transform=rt, batch_size=128) for tub in tubs]
    train_gens = [gen[0] for gen in gens]
    val_gens = [gen[1] for gen in gens]

    # Train with the data loaded from the tubs
    kl.train(combined_gen(train_gens),
             combined_gen(val_gens),
             num_epochs=num_epochs,
             saved_model_path=model_path,
             tensorboard=tensorboard)


def calibrate():
    channel = int(input('Enter the channel your actuator uses (0-15).'))
    c = dk.parts.PCA9685(channel)
    
    for i in range(10):
        pmw = int(input('Enter a PWM setting to test(100-600)'))
        c.run(pmw)


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['drive']:
        model = args['--model']
        web = args['--web']
        if web is None:
            web = False
        throttle = args['--throttle']
        if throttle is None:
            throttle = 0.25
        drive(model_path=model, web_control=web, max_throttle=throttle)
    elif args['calibrate']:
        calibrate()
    elif args['train']:
        tub = args['--tub']
        model = args['--model']
        tensorboard = args['--tensorboard']
        if tensorboard is None:
            tensorboard = False
        else:
            tensorboard = True
        epochs = args['--epochs']
        if epochs is None:
            epochs = 500
        else:
            epochs = int(epochs)	
        lr = args['--lr']
        if lr is None:
            lr = 1.0e-4
        else:
            lr = float(lr)

        train(tub, model, tensorboard=False, num_epochs=epochs, lr=lr)




