'''

pilots.py

Methods to create, use, save and load pilots. Pilots 
contain the highlevel logic used to determine the angle
and throttle of a vehicle. Pilots can include one or more 
models to help direct the vehicles motion. 

'''




import os
import numpy as np
import keras
from ... import utils


import donkeycar as dk
from donkeycar import utils

class KerasPilot():
 
    def load(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.model.summary()
    
    def train(self,
              train_gen,
              val_gen,
              saved_model_path,
              num_epochs=500,                   # Nvidia uses 20000
              steps=10,
              tensorboard=False):
        
        """
        train_gen: generator that yields an array of images an array of 
        
        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='loss',
                                                    verbose=1,
                                                    save_best_only=True, 
                                                    mode='min')
        
        #stop training if the validation error stops improving.
        #early_stop = keras.callbacks.EarlyStopping(monitor='loss',
        #                                           min_delta=.0005,
        #                                           patience=4,
        #                                           verbose=1,
        #                                           mode='auto')
        #
        #callbacks_list = [save_best, early_stop]

        callbacks_list = [save_best]

        # Add Tensorboard callback
        if tensorboard:
            print("Using Tensorboard")
	    # This will create a Graph directory
            # Run tensorboard --logdir path_to/Graph
            tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                                     histogram_freq=1,  # Histogram frequency
                                                     write_grads=True,      # Write Histogram, histogram_freq must be greater than 0
                                                     write_graph=True,  # Write graph to describe network
                                                     write_images=True,
                                                     # Write model weights to visualize as image in TensorBoard.
                                                     # embeddings_freq=3,     # Frequency selected embedding layers will be saved
                                                     # embeddings_layer_names=list(embeddings_metadata.keys()),
                                                     # embeddings_metadata=embeddings_metadata
                                                     embeddings_freq=1,
                                                     embeddings_layer_names=['dense_1', 'dense_2'],
                                                     batch_size=5
                                                     )
            callbacks_list = [save_best, tbCallBack]

        hist = self.model.fit_generator(
                        train_gen, 
                        steps_per_epoch=steps, 
                        epochs=num_epochs,
                        verbose=1, 
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=steps*.2)
        return hist


class KerasCategorical(KerasPilot):
    def __init__(self, model=None, *args, **kwargs):
        super(KerasCategorical, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = default_categorical()                              # There is also relu or linear
        
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        #angle_certainty = max(angle_binned[0])
        angle_unbinned = utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle[0][0]
    

class KerasLinear(KerasPilot):
    def __init__(self, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        
    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle, throttle = self.model.predict(img_arr)
        #angle_certainty = max(angle_binned[0])
        return angle[0][0], throttle[0][0]


class KerasNvidaEndToEnd(KerasPilot):
    def __init__(self, model=None, learning_rate=1.0e-4, *args, **kwargs):
        super(KerasNvidaEndToEnd, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            self.model = nvidia_end_to_end(learning_rate=learning_rate)

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle = self.model.predict(img_arr)
        # angle_certainty = max(angle_binned[0])
        angle_unbinned = utils.linear_unbin(angle_binned)
        return angle_unbinned, throttle[0][0]



def default_categorical():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120, 160, 3), name='img_in')                      # First layer, input layer, Shape comes from camera.py resolution, RGB
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)       # 24 features, 5 pixel x 5 pixel kernel (convolution, feauture) window, 2wx2h stride, relu activation
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)       # 32 features, 5px5p kernel window, 2wx2h stride, relu activatiion
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)       # 64 features, 5px5p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)       # 64 features, 3px3p kernal window, 2wx2h stride, relu
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)       # 64 features, 3px3p kernal window, 1wx1h stride, relu

    # Possibly add MaxPooling (will make it less sensitive to position in image).  Camera angle fixed, so may not to be needed

    x = Flatten(name='flattened')(x)                                        # Flatten to 1D (Fully connected)
    x = Dense(100, activation='relu')(x)                                    # Classify the data into 100 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out (turn off) 10% of the neurons (Prevent overfitting)
    x = Dense(50, activation='relu')(x)                                     # Classify the data into 50 features, make all negatives 0
    x = Dropout(.1)(x)                                                      # Randomly drop out 10% of the neurons (Prevent overfitting)
    #categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0
    
    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)      # Reduce to 1 number, Positive number only
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    
    
    model.compile(optimizer='rmsprop',
                  loss={'angle_out': 'categorical_crossentropy', 
                        'throttle_out': 'mean_absolute_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': 0.1})

    return model



def default_linear():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120,160,3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='linear')(x)                                  # Linear vs ReLU, Linear includes negatives
    x = Dropout(.1)(x)
    x = Dense(50, activation='linear')(x)
    x = Dropout(.1)(x)
    #categorical output of the angle
    angle_out = Dense(1, activation='linear', name='angle_out')(x)
    
    #continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    
    
    model.compile(optimizer='rmsprop',
                  loss={'angle_out': 'mean_squared_error', 
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .1})

    return model



def default_relu():
    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    
    img_in = Input(shape=(120,160,3), name='img_in')
    x = img_in
    x = Convolution2D(24, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(32, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (5,5), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(2,2), activation='relu')(x)
    x = Convolution2D(64, (3,3), strides=(1,1), activation='relu')(x)
    
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(.1)(x)
    #categorical output of the angle
    angle_out = Dense(1, activation='relu', name='angle_out')(x)
    
    #continous output of throttle
    throttle_out = Dense(1, activation='relu', name='throttle_out')(x)
    
    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    
    
    model.compile(optimizer='rmsprop',
                  loss={'angle_out': 'mean_squared_error', 
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': .1})

    return model


# Found here:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# https://www.youtube.com/watch?v=EaY5QiZwSP4
# https://github.com/llSourcell/How_to_simulate_a_self_driving_car/blob/master/model.py
def nvidia_end_to_end(keep_prob=0.5, learning_rate=1.0e-4):
    """
    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """

    from keras.layers import Input, Dense, merge
    from keras.models import Model
    from keras.models import Sequential
    from keras.layers import Convolution2D, Lambda, MaxPooling2D, Reshape, BatchNormalization
    from keras.layers import Activation, Dropout, Flatten, Dense
    from keras.optimizers import Adam

    # model = Sequential()
    # model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    # model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    # model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    # model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    # model.add(Convolution2D(64, 3, 3, activation='elu'))
    # model.add(Convolution2D(64, 3, 3, activation='elu'))
    # model.add(Dropout(keep_prob))
    # model.add(Flatten())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dense(1))
    # model.summary()

    def image_normm(val):
        return val / 127.5 - 1.0

    img_in = Input(shape=(120, 160, 3), name='img_in')
    x = img_in
    #x = Lambda(lambda image_norm, input_shape=(120, 160, 3))
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(48, (5, 5), strides=(2, 2), activation='elu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='elu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='elu')(x)

    x = Dropout(keep_prob)(x)
    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='elu')(x)
    x = Dense(50, activation='elu')(x)
    x = Dense(10, activation='elu')(x)

    # categorical output of the angle
    angle_out = Dense(15, activation='softmax', name='angle_out')(x)

    # continous output of throttle
    throttle_out = Dense(1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer=Adam(lr=learning_rate),
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.9, 'throttle_out': 0.1})

    return model
