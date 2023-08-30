"""
This script is used to:
train driver drowsiness detector model to classify diver eyes into 'open' and 'close' classes
"""
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from driver_drowsiness_detector.data_pre_process import ImageGenerator
import os


class ModelTrain(object):
    """
    this class is used to train model to classify driver drowsiness.
    """

    def __init__(self, generator: ImageGenerator):
        """
        class constructor.
        :param generator: image generator to be iterated.
        """
        self.generator = generator
        self.model = None

    def train(self, train_directory, valid_directory):
        """
        train the driver drowsiness classifier model using the training dataset
        :param train_directory: train dataset path
        :param valid_directory: valid dataset path
        """
        if not os.path.exists(train_directory):
            print(f"the train directory {train_directory} does not exist")
            return

        if not os.path.exists(valid_directory):
            print(f"the valid directory {valid_directory} does not exist")
            return

        batch_size = 32
        target_size = (24, 24)
        train_batch = self.generator.generator(train_directory, batch_size=batch_size, target_size=target_size)
        valid_batch = self.generator.generator(valid_directory, batch_size=batch_size, target_size=target_size)
        SPE = len(train_batch.classes) // batch_size
        VS = len(valid_batch.classes) // batch_size
        print(SPE, VS)

        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
            MaxPooling2D(pool_size=(1, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),
            # 32 convolution filters used each of size 3x3
            # again
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(1, 1)),

            # 64 convolution filters used each of size 3x3
            # choose the best features via pooling

            # randomly turn neurons on and off to improve convergence
            Dropout(0.25),
            # flatten since too many dimensions, we only want a classification output
            Flatten(),
            # fully connected to get all relevant data
            Dense(128, activation='relu'),
            # one more dropout for convergence' sake :)
            Dropout(0.5),
            # output a softmax to squash the matrix into output probabilities
            Dense(2, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=SPE,
                                 validation_steps=VS)

    def save(self, path: str):
        """
        save the trained model in given path
        """
        # if not os.path.exists(path):
        #     print(f"the path {path} does not exist")
        #     return

        if self.model:
            self.model.save(path, overwrite=True)
