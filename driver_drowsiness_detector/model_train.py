"""
This script is used to:
train driver drowsiness detector model to classify diver eyes into 'open' and 'close' classes
"""


class ModelTrain(object):
    """
    this class is used to train model to classify driver drowsiness.
    """

    def __init__(self, directory: str):
        """
        class constructor.
        :param directory: image directory to be iterated.
        """
        self.directory = directory

    def train(self):
        """
        train the driver drowsiness classifier model using the training dataset
        """
        pass

    def save(self, path: str):
        """
        save the trained model in given path
        """
        pass
