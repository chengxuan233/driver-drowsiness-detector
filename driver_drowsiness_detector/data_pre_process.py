"""
This script is used to:
driver image cleaning, labeling, and converting to model training data
"""
from keras.preprocessing import image


class ImageGenerator(object):
    """
    utility generator for iterating images from given directory.
    """

    def __init__(self, directory: str):
        """
        generator constructor.
        :param directory: image directory to be iterated.
        """
        self.directory = directory

    def generator(self) -> image.ImageDataGenerator:
        """
        return a generator for the given directory.
        :return: image generator
        """
        pass
