"""
This script is used to:
driver image cleaning, labeling, and converting to model training data
"""
from keras.preprocessing.image import ImageDataGenerator


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

    def generator(self, generator=ImageDataGenerator(rescale=1. / 255), target_size=(24, 24)) -> ImageDataGenerator:
        """
        return a generator for the given directory.
        :return: image generator
        """
        return generator.flow_from_directory(self.directory, target_size)
