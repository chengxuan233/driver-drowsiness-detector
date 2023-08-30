"""
This script is used to:
driver image cleaning, labeling, and converting to model training data
"""
from keras.preprocessing.image import ImageDataGenerator


class ImageGenerator(object):
    """
    utility generator for iterating images from given directory.
    """

    def __init__(self):
        """
        generator constructor.
        """
        pass

    def generator(self, directory: str, generator=ImageDataGenerator(rescale=1. / 255), batch_size=32,
                  target_size=(24, 24)) -> ImageDataGenerator:
        """
        return a generator for the given directory.
        :return: image generator
        """
        return generator.flow_from_directory(directory, batch_size=batch_size, target_size=target_size)
