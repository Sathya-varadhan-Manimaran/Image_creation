import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        if resolution % (2 * tile_size) == 0:
            self.resolution = resolution
            self.tile_size = tile_size
            self.output = None
        else:
            raise ValueError("Resolution must be evenly dividable by (2 * tile_size).")

    def draw(self):
        size = (self.tile_size, self.tile_size)

        black = np.zeros(size) # Matrix with 0 as values with size tile_size x tile_size
        white = np.ones(size) # Matrix with 1 as values with size tile_size x tile_size

        black_white_row = np.concatenate((black, white), axis=1)    #Row with black and white box with tile_size rows and 2*tile_size columns
        white_black_row = np.concatenate((white, black), axis=1)    #Row with white and black box with tile_size rows and 2*tile_size columns

        array = np.concatenate((black_white_row, white_black_row), axis=0) #Array with 2*tile_size x 2*tile_size pixels

        self.output = np.tile(array, (self.resolution // (2 * self.tile_size), self.resolution // (2 * self.tile_size)))

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        x = np.linspace(0, self.resolution, num=self.resolution)    #from 0 to resolution with resolution number of elements
        y = np.linspace(0, self.resolution, num=self.resolution)

        xv, yv = np.meshgrid(x, y)  # array of size resolution x resolution

        radius_vector = np.sqrt((xv - self.position[0]) ** 2 + (yv - self.position[1]) ** 2)    #Equation of circle # Trasforms matrix with radius as value at each element

        self.output = (radius_vector <= self.radius) #1 - if the radius is below or equal to the given radius, 0 - if the radius is above.

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros([self.resolution, self.resolution, 3])

        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)  #Red channel
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)   #Green channel
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)   #Blue channel

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.show()