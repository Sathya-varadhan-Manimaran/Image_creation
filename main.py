from pattern import Checker
from pattern import Circle
from pattern import Spectrum
import generator
import numpy as np
import matplotlib.pyplot as plt

def main():
    checker_board = Checker(250, 25)
    checker_board.draw()
    checker_board.show()

    circle = Circle(1024, 200, (512, 256))
    circle.draw()
    circle.show()

    spectrum = Spectrum(255)
    spectrum.draw()
    spectrum.show()

    image_gen = generator.ImageGenerator('./exercise_data/', './Labels.json', 12, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    # for i in range(8):
    #     image_gen.next()
    image_gen.show()

if __name__ == '__main__':
    main()
