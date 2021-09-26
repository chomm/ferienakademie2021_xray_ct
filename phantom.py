import numpy as np
from matplotlib import pyplot as plt
from bresenham import bresenham
from skimage.transform import rotate
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)

#TODO create class for phantom

# object format: [(middle, hight, width), ...]
def create_phantom(res):
    # Generate 2D and 3D structuring elements.
    struc_2d = {
        "square(15)": square(15),
        "rectangle(15, 10)": rectangle(15, 10),
        "diamond(7)": diamond(7),
        "disk(7)": disk(7),
        "octagon(7, 4)": octagon(7, 4),
        "star(5)": star(5)
    }

    frame = np.zeros((res, res))
    phantom = np.ones((res/8,res/8))
    for phantom in phantoms:
        for i in range(round(phantom[0]-phantom[1]), round(phantom[0]+phantom[1])):
            for j in range(round(phantom[0]-phantom[1]), round(phantom[0]+phantom[1])):
                frame[i][j] = 1
    rotate(phantom, 2)
    print(frame)
    plt.imshow(frame)
    plt.show()
    return frame

def reconstruction(size, angle):
    x = round(np.tan(angle * np.pi / 180) * size/2)
    return x

def get_intersect_coordinates(size, angel):
    lower_x = reconstruction(size, angel)
    lower_y = int(size / 2)
    upper_x = -reconstruction(size, angel)
    upper_y = int(-size / 2)
    index = list(bresenham(lower_x, lower_y, upper_x, upper_y))
    return index

def calculate_integral_value(anlge):
    phantom = create_phantom([(25, 5, 5), (40, 2.5, 2.5)])
    indices = get_intersect_coordinates(50, angle)
    #print(indices)
    for idx, indice in enumerate(indices):
        indice = tuple((indice[0] + 24, int(indice[1] + 24)))
        indices[idx] = indice
    print(indices)
    sum = 0
    for index in indices:
        sum = sum + phantom[index[0]][index[1]]
    return sum

if __name__ == '__main__':
    resolution = 256
    phantom = create_phantom(resolution)
    '''attenuation_coefficients = []
    for angle in range(45):
        attenuation = calculate_integral_value(angle)
        attenuation_coefficients.append(attenuation)
    print(attenuation_coefficients)
    plt.plot(attenuation_coefficients)
    plt.show()'''












