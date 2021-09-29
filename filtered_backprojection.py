import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
import scipy.interpolate
import scipy.misc
import scipy.ndimage.interpolation
from scipy.fftpack import fft, fftshift, ifft


def create_phantom_square(resolution):
    S = resolution
    image = np.zeros((S, S))
    mini = S / 4
    maxi = 3 * (S / 4)
    image[int(mini): int(maxi), int(mini): int(maxi)] = 0.5
    return image

def create_phantom_square_double(resolution):
    S = resolution
    image = np.zeros((S, S))

    # create main square
    start = S / 4
    end = 3 * (S / 4)
    image[int(start): int(end), int(start): int(end)] = 0.5

    # create small square
    start = 3 * (S / 4) + 5
    end = 3 * (S / 4) + 35
    image[int(start): int(end), int(start): int(end)] = 0.5

    return image

def create_phantom_shepp_logan(resolution):
    image = shepp_logan_phantom()
    image = resize(image, [resolution, resolution])
    return image


def create_sinogram_groundtruth(image):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title('Original')
    ax1.imshow(image, cmap='gray')

    theta = np.arange(0,180)
    sinogram = radon(image, theta=theta)
    dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]

    ax2.imshow(sinogram, extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy), cmap='gray')
    ax2.set_title("Sinogram")

    fig.tight_layout()
    return sinogram


def create_sinogram_manually(image, noise):

    theta = np.arange(0, 180)
    # Project the sinogram (ie calculate Radon transform)
    sinogram = np.zeros((len(image), len(theta)))
    for i in theta:
        sinogram[:, i] = np.array([
            np.sum(scipy.ndimage.interpolation.rotate(
                    image,
                    -i,  # NB rotate takes degrees argument
                    order=3,
                    reshape=False,
                    mode='constant',
                    cval=0.0
                )
                , axis=0
            )
        ])

        if noise:
            additive_noise = np.random.normal(0, 1, sinogram.shape)
            sinogram = sinogram + additive_noise*0.2

            # Gaussian noise filter
            sinogram = scipy.ndimage.gaussian_filter(sinogram, sigma=0.01)


    return sinogram


def filter_sinogramm(sinogram):

    projection_length, number_angles = sinogram.shape

    # Create ramp filter in frequency domain
    step = 2 * np.pi / projection_length
    omega = np.arange(-np.pi, np.pi, step)
    filter_ramp = abs(omega)
    filter_ramp = fftshift(filter_ramp)

    sinogram_filtered = np.zeros((projection_length, number_angles))

    for i in range(number_angles):
        sinogram_fourier = fft(sinogram[:, i])
        projection_filtered = sinogram_fourier * filter_ramp
        sinogram_filtered[:, i] = np.real(ifft(projection_filtered))

    return sinogram_filtered


def backproject(sinogram, theta):

    imageLen = sinogram.shape[0]
    reconMatrix = np.zeros((imageLen, imageLen))

    x = np.arange(imageLen) - imageLen / 2  # create coordinate system centered at (x,y = 0,0)
    y = x.copy()
    x_mesh, y_mesh = np.meshgrid(x, y)

    theta = theta * np.pi / 180
    numAngles = len(theta)

    for n in range(numAngles):
        x_rot = x_mesh * np.sin(theta[n]) + y_mesh * np.cos(theta[n])  # determine rotated x-coordinate about origin in mesh grid form
        x_rot_cor = np.round(x_rot + imageLen / 2)  # shift back to original image coordinates, round values to make indices
        x_rot_cor = x_rot_cor.astype('int')
        projMatrix = np.zeros((imageLen, imageLen))
        m0, m1 = np.where((x_rot_cor >= 0) & (x_rot_cor <= (imageLen - 1)))  # after rotating, you'll inevitably have new coordinates that exceed the size of the original
        s = sinogram[:, n]  # get projection
        projMatrix[m0, m1] = s[x_rot_cor[m0, m1]]  # backproject in-bounds data
        reconMatrix += projMatrix

    reconMatrix = np.fliplr(reconMatrix)
    backprojArray = np.transpose(reconMatrix)
    return backprojArray


# Define variables
resolution = 256
theta = np.arange(0, 180)
phantoms = {1: "square", 2: "square_double", 3: "shepp-logan"}
phantom = phantoms[2]
methods = {1: "skimage", 2: "picture_rotation", 3: "ray_tracing"}
method = methods[1]

# Create phantom
if phantom == "square":
    image = create_phantom_square(resolution)
elif phantom == "square_double":
    image = create_phantom_square_double(resolution)
elif phantom == "shepp-logan":
    image = create_phantom_shepp_logan(resolution)

# Simulate sinogram (forward-projection)
if method == "skimage":
    sinogram = create_sinogram_groundtruth(image)
elif method == "picture_rotation":
    sinogram = create_sinogram_manually(image, noise=False)

# Filter sinogram using  ramp filter
sinogram_filtered = filter_sinogramm(sinogram)

# Backprojection
output_filtered = backproject(sinogram, theta)
output = backproject(sinogram_filtered, theta)

plot = False
if plot:
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4), gridspec_kw={'width_ratios': [1,1,1], 'height_ratios': [1]})

    ax1.set_title('Original')
    ax1.imshow(image, cmap='gray')

    ax2.set_title(f'Sinogram {method}')
    ax2.imshow(sinogram, cmap='gray')

    ax3.set_title('Filtered Backprojection')
    ax3.imshow(output_filtered, cmap='gray')

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.show()

else:
    fig_input = plt.figure(1)
    plt.imshow(image, cmap='gray')

    fig_sinogramm, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(sinogram, cmap='gray')
    ax2.imshow(sinogram_filtered, cmap='gray')

    fig_output, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(output_filtered, cmap='gray')
    ax2.imshow(output, cmap='gray')

    plt.show()







