import numpy as np


def smooth_fct(data, kernel_size=5):
    "Smooth data with convolution of kernel_size"
    # kernel_size=5
    kernel = np.ones(kernel_size) / kernel_size
    convolved_data = np.convolve(data.squeeze(), kernel, mode='same')
    return convolved_data
