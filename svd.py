import numpy
from PIL import Image
import matplotlib.pyplot as plt # plotting library

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

'''
# @Author: Chen Wei
# - Code modified from tutorial: https://www.youtube.com/watch?v=SU851ljMIZ8
# - Idea: Doing SVD on each channel separately, and merge them together
# - Although not being part of the fractal image compression, the reason why I include SVD here is that we could do some 
# comparison based on different model
'''
def rgb_image(image_path):
    image = Image.open(image_path)
    image_arr = numpy.array(image)
    R = image_arr[:, :, 0]
    G = image_arr[:, :, 1]
    B = image_arr[:, :, 2]

    return [R, G, B, image]

# compress the matrix of a single channel
def compress_single_channel(channel, limit):  #limit - singular value limit
    U, S, V = numpy.linalg.svd(channel)
    compressed = numpy.zeros((channel.shape[0], channel.shape[1]))
    left = numpy.matmul(U[:, 0:limit], numpy.diag(S)[0:limit, 0:limit])
    inner = numpy.matmul(left, V[0:limit, :])
    compressed = inner.astype('uint8')
    return compressed


if __name__ == "__main__":
    print('SVD image compression')
    R, G, B, image = rgb_image('data/lena.jpg')

    # number of singular values to use for reconstructing the compressed image
    singular_limit = 160

    R_Compressed = compress_single_channel(R, singular_limit)
    G_Compressed = compress_single_channel(G, singular_limit)
    B_Compressed = compress_single_channel(B, singular_limit)

    img_red = Image.fromarray(R_Compressed, mode=None)
    img_green = Image.fromarray(G_Compressed, mode=None)
    img_blue = Image.fromarray(B_Compressed, mode=None)

    generated_image = Image.merge("RGB", (img_red, img_green, img_blue))

    # Todo: add legency
    plt.imshow(image)
    plt.show()
    plt.imshow(generated_image)
    plt.show()

    original_size = IMAGE_HEIGHT * IMAGE_WIDTH * 3
    compressed_size = singular_limit * (1 + IMAGE_HEIGHT + IMAGE_WIDTH) * 3
    ratio = compressed_size / original_size * 1.0

    print('original size:', original_size)
    print('compressed size:', compressed_size)
    print('Ratio compressed size / original size:', ratio)

