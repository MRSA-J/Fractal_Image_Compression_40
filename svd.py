import numpy
from PIL import Image
import matplotlib.pyplot as plt

# Defalt width, height for lena image
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
SINGULAR_VALUE_LIMIT = 160   # number of singular values to use for reconstructing the compressed image

MONKEY_PATH = 'data/monkey.jpg'
LENA_PATH = 'data/lena.jpg'

'''
# @Author: Chen Wei
# - Code modified from tutorial: https://www.youtube.com/watch?v=SU851ljMIZ8
# - Idea: Doing SVD on each channel separately, and merge them together
# - Although not being part of the fractal image compression, the reason why I include SVD here is that we could do some 
# comparison based on different model
'''
def extract_rgb(image_path):
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

# PS: THIS IS FOR RGB ONLY cuz only RGB image can extract_rgb
def svd_image_demo(image_path, limit):
    print('SVD image compression')
    R, G, B, image = extract_rgb(image_path)

    R_Compressed = compress_single_channel(R, limit)
    G_Compressed = compress_single_channel(G, limit)
    B_Compressed = compress_single_channel(B, limit)

    img_red = Image.fromarray(R_Compressed, mode=None)
    img_green = Image.fromarray(G_Compressed, mode=None)
    img_blue = Image.fromarray(B_Compressed, mode=None)

    generated_image = Image.merge("RGB", (img_red, img_green, img_blue))

    plt.imshow(image)
    plt.title("Original Image")
    plt.show()
    plt.imshow(generated_image)
    plt.title("Generated Image using SVD")
    plt.show()

    # if len(image.size) == 3:
    original_size = image.size[0] * image.size[1] * 3
    compressed_size = limit * (1 + image.size[0] + image.size[1]) * 3
    # else:  #grey_scaled, cannot (cuz only RGB image can extract_rgb)
    #     original_size = image.size[0] * image.size[1]
    #     compressed_size = limit * (1 + image.size[0] + image.size[1])
    ratio = compressed_size / original_size * 1.0

    print("Image orginal dimension:", image.size)
    print("Compressed image dimension:", generated_image.size)

    print('Original size:', original_size)
    print('Compressed size:', compressed_size)
    print('Ratio compressed size / original size:', ratio)

def svd_image_grey_scaled_demo(image_path, limit):
    print("Have not implemented yet")
    # Todo: Implement this one

if __name__ == "__main__":
    svd_image_demo(LENA_PATH, SINGULAR_VALUE_LIMIT)
    svd_image_grey_scaled_demo(MONKEY_PATH, SINGULAR_VALUE_LIMIT)




