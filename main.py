import matplotlib.pyplot as plt
from PIL import Image
import utils
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import random

DIR_NAME = "img"
# size of square used to GLCM
PATCH_SIZE = 21
NUMBER_OF_SAMPLES = 5
RANDOM = True

# Get the list of textures
img_names = utils.get_images_names_from_dir(DIR_NAME)

image_features = {}

for img_idx, img_name in enumerate(img_names):

    # loading images
    img = Image.open('img/' + img_name).convert('L')
    WIDTH, HEIGHT = img.size
    image_data = list(img.getdata())
    image = [image_data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    # convert to numpy array
    image = np.array([np.array(xi) for xi in image])

    patch_locations = []
    # select random patches equal to number of samples
    if RANDOM:
        for i in range(NUMBER_OF_SAMPLES):
            x_rand = random.randrange(0, WIDTH - PATCH_SIZE, 1)
            y_rand = random.randrange(0, HEIGHT - PATCH_SIZE, 1)
            patch_locations.append((x_rand, y_rand))
    else:
        locations = np.linspace(10, WIDTH - PATCH_SIZE - 10, num=NUMBER_OF_SAMPLES, dtype=int)
        for location in locations:
            patch_locations.append((location, location))

    img_patches = []
    for loc in patch_locations:
        img_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])

    print(type(patch_locations))
    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in img_patches:
        glcm = greycomatrix(patch, distances=[4], angles=[90], levels=256,
                            symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])
    image_features[img_name] = (xs, ys)
    print(xs)
    print(ys)
    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    ax = fig.add_subplot(3, 2, 1)
    ax.imshow(image, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    for (y, x) in patch_locations:
        ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    ax.set_xlabel('Original Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('image')

    # for each patch, plot (dissimilarity, correlation)
    ax = fig.add_subplot(3, 2, 2)
    ax.plot(xs, ys, 'go',
            label='Patches')
    ax.set_xlabel('GLCM Dissimilarity')
    ax.set_ylabel('GLCM Correlation')
    ax.legend()

    # display the image patches
    for i, patch in enumerate(img_patches):
        ax = fig.add_subplot(3, len(img_patches), len(img_patches)*1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        ax.set_xlabel('Patch %d' % (i + 1))

    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    plt.tight_layout()
    if img_idx != len(img_names) -1:
        plt.show(block=False)
    else:
        plt.show()
print(image_features)


def draw_all_data_graph(image_features, img_names):
    # create the figure
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot()
    # for each patch, plot (dissimilarity, correlation)
    # for img_idx, img_name in enumerate(img_names):

        # ax2.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
        #         label='Grass')
        # ax2.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
        #         label='Sky')
        # ax.set_xlabel('GLCM Dissimilarity')
        # ax.set_ylabel('GLCM Correlation')
        # ax.legend()
