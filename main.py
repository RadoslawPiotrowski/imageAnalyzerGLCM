import matplotlib.pyplot as plt
from PIL import Image
import utils
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import random
import math
import csv
import matplotlib.patches as patches

DIR_NAME = "img"
DIR_TESTING_NAME = "img_testing"
# size of square used to GLCM
PATCH_SIZE = 40
NUMBER_OF_SAMPLES = 32
RANDOM = False
OFFSETS = [8]
ANGLES = [0]
lerned_centroids = []
concentrations = []
accuracies = []


def listToString(s):
    str1 = ""
    for idx, ele in enumerate(s):
        if idx != len(s) - 1:
            str1 += str(ele) + ";"
        else:
            str1 += str(ele)
    return str1


OFFSET_STRING = listToString(OFFSETS)
ANGLES_STRING = listToString(ANGLES)
# Get the list of textures
img_names = utils.get_images_names_from_dir(DIR_NAME)
img_testing_names = utils.get_images_names_from_dir(DIR_TESTING_NAME)
image_features = {}


def draw_all_data_graph(image_features, img_names):
    # create the figure
    fig = plt.figure()
    # Mapping colors
    cmap = utils.get_cmap(len(img_names) + 1)
    # for each patch, plot (dissimilarity, correlation)
    for img_idx, img_name in enumerate(img_names):
        color = cmap(img_idx)
        x_val = image_features[img_name][0]
        y_val = image_features[img_name][1]
        # print("X VALUES: ", x_val)
        # print("Y VALUES: ", y_val)
        # print("COLOR: ", color)
        plt.scatter(x_val, y_val, marker='o', color=color, label=img_name)
        plt.legend()
    plt.show()


def draw_image_with_bounding_boxes(image, bounding_boxes, resuts):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # print(resuts)

    for idx, bound_box in enumerate(bounding_boxes):
        if resuts[idx] == 0:
            col = 'r'
        elif resuts[idx] == 1:
            col = 'b'
        elif resuts[idx] == 2:
            col = 'g'
        else:
            col = 'm'
        # print(bound_box, results[idx])
        rect = patches.Rectangle(bound_box, PATCH_SIZE, PATCH_SIZE, linewidth=1, edgecolor=col, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    ax.imshow(image, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.show()


def calculate_accuracy(bounding_boxes, categories):
    samples = len(bounding_boxes)
    correct_samples = 0
    index_samples = [0, 0, 0, 0]
    correct_texture_samples = [0, 0, 0, 0]
    for idx, bound_box in enumerate(bounding_boxes):
        if bound_box[0] < 400 and bound_box[1] < 400:
            index_samples[3] += 1
            if categories[idx] == 3:
                correct_samples += 1
                correct_texture_samples[3] += 1
        elif bound_box[0] < 400 <= bound_box[1]:
            index_samples[2] += 1
            if categories[idx] == 2:
                correct_samples += 1
                correct_texture_samples[2] += 1
        elif bound_box[1] < 400 <= bound_box[0]:
            index_samples[1] += 1
            if categories[idx] == 1:
                correct_samples += 1
                correct_texture_samples[1] += 1
        elif bound_box[1] >= 400 and bound_box[0]:
            index_samples[0] += 1
            if categories[idx] == 0:
                correct_samples += 1
                correct_texture_samples[0] += 1
    result = correct_samples / samples
    all_texture_results = [x / y for x, y in zip(correct_texture_samples, index_samples)]
    all_texture_results.append(result)
    return all_texture_results


def get_sum_of_distances_to_centroid(centroid_point, points):
    distance = 0
    for point in points:
        distance += calculate_distance(centroid_point, point)
    return distance


def calculate_distance(point_1, point_2):
    dist = math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)
    return dist


def centroid(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    _len = len(points)
    centroid_x = sum(x_coords)/_len
    centroid_y = sum(y_coords)/_len
    return [centroid_x, centroid_y]


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

    # print(type(patch_locations))
    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in img_patches:
        glcm = greycomatrix(patch, distances=OFFSETS, angles=ANGLES, levels=256,
                            symmetric=True, normed=True)
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])
    image_features[img_name] = (xs, ys)
    # print("calculated points for: " + img_name)
    # print(xs)
    # print(ys)
    values = [[xs[i], ys[i]] for i in range(len(xs))]
    calculated_centroid = centroid(values)
    lerned_centroids.append(calculated_centroid)
    # print("centroid is equeal: ", calculated_centroid)
    concentration = get_sum_of_distances_to_centroid(calculated_centroid, values)
    concentrations.append(concentration)
    # print("distances is equal: ", concentration)

    # # create the figure
    # fig = plt.figure(figsize=(8, 8))
    #
    # # display original image with locations of patches
    # ax = fig.add_subplot(3, 2, 1)
    # ax.imshow(image, cmap=plt.cm.gray,
    #           vmin=0, vmax=255)
    # for (y, x) in patch_locations:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs')
    # ax.set_xlabel('Original Image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.axis('image')
    #
    # # for each patch, plot (dissimilarity, correlation)
    # ax = fig.add_subplot(3, 2, 2)
    # ax.plot(xs, ys, 'go',
    #         label='Patches')
    # ax.set_xlabel('GLCM Dissimilarity')
    # ax.set_ylabel('GLCM Correlation')
    # ax.legend()
    #
    # # display the image patches
    # for i, patch in enumerate(img_patches):
    #     ax = fig.add_subplot(3, len(img_patches), len(img_patches)*1 + i + 1)
    #     ax.imshow(patch, cmap=plt.cm.gray,
    #               vmin=0, vmax=255)
    #     ax.set_xlabel('Patch %d' % (i + 1))
    #
    # # display the patches and plot
    # fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
    # plt.tight_layout()
    # if img_idx != len(img_names) -1:
    #     plt.show(block=False)
    # else:
    #     plt.show()
    # plt.show(block=False)
# print(image_features)
# draw_all_data_graph(image_features, img_names)

# ------------------ TESTING THE CLASSIFICATOR -------------------


def get_testing_image_patches_locations(size, image_array, img_nam):
    locs = []
    width_probes = size[0] / PATCH_SIZE
    higth_probes = size[1] / PATCH_SIZE

    for i in range(int(higth_probes)):
        row = i * PATCH_SIZE
        for j in range(int(width_probes)):
            column = j * PATCH_SIZE
            locs.append([row, column])
            if higth_probes % PATCH_SIZE != 0 and i == int(higth_probes) - 1:
                locs.append([size[1] - PATCH_SIZE, column])
        if width_probes % PATCH_SIZE != 0:
            locs.append([row, size[0] - PATCH_SIZE])
    # print(img_nam, size, width_probes, higth_probes, locs)
    return locs


def get_idx_of_closest_center(centroid, location):
    min = 10000000000
    centr_idx = 0
    for idx, center in enumerate(lerned_centroids):

        dist = calculate_distance(center, centroid)
        if dist < min:
            min = dist
            centr_idx = idx
    # print(centroid, centr_idx, location)
    return centr_idx


for img_idx, img_name in enumerate(img_testing_names):
    if img_idx != 0:
        continue
    # loading images
    img = Image.open('img_testing/' + img_name).convert('L')
    WIDTH, HEIGHT = img.size
    image_data = list(img.getdata())
    image = [image_data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
    # convert to numpy array
    image = np.array([np.array(xi) for xi in image])

    testing_patch_locations = get_testing_image_patches_locations((WIDTH, HEIGHT), image, img_name)

    img_patches = []
    for loc in testing_patch_locations:
        img_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])
    results = []
    for idx, patch in enumerate(img_patches):
        patch_dic = {}
        glcm = greycomatrix(patch, distances=OFFSETS, angles=ANGLES, levels=256,
                            symmetric=True, normed=True)
        center = [greycoprops(glcm, 'dissimilarity')[0, 0], greycoprops(glcm, 'correlation')[0, 0]]
        closest_center = get_idx_of_closest_center(center, testing_patch_locations[idx])
        results.append(closest_center)
    accuracies = (calculate_accuracy(testing_patch_locations, results))
    draw_image_with_bounding_boxes(image, testing_patch_locations, results)


for img_idx, img_name in enumerate(img_names):
    with open('experience.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([img_name, NUMBER_OF_SAMPLES, PATCH_SIZE, OFFSET_STRING, ANGLES_STRING, concentrations[img_idx],
                         accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4]])
