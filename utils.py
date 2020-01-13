from os import walk


def get_images_names_from_dir(dir_name):
    f = []
    for (dirpath, dirnames, filenames) in walk(dir_name):
        f.extend(filenames)
        break
    return f


