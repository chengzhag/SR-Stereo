import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    imagedirs = [os.path.join(filepath, d) for d in os.listdir(filepath) if is_image_file(d)]
    imagedirs.sort()
    return imagedirs
