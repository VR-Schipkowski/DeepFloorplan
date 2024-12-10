from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='demo/Floorplan_1stFloor.png',
                    help='input image paths.')


def preprocessing(image, noise_removal_threshold=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Filter using contour area and remove small noise
    cnts = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10000:
            cv2.drawContours(binary, [c], -1, (0, 0, 0), -1)
    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = 255 - cv2.morphologyEx(binary,
                                   cv2.MORPH_CLOSE, kernel, iterations=2)

    return image, close


def makeImageBetter(img, noise_removal_threshold=200):
    img, processed_image = preprocessing(img, noise_removal_threshold)
    # Convert the processed image to RGB
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    # Ensure the processed image has the same type and format as the input image
    processed_image_rgb = processed_image_rgb.astype(np.uint8)

    return processed_image_rgb


def main(args):
    image = imageio.imread(args.im_path)
    image_better = makeImageBetter(image)

    plt.figure(figsize=(12, 6))

    # Plot original image and its histogram
    plt.subplot(221)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(222)
    plt.title('Histogram of Original Image')
    plt.hist(image.ravel(), bins=256, color='black', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    # Plot enhanced image and its histogram
    plt.subplot(223)
    plt.title('Enhanced Image')
    plt.imshow(image_better)

    plt.subplot(224)
    plt.title('Histogram of Enhanced Image')
    plt.hist(image_better.ravel(), bins=256, color='black', alpha=0.7)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
