from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='demo/Floorplan_1stFloor3.png',
                    help='input image paths.')


def makeImageBetter(image):
    # Convert to PIL Image
    im_pil = Image.fromarray(image)

    plt.imshow(im_pil)
    plt.show()

    # Convert to grayscale
    im_gray = im_pil.convert('L')

    plt.imshow(im_gray, cmap='gray')
    plt.show()

    # Increase contrast
    enhancer = ImageEnhance.Contrast(im_gray)
    im_contrast = enhancer.enhance(1.5)

    plt.imshow(im_contrast, cmap='gray')
    plt.show()

    # Reduce noise
    im_denoised = im_contrast.filter(ImageFilter.MedianFilter(size=3))

    # Edge detection
    im_edges = im_denoised.filter(ImageFilter.FIND_EDGES)

   # Apply threshold to create a binary image
    threshold = 125
    im_binary = im_edges.point(lambda p: p > threshold and 255)

    # Invert the binary image
    im_binary = ImageOps.invert(im_binary)

    # Create an alpha mask from the binary image
    im_alpha = im_contrast.convert('L')

    # Convert the edge-detected image to RGB
    im_edges_rgb = im_edges.convert('RGB')

    # Create a transparent image
    im_transparent = Image.new('RGBA', im_pil.size, (0, 0, 0, 0))

    # Blend the edge-detected image with the original image using the alpha mask
    im_overlay = Image.composite(im_edges_rgb, im_transparent, im_alpha)

    # Use morphological operations to enhance features
    im_morph = im_overlay.filter(ImageFilter.MaxFilter(3))

    im_morph = im_morph.filter(ImageFilter.MinFilter(3))
    # Convert back to NumPy array
    im_better = np.array(im_contrast)

    # Ensure the image has 3 channels
    if im_better.ndim == 2:
        im_better = np.stack((im_better,) * 3, axis=-1)
    elif im_better.shape[2] == 1:
        im_better = np.concatenate([im_better] * 3, axis=-1)

    return im_better


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
