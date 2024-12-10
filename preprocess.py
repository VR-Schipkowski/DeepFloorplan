from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from matplotlib import pyplot as plt
import imageio.v2 as imageio
import numpy as np


def makeImageBetter(image):
    # Convert to PIL Image
    im_pil = Image.fromarray(image)

    # Convert to grayscale
    im_gray = im_pil.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(im_gray)
    im_contrast = enhancer.enhance(2.0)  # Increase contrast by a factor of 2

    # Reduce noise
    im_denoised = im_contrast.filter(ImageFilter.MedianFilter(size=3))

    # Edge detection
    im_edges = im_denoised.filter(ImageFilter.FIND_EDGES)

   # Apply threshold to create a binary image
    threshold = 125
    im_binary = im_edges.point(lambda p: p > threshold and 255)

    # Invert the binary image
    im_binary = ImageOps.invert(im_binary)

    # Use morphological operations to enhance features
    im_morph = im_edges.filter(ImageFilter.MaxFilter(3))

    im_morph = im_morph.filter(ImageFilter.MinFilter(3))
    im_morph = ImageOps.invert(image=im_morph)
    # Convert back to NumPy array
    im_better = np.array(im_morph)

    # Ensure the image has 3 channels
    if im_better.ndim == 2:
        im_better = np.stack((im_better,) * 3, axis=-1)
    elif im_better.shape[2] == 1:
        im_better = np.concatenate([im_better] * 3, axis=-1)

    return im_better/255.0


image_path = "demo/Floorplan_1stFloor2.png"
image = imageio.imread(image_path)

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
