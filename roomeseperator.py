

import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--im_path', type=str, default='demo/45765448.jpg',
                    help='input image paths.')


def preprocessing(image_path):
    image = cv2.imread(image_path)
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
        if area < 50:
            cv2.drawContours(binary, [c], -1, (0, 0, 0), -1)
    # Morph close and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = 255 - cv2.morphologyEx(binary,
                                   cv2.MORPH_CLOSE, kernel, iterations=2)

    return image, close


def removeSmallObjects(image, threshold):

    return image, image


def detect_rooms(image, processed_image):
    # Find contours
    contours, _ = cv2.findContours(
        ~processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rooms = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter out small contours
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            room = image[y:y+h, x:x+w]
            rooms.append(room)

            # Draw bounding box on the original image
            # Generate a random color for the bounding box
            color = np.random.randint(0, 255, size=3).tolist()

            # Draw bounding box on the original image
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            # Draw the contour on the original image
            # cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    return image, rooms


def find_rooms(img, noise_removal_threshold=50, corners_threshold=0.5,
               room_closing_max_length=100, gap_in_wall_threshold=500):
    """

    :param img: grey scale image of rooms, already eroded and doors removed etc.
    :param noise_removal_threshold: Minimal area of blobs to be kept.
    :param corners_threshold: Threshold to allow corners. Higher removes more of the house.
    :param room_closing_max_length: Maximum line length to add to close off open doors.
    :param gap_in_wall_threshold: Minimum number of pixels to identify component as room instead of hole in the wall.
    :return: rooms: list of numpy arrays containing boolean masks for each detected room
             colored_house: A colored version of the input image, where each room has a random color.
    """
    assert 0 <= corners_threshold <= 1
    # Remove noise left from door removal
    cv2.imshow('imut', img)
    cv2.waitKey()

    img[img < 128] = 0
    cv2.imshow('low', img)
    cv2.waitKey()

    img[img > 128] = 255
    cv2.imshow('high', img)
    cv2.waitKey()

    contours, _ = cv2.findContours(
        ~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)

    img = ~mask

    cv2.imshow('removed', img)
    cv2.waitKey()

    # Detect corners (you can play with the parameters here)
    dst = cv2.cornerHarris(img, 2, 3, 0.6)
    dst = cv2.dilate(dst, None)
    corners = dst > corners_threshold * dst.max()
    print(corners)

    # Draw lines to close the rooms off by adding a line between corners on the same x or y coordinate
    # This gets some false positives.
    # You could try to disallow drawing through other existing lines for example.
    for y, row in enumerate(corners):
        print(row)
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):

            if x2[0] - x1[0] < room_closing_max_length:

                color = 0
                cv2.line(img, (x1[0], y), (x2[0], y), color, 1)

    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if y2[0] - y1[0] < room_closing_max_length:
                color = 0
                cv2.line(img, (x, y1[0]), (x, y2[0]), color, 1)

    cv2.imshow('outside', img)
    cv2.waitKey()

    # Find the connected components in the house
    ret, labels = cv2.connectedComponents(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    unique = np.unique(labels)
    rooms = []
    for label in unique:
        component = labels == label
        if img[component].sum() == 0 or np.count_nonzero(component) < gap_in_wall_threshold:
            color = 0
        else:
            rooms.append(component)
            color = np.random.randint(0, 255, size=3)
        img[component] = color

    return img, rooms


def main(args):
    image_path = args.im_path
    image, processed_image = preprocessing(image_path)
    image, rooms = detect_rooms(image, processed_image)

    print(len(rooms))

    # Display the original image with bounding boxes
    cv2.imshow('processed_image', processed_image)
    cv2.waitKey(0)
    cv2.imshow('processed_image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
