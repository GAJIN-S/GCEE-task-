import cv2


def create_kernel(size=3):
    return cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )


def erosion(image, kernel):
    return cv2.erode(image, kernel)


def dilation(image, kernel):
    return cv2.dilate(image, kernel)


def opening(image, kernel):
    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )


def closing(image, kernel):
    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel
    )