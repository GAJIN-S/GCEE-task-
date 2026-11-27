import cv2


def global_threshold(image, threshold=127):
    _, result = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return result
    

def otsu_threshold(image):
    _, result = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return result


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )