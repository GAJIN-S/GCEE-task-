import cv2
"""SOBEL EDGE DETECTION  """

def sobel_x(image):
    result = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    return cv2.convertScaleAbs(result)

def sobel_y(image):
    result = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    return cv2.convertScaleAbs(result)

"""LAPLACIAN ED"""

def laplacian(image):
    result = cv2.Laplacian(
        image,
        cv2.CV_64F
    )

    return cv2.convertScaleAbs(result)

'CANNY'

def canny(image, lower_threshold=100, upper_threshold=100):
    return cv2.Canny(
        image,
        lower_threshold,
        upper_threshold
    )