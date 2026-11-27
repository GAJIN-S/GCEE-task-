import cv2


def read_image(path):
    return cv2.imread(path)


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(path, image):
    cv2.imwrite(path, image)