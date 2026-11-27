import cv2

"""
from filtering.filters import (
    average_filter,
    gaussian_filter,
    median_filter,
    bilateral_filter
)


image = cv2.imread("images/spidy.webp")

average = average_filter(image)
gaussian = gaussian_filter(image)
median = median_filter(image)
bilateral = bilateral_filter(image)

cv2.imshow("Original", image)
cv2.imshow("Average", average)
cv2.imshow("Gaussian", gaussian)
cv2.imshow("Median", median)
cv2.imshow("Bilateral", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()


from edges.edge_detection import sobel_x, sobel_y


image = cv2.imread("images/spidy.webp")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


x_edges = sobel_x(gray)
y_edges = sobel_y(gray)


cv2.imshow("Original", image)
cv2.imshow("Sobel X", x_edges)
cv2.imshow("Sobel Y", y_edges)

cv2.waitKey(0)
cv2.destroyAllWindows()



image = cv2.imread("images/spidy.webp")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(laplacian)

cv2.imshow("Original", image)
cv2.imshow("Laplacian", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()




from edges.edge_detection import canny


image = cv2.imread("images/spidy.webp")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edges = canny(blur)

cv2.imshow("Original", image)
cv2.imshow("Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()




from thresholding.threshold import (
    global_threshold,
    otsu_threshold,
    adaptive_threshold
)


image = cv2.imread("images/spidy.webp")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

global_result = global_threshold(gray)
otsu_result = otsu_threshold(gray)
adaptive_result = adaptive_threshold(gray)

cv2.imshow("Original", image)
cv2.imshow("Global", global_result)
cv2.imshow("Otsu", otsu_result)
cv2.imshow("Adaptive", adaptive_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
from morphology.morphological import (
    create_kernel,
    erosion,
    dilation,
    opening,
    closing
)

image = cv2.imread("images/spidy.webp")

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = create_kernel(3)

eroded = erosion(binary, kernel)
# dilated = dilation(binary, kernel)
# opened = opening(binary, kernel)
# closed = closing(binary, kernel)

cv2.imshow("Original", binary)
cv2.imshow("Original1", image)
cv2.imshow("Erosion", eroded)
# cv2.imshow("Dilation", dilated)
# cv2.imshow("Opening", opened)
# cv2.imshow("Closing", closed)

cv2.waitKey(0)
cv2.destroyAllWindows()
