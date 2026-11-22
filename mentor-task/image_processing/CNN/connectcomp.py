import cv2
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread(r"C:\Users\gokul\OneDrive\Desktop\GAJIN-S\IP-AIML\image_processing\CNN\img\cnn.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Convert to binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 3. Connected Component Analysis
count, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary, connectivity=8
)

# 4. Draw bounding boxes and labels
result = img.copy()

for i in range(1, count):  # Skip background label 0
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]

    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(result, str(i), (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print(f"Object {i}: Area={area}, Centroid=({cx:.1f}, {cy:.1f})")

# 5. Display results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(binary, cmap="gray")
plt.title("Binary Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Connected Components")
plt.axis("off")
plt.show()