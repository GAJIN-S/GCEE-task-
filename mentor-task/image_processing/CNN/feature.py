import cv2
import matplotlib.pyplot as plt

# 1. Load image
img = cv2.imread(r"C:\Users\gokul\OneDrive\Desktop\GAJIN-S\IP-AIML\image_processing\CNN\img\cnn.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Create ORB detector
orb = cv2.ORB_create(nfeatures=500)

# 3. Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# 4. Draw keypoints
result = cv2.drawKeypoints(
    img, keypoints, None,
    color=(0, 255, 0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# 5. Display results
print("Number of keypoints:", len(keypoints))
print("Descriptor shape:",
      descriptors.shape if descriptors is not None else None)

plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("ORB Feature Keypoints")
plt.axis("off")
plt.show()