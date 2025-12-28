import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# ------------------- CONSTANTS -------------------
NUM_OF_FEATURES = 7        # Total number of features to check on the currency note
NUM_OF_TEMPLATES = 6       # Number of template images for each feature

# Minimum SSIM score required to consider a feature verified
min_ssim_score_list = [0.4,0.4,0.5,0.4,0.5,0.45,0.5]

# Predefined search areas for each feature [x_start, x_end, y_start, y_end]
search_area_list = [
    [200,300,200,370],
    [1050,1500,300,450],
    [100,450,20,120],
    [690,1050,20,120],
    [820,1050,350,430],
    [700,810,330,430],
    [400,650,0,100]
]

# Acceptable feature area range for each feature [min_area, max_area]
feature_area_limits_list = [
    [12000,17000],
    [10000,18000],
    [20000,30000],
    [24000,36000],
    [15000,25000],
    [7000,13000],
    [11000,18000]
]

# ------------------- HELPER FUNCTIONS -------------------

def calculateSSIM(a, b):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Ensures both images are resized to the same dimensions before comparison.
    Returns a float value between 0 and 1 (higher means more similar).
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w,h))
    b = cv2.resize(b, (w,h))
    return ssim(a, b)

def computeORB(template, query):
    """
    Detect ORB keypoints and descriptors in both template and query images,
    match them using Hamming distance, and compute the homography (perspective transform).
    Returns transformed coordinates of the template on the query image or None if matching fails.
    """
    orb = cv2.ORB_create(700)  # Create ORB detector with max 700 keypoints

    # Detect keypoints and descriptors
    k1, d1 = orb.detectAndCompute(template, None)
    k2, d2 = orb.detectAndCompute(query, None)

    # If no descriptors found, return None
    if d1 is None or d2 is None:
        return None

    # Match descriptors using Brute-Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = bf.match(d1, d2)

    # Need at least 4 matches to compute homography
    if len(matches) < 4:
        return None

    # Extract coordinates of matched keypoints
    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    # Compute homography matrix using RANSAC
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    if M is None:
        return None

    # Get template corners and transform to query image
    h, w = template.shape
    box = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    return cv2.perspectiveTransform(box, M)

# ------------------- MAIN VERIFICATION FUNCTION -------------------

def verify_currency(image_path):
    """
    Verify whether a currency note is REAL or FAKE based on multiple features.
    Returns a string like "X/10 FEATURES VERIFIED → REAL/FAKE".
    """
    # Check if image exists
    if not os.path.exists(image_path):
        return "Image not found"

    # Load image, resize to standard dimensions
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1167, 519))

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    verified = 0  # Counter for verified features

    # Loop through each feature
    for f in range(NUM_OF_FEATURES):
        scores = []  # Store SSIM scores for all templates of this feature

        # Loop through each template of the feature
        for t in range(NUM_OF_TEMPLATES):
            path = f"Dataset/500_Features Dataset/Feature {f+1}/{t+1}.jpg"
            if not os.path.exists(path):
                continue

            temp = cv2.imread(path, 0)  # Load template in grayscale

            # Apply search area mask to focus on specific region
            mask = gray.copy()
            sa = search_area_list[f]
            mask[:, :sa[0]] = 0          # Mask left
            mask[:, sa[1]:] = 0          # Mask right
            mask[:sa[2], :] = 0          # Mask top
            mask[sa[3]:, :] = 0          # Mask bottom

            # Detect and locate feature using ORB
            dst = computeORB(temp, mask)
            if dst is None:
                continue

            # Calculate bounding box of detected feature
            x, y, w, h = cv2.boundingRect(dst)
            area = w * h
            mn, mx = feature_area_limits_list[f]

            # Skip if detected area is not within expected range
            if not (mn <= area <= mx):
                continue

            # Crop the detected feature and calculate SSIM
            crop = blur[y:y+h, x:x+w]
            score = calculateSSIM(temp, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            scores.append(score)

        # If any template matches with SSIM above threshold, consider feature verified
        if scores and max(scores) >= min_ssim_score_list[f]:
            verified += 1

    # Final verdict: at least 8 features verified → REAL, otherwise FAKE
    verdict = "REAL" if verified >= 8 else "FAKE"
    return f"{verified}/10 FEATURES VERIFIED → {verdict}"
