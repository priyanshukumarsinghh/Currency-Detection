# ================= IMPORTS =================
import os
import cv2
import numpy as np
import webbrowser
from threading import Timer
from flask import Flask, render_template, request
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

# ================= FLASK APP =================
app = Flask(__name__)

# ================= UPLOAD CONFIG =================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= CONFIG =================
NUM_OF_FEATURES = 7           # Features 1–7 (ORB + SSIM)
NUMBER_OF_TEMPLATES = 6       # Templates per feature

FEATURE_NAMES = [
    "Watermark",
    "Security Thread",
    "Micro Lettering",
    "English inscription",
    "Denomination Numeral",
    "RBI Emblem",
    "Denomination in Hindi",
    "Left Bleed Line",
    "Right Bleed Line",
    "Number Panel"
]

# Search areas for features 1–7
search_area_list = [
    [200,300,200,370],
    [1050,1500,300,450],
    [100,450,20,120],
    [690,1050,20,120],
    [820,1050,350,430],
    [700,810,330,430],
    [400,650,0,100]
]

# Acceptable bounding box areas for features 1–7
feature_area_limits_list = [
    [12000,17000],
    [10000,18000],
    [20000,30000],
    [24000,36000],
    [15000,25000],
    [7000,13000],
    [11000,18000]
]

# Minimum SSIM thresholds for features 1–7
min_ssim_score_list = [0.4,0.4,0.5,0.4,0.5,0.45,0.5]

# ================= UTILITY FUNCTIONS =================
def calculateSSIM(img1, img2):
    """Calculate SSIM between two images after resizing to same dimensions"""
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    return ssim(img1, img2)

def computeORB(template, query):
    """Detect ORB keypoints and compute homography"""
    orb = cv2.ORB_create(700)
    k1, d1 = orb.detectAndCompute(template, None)
    k2, d2 = orb.detectAndCompute(query, None)
    if d1 is None or d2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    if len(matches) < 4:
        return None

    src = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    if M is None:
        return None

    h, w = template.shape
    box = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    return cv2.perspectiveTransform(box, M)

# ================= FEATURES 1–7 =================
def testFeature_1_7(gray, blur, annotated, feature_info):
    """Detect features 1–7 using ORB + SSIM"""
    verified_count = 0
    for j in range(NUM_OF_FEATURES):
        scores, best_score, detected_box = [], -1, None
        for i in range(NUMBER_OF_TEMPLATES):
            path = f"Dataset/500_Features Dataset/Feature {j+1}/{i+1}.jpg"
            if not os.path.exists(path):
                continue
            template = cv2.imread(path, 0)
            mask = gray.copy()
            sa = search_area_list[j]
            mask[:, :sa[0]] = 0
            mask[:, sa[1]:] = 0
            mask[:sa[2], :] = 0
            mask[sa[3]:, :] = 0

            dst = computeORB(template, mask)
            if dst is None:
                continue

            x, y, w, h = cv2.boundingRect(dst)
            area = w * h
            mn, mx = feature_area_limits_list[j]
            if not mn <= area <= mx:
                continue

            crop = blur[y:y+h, x:x+w]
            score = calculateSSIM(template, cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
            scores.append(score)
            if score > best_score:
                best_score, detected_box = score, (x, y, w, h)

        avg = sum(scores)/len(scores) if scores else 0
        passed = avg >= min_ssim_score_list[j] or best_score >= 0.79
        if passed: verified_count += 1

        if detected_box and passed:
            x, y, w, h = detected_box
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(annotated, FEATURE_NAMES[j], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        feature_info.append({
            "name": FEATURE_NAMES[j],
            "status": "Passed" if passed else "Failed"
        })
    return verified_count

# ================= BLEED LINES (FEATURES 8–9) =================
def count_bleed(crop):
    """Count transitions in bleed line"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    cols = t.shape[1]
    res = []
    for j in range(cols):
        col = t[:, j]
        count = sum(col[i] == 255 and col[i+1] == 0 for i in range(len(col)-1))
        if 0 < count < 10:
            res.append(count)
    return (sum(res)/len(res)) if res else -1

def testFeature_8_9(img, feature_info):
    """Detect left and right bleed lines"""
    left = img[120:240, 12:35]
    right = img[120:260, 1135:1155]
    res_l = count_bleed(left)
    res_r = count_bleed(right)

    passed_l = 4.7 <= res_l <= 5.6
    passed_r = 4.7 <= res_r <= 5.6

    feature_info.append({"name": "Left Bleed Line", "status": "Passed" if passed_l else "Failed"})
    feature_info.append({"name": "Right Bleed Line", "status": "Passed" if passed_r else "Failed"})
    return (1 if passed_l else 0) + (1 if passed_r else 0)

# ================= NUMBER PANEL (FEATURE 10) =================
def testFeature_10(gray, feature_info):
    """Verify number panel contours"""
    crop = gray[380:490, 650:1100]
    _, thresh = cv2.threshold(crop, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    passed = len(contours) >= 9
    feature_info.append({"name": "Number Panel", "status": "Passed" if passed else "Failed"})
    return 1 if passed else 0

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process Image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (1167, 519))
            blur = cv2.GaussianBlur(img, (5,5), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            annotated = img.copy()
            
            feature_info = []
            score = 0
            score += testFeature_1_7(gray, blur, annotated, feature_info)
            score += testFeature_8_9(img, feature_info)
            score += testFeature_10(gray, feature_info)

            # Save Annotated Image
            annotated_filename = "annotated_" + filename
            cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], annotated_filename), annotated)

            result = {
                "verdict": "REAL" if score >= 8 else "FAKE",
                "uploaded_image": filename,
                "annotated_image": annotated_filename,
                "feature_info": feature_info
            }
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)

# ================= EXECUTION =================
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == "__main__":
    Timer(1.5, open_browser).start()
    app.run(debug=True, port=5000)