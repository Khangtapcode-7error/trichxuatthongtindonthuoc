import cv2
import numpy as np
import pytesseract
import os
import rotate_func1

# Load và deskew ảnh gốc
image = cv2.imread(r"bachmai_drug\Mau3 - Crop.jpg")
if image is None:
    raise FileNotFoundError("Không thể đọc ảnh.")
image = rotate_func1.deskew_image(image, angle_threshold=1)
orig = image.copy()
(H, W) = image.shape[:2]

# Resize ảnh cho EAST
(newW, newH) = (W // 32 * 32, H // 32 * 32)
rW, rH = W / float(newW), H / float(newH)
resized = cv2.resize(image, (newW, newH))

# Tiền xử lý
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 51, 21)

# Load EAST model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH),
                             (123.68, 116.78, 103.94), True, False)
net.setInput(blob)
(scores, geometry) = net.forward([
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
])

# Giải mã output EAST
def decode(scores, geometry, scoreThresh):
    rects, confidences = [], []
    rows, cols = scores.shape[2:4]
    for y in range(rows):
        for x in range(cols):
            score = scores[0, 0, y, x]
            if score < scoreThresh:
                continue
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = geometry[0, 4, y, x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
            w = geometry[0, 1, y, x] + geometry[0, 3, y, x]
            endX = int(offsetX + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
            endY = int(offsetY - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(float(score))
    return rects, confidences

# Lấy boxes
boxes, confidences = decode(scores, geometry, 0.3)
rects_for_nms = [[x, y, x2-x, y2-y] for (x, y, x2, y2) in boxes]
indices = cv2.dnn.NMSBoxes(rects_for_nms, confidences, 0.5, 0.3)

# Tính box theo ảnh gốc
final_boxes = []
if len(indices) > 0:
    for i in indices.flatten():
        (x1, y1, x2, y2) = boxes[i]
        x1 = int(x1 * rW)
        y1 = int(y1 * rH)
        x2 = int(x2 * rW)
        y2 = int(y2 * rH)
        padding = 5
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(W - 1, x2 + padding), min(H - 1, y2 + padding)
        final_boxes.append((x1, y1, x2, y2))

# Sắp xếp box từ trên xuống dưới, trái sang phải
def sort_boxes(boxes, line_thresh=20):
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    grouped = []
    current_line = []
    for box in boxes:
        if not current_line or abs(box[1] - current_line[-1][1]) < line_thresh:
            current_line.append(box)
        else:
            grouped.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [box]
    if current_line:
        grouped.append(sorted(current_line, key=lambda b: b[0]))
    return [b for group in grouped for b in group]

final_boxes = sort_boxes(final_boxes)

# OCR và lưu text theo box có văn bản
os.makedirs("cropped_img", exist_ok=True)
final_boxes_with_text = []

for idx, (x1, y1, x2, y2) in enumerate(final_boxes):
    cropped = orig[y1:y2, x1:x2]
    deskewed = rotate_func1.deskew_image(cropped, angle_threshold=1)
    cv2.imwrite(f"cropped_img/crop_{idx}.jpg", deskewed)

    gray_crop = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(gray_crop, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 51, 21)

    config = r'--oem 3 --psm 6 -l vie+eng'
    text = pytesseract.image_to_string(thres, config=config).strip()
    if text:
        final_boxes_with_text.append(((x1, y1, x2, y2), text))

# Gom nhóm dòng theo vị trí y
grouped_lines = []
current_line = []
prev_y = None
line_thresh = 20

for (box, text) in final_boxes_with_text:
    x1, y1, x2, y2 = box
    if prev_y is None or abs(y1 - prev_y) < line_thresh:
        current_line.append(text)
    else:
        grouped_lines.append(current_line)
        current_line = [text]
    prev_y = y1
if current_line:
    grouped_lines.append(current_line)

# Ghi vào file text
with open("output_text.txt", "w", encoding="utf-8") as f:
    for line in grouped_lines:
        f.write(" ".join(line) + "\n")

# Vẽ bounding boxes chỉ cho box có text
boxed_image = orig.copy()
for (box, _) in final_boxes_with_text:
    x1, y1, x2, y2 = box
    cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite("result_with_boxes.jpg", boxed_image)

print("Đã lưu:")
print("- Ảnh vùng crop: cropped_img/")
print("- Ảnh gốc có box: result_with_boxes.jpg")
print("- Text nhận diện: output_text.txt")
