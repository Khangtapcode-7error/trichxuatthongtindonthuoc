import cv2
import pytesseract
import rotate_func1
import os
# read image
img_path =r'bachmai_drug\Mau3 - Crop.jpg'
img = cv2.imread(img_path)
cv2.imshow('img', img)

#rotate
fixed = rotate_func1.deskew_image(img, angle_threshold=1)
cv2.imshow('fixed', fixed)

# RGB to Grayn
gray = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)

# Resize image
high_res_image = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#threshold
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)
cv2.imshow('adaptive_thresh', adaptive_thresh)
# Lưu ảnh adaptive_thresh vào file
cv2.imwrite('adaptive_threshold_output.jpg', adaptive_thresh)
# OCR
# link to Tesseract 
pytesseract.pytesseract.tesseract_cmd = r'C:\VS_CODE\Newvscode\tesseract\tesseract.exe'

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(adaptive_thresh,lang="vie+eng", config=custom_config)
print(text)

# create text and write
with open(r"test.txt", "a", encoding="utf-8") as f:
    f.writelines(text)
cv2.waitKey(0)
cv2.destroyAllWindows()





#  # image to data
# boxes = pytesseract.image_to_data(gray, lang="vie")
# print(boxes)

# for x, b in enumerate(boxes.splitlines()):
#     if x != 0:
#         b = b.split()
#         if len(b) == 12:
#             print(b)
#             x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
#             cv.rectangle(gray, (x, y), (x + w, h + y), (0, 0, 255), 2)
#             cv.putText(gray, b[11], (x, y), cv.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 1)
#cv.imshow('bound', gray)

# adaptive threshold
#adaptive_thresh= cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 101, 31)

# # cv.imshow('Gray', gray)
#print(pytesseract.get_tesseract_version())
# # Resize ảnh
# resized_image = cv.resize(img, (1024, 1024), interpolation=cv.INTER_LINEAR_EXACT)
# # Resize ảnh để tăng độ phân giải
# high_res_image = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)