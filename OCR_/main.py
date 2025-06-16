import cv2
import pytesseract
import rotate_func1
import os
import re //new
# read image
img_path =r'bachmai_drug\Mau3 - Crop.jpg'
img = cv2.imread(img_path)


#rotate
fixed = rotate_func1.deskew_image(img, angle_threshold=1)


# RGB to Grayn
gray = cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY)

# Resize image
high_res_image = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#threshold
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 21)

# Lưu ảnh adaptive_thresh vào file
# cv2.imwrite('adaptive_threshold_output.jpg', adaptive_thresh)
# OCR
# link to Tesseract 
pytesseract.pytesseract.tesseract_cmd = r'tesseract.exe'

custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(adaptive_thresh,lang="vie+eng", config=custom_config)
#print(text) // #new
filtered_text = re.sub(
    r"[^a-zA-Z0-9"
    r"àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệ"
    r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    r"ùúủũụưứừửữựỳýỷỹỵ"
    r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆ"
    r"ÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    r"ÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
    r"\s,;()]",  # giữ các ký tự đặc biệt và khoảng trắng
    "", 
    text
)

# create text and write
with open(r"text.txt", "a", encoding="utf-8") as f:
    f.writelines(filtered_text)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
