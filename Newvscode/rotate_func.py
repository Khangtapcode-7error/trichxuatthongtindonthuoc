import cv2
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

def getSkewAngle(cvImage) -> float:
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    
    angle = minAreaRect[-1]

    # Điều chỉnh góc về khoảng -45 đến 45 độ
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    return -1.0 * angle

def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage, angle_threshold=1.0):
    angle = getSkewAngle(cvImage)
    print(f"Detected skew angle: {angle}")
    
    if abs(angle) < angle_threshold:  # Nếu góc lệch nhỏ hơn ngưỡng, không xoay
        print(f"Angle {angle} is below threshold, no rotation applied.")
        return cvImage
    return rotateImage(cvImage, -1.0 * angle)
