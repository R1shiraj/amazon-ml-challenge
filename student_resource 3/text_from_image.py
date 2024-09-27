import pytesseract
import cv2
from PIL import Image

# Make sure to update the correct path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

ans = extract_text(r'C:\Users\Rishiraj\OneDrive\Desktop\Amazon ML Challenge\student_resource 3\downloaded_images\width\61Drr5Mq3nL.jpg')
print(ans)