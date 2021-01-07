#pdf 2 img
from wand.image import Image as wi
import pytesseract
from PIL import Image
import argparse

pdf = wi(filename="pdf.pdf", resolution=300)
pdfImage = pdf.convert("jpeg")
i=1
for img in pdfImage.sequence:
    page = wi(image=img)
    page.save(filename=str(i)+".jpeg")
    i +=1
