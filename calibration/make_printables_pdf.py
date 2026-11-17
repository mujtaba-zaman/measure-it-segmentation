"""Generate two exact-size A4 landscape printables for Measure It.
Page 1: 9x7 inner-corner checkerboard (10x8 squares, 20 mm each).
Page 2: OpenCV DICT_4X4_50 marker ID 0, 50 mm outer black square.
Print at 100% / actual size; disable fit-to-page.
"""
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm

OUT = Path(__file__).resolve().parent / "printables" / "measure-it-printables.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
PAGE = landscape(A4)
PAGE_W, PAGE_H = PAGE
c = canvas.Canvas(str(OUT), pagesize=PAGE, pageCompression=1)

# Page 1: 10 x 8 squares means exactly 9 x 7 interior corners.
sq = 20 * mm
cols, rows = 10, 8
board_w, board_h = cols * sq, rows * sq
left = (PAGE_W - board_w) / 2
bottom = (PAGE_H - board_h) / 2 - 2 * mm
c.setFont("Helvetica-Bold", 14)
c.drawCentredString(PAGE_W / 2, PAGE_H - 12 * mm, "Camera calibration checkerboard")
for r in range(rows):
    for col in range(cols):
        if (r + col) % 2 == 0:
            c.setFillColorRGB(0, 0, 0)
            c.rect(left + col * sq, bottom + (rows - 1 - r) * sq, sq, sq, stroke=0, fill=1)
c.setFillColorRGB(0, 0, 0)
c.setFont("Helvetica", 10)
c.drawCentredString(PAGE_W / 2, bottom - 8 * mm, "10 x 8 squares | 9 x 7 inner corners | each square = 20 mm")
c.showPage()

# Page 2: OpenCV DICT_4X4_50, ID 0. The first 4x4 code is 0xB5 0x32.
# OpenCV stores each row-major code MSB first; its 4x4_50 table uses the first 50 codes of 4x4_1000.
code = bytes((0xB5, 0x32))
bits = [((code[i // 8] >> (7 - (i % 8))) & 1) for i in range(16)]
inner = [bits[r * 4:(r + 1) * 4] for r in range(4)]
marker_size = 50 * mm
module = marker_size / 6
x0 = (PAGE_W - marker_size) / 2
y0 = (PAGE_H - marker_size) / 2 + 6 * mm
c.setFont("Helvetica-Bold", 14)
c.drawCentredString(PAGE_W / 2, PAGE_H - 12 * mm, "Reference marker - OpenCV DICT_4X4_50, ID 0")
# Marker has a solid black one-cell border and 4x4 code cells.
for r in range(6):
    for col in range(6):
        black = (r == 0 or r == 5 or col == 0 or col == 5)
        if 1 <= r <= 4 and 1 <= col <= 4:
            black = bool(inner[r - 1][col - 1])
        if black:
            c.setFillColorRGB(0, 0, 0)
            c.rect(x0 + col * module, y0 + (5 - r) * module, module, module, stroke=0, fill=1)
c.setFillColorRGB(0, 0, 0)
c.setFont("Helvetica", 10)
c.drawCentredString(PAGE_W / 2, y0 - 10 * mm, "Measure the outside black edge: 50 mm. Keep the full marker visible in every photo.")
c.setFont("Helvetica", 9)
c.drawCentredString(PAGE_W / 2, 12 * mm, "Print at 100% / actual size - turn off fit-to-page or shrink-to-fit.")
c.save()
print(OUT)
