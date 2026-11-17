"""Create exact-size printable SVGs: 9x7 inner-corner board and 50 mm ArUco ID 0."""
from pathlib import Path
import argparse
import cv2

ap=argparse.ArgumentParser(); ap.add_argument('--square-mm',type=float,default=20); ap.add_argument('--marker-mm',type=float,default=50); ap.add_argument('--output',default='calibration/printables'); a=ap.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
cols,rows,margin=10,8,10
board_w,board_h=cols*a.square_mm,rows*a.square_mm
s=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{board_w+2*margin}mm" height="{board_h+2*margin}mm" viewBox="0 0 {board_w+2*margin} {board_h+2*margin}">','<rect width="100%" height="100%" fill="white"/>']
for y in range(rows):
 for x in range(cols):
  if (x+y)%2==0:s.append(f'<rect x="{margin+x*a.square_mm}" y="{margin+y*a.square_mm}" width="{a.square_mm}" height="{a.square_mm}" fill="black"/>')
s.append('</svg>'); (out/'checkerboard_9x7_inner_20mm.svg').write_text('\n'.join(s),encoding='utf-8')
dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
bitmap=cv2.aruco.generateImageMarker(dictionary,0,600,borderBits=1)
rows_px,cols_px=bitmap.shape
pixels=' '.join('1' if bitmap[y,x]<128 else '0' for y in range(rows_px) for x in range(cols_px))
size=a.marker_mm; pad=5; page=size+2*pad
svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{page}mm" height="{page}mm" viewBox="0 0 {page} {page}"><rect width="100%" height="100%" fill="white"/><image x="{pad}" y="{pad}" width="{size}" height="{size}" href="data:image/png;base64,{__import__('base64').b64encode(cv2.imencode('.png',bitmap)[1]).decode()}"/><text x="{pad}" y="{page-1}" font-size="2">ID 0 | DICT_4X4_50 | marker edge {size} mm</text></svg>'''
(out/'aruco_dict4x4_50_id0_50mm.svg').write_text(svg,encoding='utf-8')
print(f'Created printable SVGs in {out.resolve()}')
