"""Single-image measurement with calibrated Mask R-CNN and an ArUco scale marker."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from calibration.image_ops import normalize_landscape

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('image'); ap.add_argument('--calibration',default='calibration/intrinsics.json'); ap.add_argument('--weights',default='models/maskrcnn.pt'); ap.add_argument('--marker-mm',type=float,required=True,help='measured ArUco marker side length'); ap.add_argument('--confidence',type=float,default=.75); ap.add_argument('--skip-undistortion',action='store_true',help='use normalized raw pixels for the raw-vs-undistorted comparison'); ap.add_argument('--output',default='measurement/overlay.png'); a=ap.parse_args()
 cal=json.loads(Path(a.calibration).read_text()); raw=cv2.imread(a.image)
 if raw is None: raise SystemExit('ERROR: image not found or unreadable')
 raw=normalize_landscape(raw)
 h,w=raw.shape[:2]
 if [w,h] != cal['image_size_px']: raise SystemExit(f'ERROR: normalized image size {(w,h)} does not match calibrated size {tuple(cal["image_size_px"])}')
 k=np.array(cal['camera_matrix']); d=np.array(cal['distortion_coefficients'])
 if a.skip_undistortion: im=raw
 else:
  nk,_=cv2.getOptimalNewCameraMatrix(k,d,(w,h),1,(w,h)); im=cv2.undistort(raw,k,d,None,nk)
 aruco=cv2.aruco; dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_50); detector=aruco.ArucoDetector(dictionary,aruco.DetectorParameters()); corners,ids,_=detector.detectMarkers(im)
 if ids is None: raise SystemExit('REFUSED: reference ArUco marker not found')
 side_px=float(np.mean([np.linalg.norm(c[0][i]-c[0][(i+1)%4]) for c in corners for i in range(4)])); mm_per_px=a.marker_mm/side_px
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=maskrcnn_resnet50_fpn(weights=None); model.roi_heads.box_predictor.cls_score=torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features,2); model.roi_heads.box_predictor.bbox_pred=torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features,8)
 from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
 old=model.roi_heads.mask_predictor; model.roi_heads.mask_predictor=MaskRCNNPredictor(old.conv5_mask.in_channels,256,2)
 state=torch.load(a.weights,map_location='cpu',weights_only=False); model.load_state_dict(state['model']); model.to(dev).eval()
 rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); tensor=torch.from_numpy(rgb).permute(2,0,1).float()/255
 with torch.no_grad(): pred=model([tensor.to(dev)])[0]
 if not len(pred['scores']): raise SystemExit('REFUSED: object not found')
 idx=int(torch.argmax(pred['scores'])); conf=float(pred['scores'][idx]);
 if conf<a.confidence: raise SystemExit(f'REFUSED: mask confidence {conf:.3f} below threshold {a.confidence:.3f}')
 mask=pred['masks'][idx,0].cpu().numpy()>=.5; ys,xs=np.where(mask)
 if not len(xs): raise SystemExit('REFUSED: empty mask')
 contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if not contours: raise SystemExit('REFUSED: no object contour')
 rect=cv2.minAreaRect(max(contours,key=cv2.contourArea)); side_a,side_b=rect[1]
 width=min(side_a,side_b)*mm_per_px; height=max(side_a,side_b)*mm_per_px
 box=cv2.boxPoints(rect).astype(np.int32)
 overlay=im.copy(); overlay[mask]=(0.45*overlay[mask]+0.55*np.array([0,220,70])).astype(np.uint8); cv2.aruco.drawDetectedMarkers(overlay,corners,ids); cv2.polylines(overlay,[box],True,(0,255,0),2); cv2.putText(overlay,f'{width:.1f} x {height:.1f} mm | conf {conf:.2f}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(out),overlay)
 print(json.dumps({'width_mm':width,'height_mm':height,'confidence':conf,'reference_mm_per_pixel':mm_per_px,'undistortion_applied':not a.skip_undistortion,'overlay':str(out),'warning':'Width is the shorter oriented-box side; height is the longer. Assumes flat object and marker coplanar with low camera tilt.'},indent=2))
if __name__=='__main__':main()
