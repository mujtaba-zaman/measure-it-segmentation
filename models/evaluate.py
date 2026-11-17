"""Evaluate one-object-per-image instance masks on the untouched test split."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import ROOT, ObjectDataset, collate, maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

def ap_at(records, threshold):
    ranked=sorted(records,key=lambda x:x[0],reverse=True); tp=[]; fp=[]
    for score,iou in ranked: tp.append(float(iou>=threshold)); fp.append(float(iou<threshold))
    if not tp:return 0.0
    tp=np.cumsum(tp); fp=np.cumsum(fp); recall=tp/len(records); precision=tp/np.maximum(tp+fp,1)
    return float(np.mean([precision[recall>=r].max() if np.any(recall>=r) else 0 for r in np.linspace(0,1,101)]))

def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',default='models/maskrcnn.pt'); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=ap.parse_args()
    ds=ObjectDataset(ROOT/'dataset/splits/test.txt')
    if not len(ds): raise SystemExit('Test split is empty')
    loader=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate)
    model=maskrcnn_resnet50_fpn(weights=None)
    model.roi_heads.box_predictor.cls_score=torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features,2)
    model.roi_heads.box_predictor.bbox_pred=torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features,8)
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    old=model.roi_heads.mask_predictor; model.roi_heads.mask_predictor=MaskRCNNPredictor(old.conv5_mask.in_channels,256,2)
    state=torch.load(ROOT/a.weights,map_location='cpu',weights_only=False); model.load_state_dict(state['model']); model.to(a.device).eval()
    records=[]; ious=[]; tp=fp=fn=0
    with torch.no_grad():
      for imgs,targets in loader:
        out=model([imgs[0].to(a.device)])[0]; gt=targets[0]['masks'][0].numpy().astype(bool)
        if len(out['scores']):
          j=int(torch.argmax(out['scores'])); score=float(out['scores'][j]); pm=out['masks'][j,0].cpu().numpy()>=.5
        else: score=0.0; pm=np.zeros_like(gt)
        inter=int((pm&gt).sum()); union=int((pm|gt).sum()); iou=inter/max(union,1); records.append((score,iou)); ious.append(iou)
        if score>=.5 and iou>=.5: tp+=1
        elif score>=.5: fp+=1
        else: fn+=1
    thresholds=np.arange(.5,.951,.05); result={'images':len(records),'mask_iou_mean':float(np.mean(ious)),'mAP_50':ap_at(records,.5),'mAP_50_95':float(np.mean([ap_at(records,t) for t in thresholds])),'precision_at_score_0_5_iou_0_5':tp/max(tp+fp,1),'recall_at_score_0_5_iou_0_5':tp/max(tp+fn,1),'score_threshold':.5,'note':'Single-object-per-image AP; fixed test split only.'}
    path=ROOT/'measurement/test_metrics.json'; path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
