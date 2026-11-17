"""Fine-tune torchvision Mask R-CNN for one foreground object per image.
Expected masks: dataset/labels/<image-stem>.png; foreground pixels are nonzero.
Split text files contain relative paths under dataset/.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

ROOT = Path(__file__).resolve().parent.parent
class ObjectDataset(Dataset):
    def __init__(self, list_file):
        self.items = [x.strip() for x in Path(list_file).read_text().splitlines() if x.strip() and not x.startswith('#')]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = ROOT / 'dataset' / self.items[i]
        image = Image.open(p).convert('RGB')
        mask_path = ROOT / 'dataset' / 'labels' / (p.stem + '.png')
        mask = np.array(Image.open(mask_path).convert('L')) > 0
        ys, xs = np.where(mask)
        if not len(xs): raise ValueError(f"Empty object mask: {mask_path}")
        boxes = torch.tensor([[xs.min(), ys.min(), xs.max()+1, ys.max()+1]], dtype=torch.float32)
        target = {'boxes': boxes, 'labels': torch.ones((1,), dtype=torch.int64),
                  'masks': torch.as_tensor(mask[None], dtype=torch.uint8), 'image_id': torch.tensor([i]),
                  'area': torch.tensor([(xs.max()-xs.min()+1)*(ys.max()-ys.min()+1)], dtype=torch.float32),
                  'iscrowd': torch.zeros((1,), dtype=torch.int64)}
        return F.to_tensor(image), target

def collate(batch): return tuple(zip(*batch))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--batch-size', type=int, default=2); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); ap.add_argument('--output', default='models/maskrcnn.pt'); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train=DataLoader(ObjectDataset(ROOT/'dataset/splits/train.txt'), batch_size=a.batch_size, shuffle=True, collate_fn=collate)
    val=DataLoader(ObjectDataset(ROOT/'dataset/splits/val.txt'), batch_size=1, shuffle=False, collate_fn=collate)
    weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model=maskrcnn_resnet50_fpn(weights=weights)
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(model.roi_heads.box_predictor.bbox_pred.in_features, 8)
    old=model.roi_heads.mask_predictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    model.roi_heads.mask_predictor=MaskRCNNPredictor(old.conv5_mask.in_channels, 256, 2)
    dev=torch.device(a.device); model.to(dev)
    opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.002, momentum=.9, weight_decay=.0005)
    history=[]
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for imgs, targets in train:
            loss=model([im.to(dev) for im in imgs], [{k:v.to(dev) for k,v in t.items()} for t in targets]); total=sum(loss.values())
            opt.zero_grad(); total.backward(); opt.step(); losses.append(float(total.detach().cpu()))
        model.eval(); intersections=unions=0
        with torch.no_grad():
            for imgs, targets in val:
                pred=model([im.to(dev) for im in imgs])[0]; gt=targets[0]['masks'][0].numpy().astype(bool)
                if len(pred['masks']) and float(pred['scores'][0])>=.5:
                    pm=pred['masks'][0,0].cpu().numpy()>=.5; intersections += int((pm & gt).sum()); unions += int((pm | gt).sum())
                else: unions += int(gt.sum())
        row={'epoch':epoch+1,'train_loss_mean':float(np.mean(losses)),'val_pixel_iou':intersections/max(unions,1)}; history.append(row); print(json.dumps(row))
    out=ROOT/a.output; out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'seed':a.seed,'epochs':a.epochs,'architecture':'torchvision Mask R-CNN ResNet50-FPN'},out)
    (out.parent/'training_history.json').write_text(json.dumps(history,indent=2))
    print(f'Saved checkpoint: {out}')
if __name__=='__main__': main()
