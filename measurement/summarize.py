"""Summarize recorded ground-truth comparisons. CSV columns: id,true_width_mm,true_height_mm,pred_width_mm,pred_height_mm,raw_width_mm,raw_height_mm"""
import argparse,csv,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('csv');a=p.parse_args()
rows=list(csv.DictReader(open(a.csv,newline='',encoding='utf-8')))
if len(rows)<10: raise SystemExit(f'Need >=10 measured instances; found {len(rows)}')
out={}
for dim in ('width','height'):
 true=np.array([float(r[f'true_{dim}_mm']) for r in rows]); pred=np.array([float(r[f'pred_{dim}_mm']) for r in rows]); raw=np.array([float(r[f'raw_{dim}_mm']) for r in rows]); err=np.abs(pred-true); rawerr=np.abs(raw-true)
 out[dim]={'mae_mm':float(err.mean()),'median_abs_error_mm':float(np.median(err)),'worst_abs_error_mm':float(err.max()),'mean_absolute_percentage_error':float((err/np.maximum(true,1e-9)).mean()*100),'raw_mae_mm':float(rawerr.mean())}
print(json.dumps({'instances':len(rows),'summary':out},indent=2))
