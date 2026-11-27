import json, os, sys
for tag in sys.argv[1:]:
    f=f'logs/fastflow_sweep/{tag}.json'
    if not os.path.exists(f): continue
    print('==', tag)
    for e in json.load(open(f))['history']:
        v=[r for r in e['variants'] if r['map']=='z2' and r['scales']==[0,1,2] and r['sigma']==16.0 and r['fg_dilate']==-24 and r['score']=='max']
        b=max(e['variants'],key=lambda r:r['img_auc'])
        s=f"  {e['step']:4d}"
        if v: s+=f" cfg auc {v[0]['img_auc']:.4f} iAP {v[0]['img_ap']:.3f} pAP128 {v[0]['pxl_ap128']:.3f}"
        s+=f" | best {b['map']} {b['scales']} s{b['sigma']} fg{b['fg_dilate']} auc {b['img_auc']:.4f} iAP {b['img_ap']:.3f}"
        print(s)
