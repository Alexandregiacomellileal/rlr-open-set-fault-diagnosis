#!/usr/bin/env python3
from pathlib import Path
import argparse, zipfile, tempfile, shutil, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

FEATURES=[
    "voltage_V","current_A","power_W","frequency_Hz","power_factor",
    "PE_m3","PE_m4","PE_m5","WPE","MSPE","CV","delta_PF","SII"
]
STATION_FILES={
    "S1":"station1_normal_features.csv",
    "S2":"station2_medium_features.csv",
    "S3":"station3_high_features.csv",
}
SPARSE_TYPES={"inrush_surge","spike_transient","overload"}
SEED=42
MIN_KNOWN_ACCEPT=0.85
MODES=["CONTAMINATED","CLEAN_ORACLE","FILTERED"]

def robust_stats(ref):
    center={}; scale={}
    for f in FEATURES:
        x=pd.to_numeric(ref[f],errors="coerce").dropna().to_numpy()
        if len(x)==0:
            center[f],scale[f]=0.0,1.0
            continue
        c=float(np.median(x))
        q1,q3=np.quantile(x,[.25,.75])
        s=float(q3-q1)
        if not np.isfinite(s) or s<1e-12:
            mad=float(np.median(np.abs(x-c)))
            s=1.4826*mad
        if not np.isfinite(s) or s<1e-12:
            s=float(np.nanstd(x))
        if not np.isfinite(s) or s<1e-12:
            s=1.0
        center[f],scale[f]=c,s
    return center,scale

def apply_norm(df,c,s):
    out=df.copy()
    for f in FEATURES:
        out[f]=(pd.to_numeric(out[f],errors="coerce")-c[f])/s[f]
    return out

def filter_reference_unsupervised(ref):
    c,s=robust_stats(ref)
    Z=[]
    for f in FEATURES:
        z=(pd.to_numeric(ref[f],errors="coerce")-c[f])/s[f]
        Z.append(z.to_numpy(float))
    Z=np.column_stack(Z)
    score=np.nanmax(np.abs(Z),axis=1)
    keep=np.isfinite(score) & (score<=6.0)
    filtered=ref.loc[keep].copy()
    if len(filtered)<0.8*len(ref):
        keep=np.isfinite(score) & (score<=8.0)
        filtered=ref.loc[keep].copy()
    return filtered,score,keep

def load_station_frames(zip_path):
    tmp=Path(tempfile.mkdtemp(prefix="step15_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            for fn in STATION_FILES.values():
                z.extract(f"controlled-fault-benchmark-v1.0/{fn}",tmp)
        base=tmp/"controlled-fault-benchmark-v1.0"
        need=FEATURES+["anomaly","anomaly_type","event_id","segment_id"]
        return {st:pd.read_csv(base/fn,usecols=need) for st,fn in STATION_FILES.items()}
    finally:
        shutil.rmtree(tmp,ignore_errors=True)

def build_events(station_frames,mode):
    frames=[]; audit=[]
    for st,df in station_frames.items():
        first=sorted(df["segment_id"].dropna().unique())[0]
        ref_all=df[df["segment_id"].eq(first)].copy()
        evaldf=df[~df["segment_id"].eq(first)].copy()

        if mode=="CONTAMINATED":
            ref=ref_all.copy()
        elif mode=="CLEAN_ORACLE":
            ref=ref_all[ref_all["anomaly"].eq(0)].copy()
        elif mode=="FILTERED":
            ref,score,keep=filter_reference_unsupervised(ref_all)
        else:
            raise ValueError(mode)

        c,s=robust_stats(ref)
        tr=apply_norm(evaldf,c,s)
        tr=tr[tr["anomaly"].eq(1)].copy()
        tr["station"]=st
        tr["event_uid"]=st+"::"+tr["event_id"].astype(str)
        ev=tr.groupby(["station","event_uid","anomaly_type"],as_index=False)[FEATURES].median()
        frames.append(ev)

        audit.append({
            "station":st,
            "mode":mode,
            "original_reference_rows":len(ref_all),
            "used_reference_rows":len(ref),
            "original_reference_anomaly_rows":int(ref_all["anomaly"].eq(1).sum()),
            "used_reference_anomaly_rows":int(ref["anomaly"].eq(1).sum()),
            "removed_rows":len(ref_all)-len(ref),
            "removed_anomaly_rows":int(ref_all["anomaly"].eq(1).sum())-int(ref["anomaly"].eq(1).sum()),
            "overlap_rows":0
        })

    data=pd.concat(frames,ignore_index=True)
    types=sorted(data["anomaly_type"].dropna().unique())
    main13=sorted([c for c in types if c not in SPARSE_TYPES])
    data=data[data["anomaly_type"].isin(main13)].reset_index(drop=True)
    return data,main13,pd.DataFrame(audit)

def impute(train,test):
    med=train[FEATURES].median(numeric_only=True)
    tr=train.copy(); te=test.copy()
    tr[FEATURES]=tr[FEATURES].fillna(med)
    te[FEATURES]=te[FEATURES].fillna(med)
    return tr,te

def fit_rf(train,trees):
    rf=RandomForestClassifier(
        n_estimators=trees,class_weight="balanced_subsample",
        random_state=SEED,n_jobs=1,max_features="sqrt"
    )
    rf.fit(train[FEATURES],train["anomaly_type"])
    return rf

def entropy_pred(model,test):
    P=model.predict_proba(test[FEATURES])
    eps=1e-12
    ent=-(P*np.log(np.clip(P,eps,1))).sum(axis=1)/np.log(P.shape[1])
    pred=np.asarray(model.classes_,dtype=object)[P.argmax(axis=1)]
    return ent,pred

def make_meta_entropy(outer_train,heldout,trees):
    sts=sorted(outer_train["station"].unique())
    pseudos=sorted([c for c in outer_train["anomaly_type"].unique() if c!=heldout])
    rows=[]
    for src,tgt in [(sts[0],sts[1]),(sts[1],sts[0])]:
        source=outer_train[outer_train["station"].eq(src)]
        target=outer_train[outer_train["station"].eq(tgt)]
        for pseudo in pseudos:
            fit=source[source["anomaly_type"]!=pseudo].copy()
            if fit["anomaly_type"].nunique()<2: continue
            fit,tgt_imp=impute(fit,target)
            rf=fit_rf(fit,trees)
            ent,_=entropy_pred(rf,tgt_imp)
            rows.append(pd.DataFrame({
                "Entropy":ent,
                "meta_y":(target["anomaly_type"].to_numpy()==pseudo).astype(int)
            }))
    return pd.concat(rows,ignore_index=True)

def optimize_threshold(score,y):
    score=np.asarray(score,float); y=np.asarray(y,int)
    best=None
    for tau in np.unique(np.quantile(score,np.linspace(.40,.995,150))):
        ka=float(np.mean(score[y==0]<=tau))
        if ka+1e-12<MIN_KNOWN_ACCEPT: continue
        ur=float(np.mean(score[y==1]>tau))
        h=0 if ka+ur==0 else 2*ka*ur/(ka+ur)
        key=(ur,h,-tau)
        if best is None or key>best[0]:
            best=(key,float(tau))
    if best is None:
        return float(np.quantile(score[y==0],MIN_KNOWN_ACCEPT))
    return best[1]

def eval_open(ytrue,pred,score,tau,heldout,known_classes):
    ytrue=np.asarray(ytrue,dtype=object); pred=np.asarray(pred,dtype=object)
    is_u=ytrue==heldout; is_k=~is_u
    reject=np.asarray(score)>tau
    out=pred.copy(); out[reject]="UNKNOWN"
    target=ytrue.copy(); target[is_u]="UNKNOWN"
    ka=float(np.mean(~reject[is_k]))
    kc=float(np.mean(out[is_k]==target[is_k]))
    ur=float(np.mean(reject[is_u]))
    h=0 if kc+ur==0 else 2*kc*ur/(kc+ur)
    auc=float(roc_auc_score(is_u.astype(int),score))
    mf1=float(f1_score(target,out,labels=list(known_classes)+["UNKNOWN"],average="macro",zero_division=0))
    return ka,kc,ur,h,auc,mf1

def run_one(data,main13,test_station,mode,inner_trees,outer_trees):
    otr_all=data[data["station"]!=test_station]
    ote=data[data["station"]==test_station]
    rows=[]
    for heldout in main13:
        otr=otr_all[otr_all["anomaly_type"]!=heldout].copy()
        otr,ote_imp=impute(otr,ote)
        meta=make_meta_entropy(otr,heldout,inner_trees)
        tau=optimize_threshold(meta["Entropy"],meta["meta_y"])
        rf=fit_rf(otr,outer_trees)
        ent,pred=entropy_pred(rf,ote_imp)
        vals=eval_open(ote["anomaly_type"],pred,ent,tau,heldout,sorted(otr["anomaly_type"].unique()))
        rows.append({
            "test_station":test_station,
            "heldout_type":heldout,
            "mode":mode,
            "known_acceptance":vals[0],
            "known_correct_rate":vals[1],
            "unknown_recall":vals[2],
            "h_score":vals[3],
            "unknown_auroc":vals[4],
            "open_macro_f1":vals[5],
            "tau":tau
        })
    return pd.DataFrame(rows)

def main(zip_path,out,test_station,inner_trees,outer_trees):
    station_frames=load_station_frames(zip_path)
    results=[]; audits=[]
    for mode in MODES:
        data,main13,audit=build_events(station_frames,mode)
        audits.append(audit)
        results.append(run_one(data,main13,test_station,mode,inner_trees,outer_trees))
        print(test_station,mode,"done")

    rr=pd.concat(results,ignore_index=True)
    rr.to_csv(out/f"contamination_results_{test_station}.csv",index=False)
    pd.concat(audits,ignore_index=True).drop_duplicates().to_csv(
        out/f"contamination_audit_{test_station}.csv",index=False
    )
    print(rr.groupby("mode")[[
        "unknown_auroc","unknown_recall","known_acceptance","known_correct_rate","h_score","open_macro_f1"
    ]].mean().sort_values("h_score",ascending=False).to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--station",choices=["S1","S2","S3"],required=True)
    ap.add_argument("--inner-trees",type=int,default=3)
    ap.add_argument("--outer-trees",type=int,default=12)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    main(a.zip,a.out,a.station,a.inner_trees,a.outer_trees)
