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
DURATIONS=[1,3,6,12,24]

def robust_reference(ref):
    med,scale={},{}
    for f in FEATURES:
        x=pd.to_numeric(ref[f],errors="coerce").dropna().to_numpy()
        m=float(np.median(x)) if len(x) else 0.0
        if len(x):
            q1,q3=np.quantile(x,[.25,.75])
            s=float(q3-q1)
            if not np.isfinite(s) or s<1e-12:
                s=1.4826*float(np.median(np.abs(x-m)))
            if not np.isfinite(s) or s<1e-12:
                s=float(np.nanstd(x))
        else:
            s=1.0
        if not np.isfinite(s) or s<1e-12:
            s=1.0
        med[f],scale[f]=m,s
    return med,scale

def apply_rlr(df,med,scale):
    out=df.copy()
    for f in FEATURES:
        out[f]=(pd.to_numeric(out[f],errors="coerce")-med[f])/scale[f]
    return out

def load_data(zip_path):
    tmp=Path(tempfile.mkdtemp(prefix="step13_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            for fn in STATION_FILES.values():
                z.extract(f"controlled-fault-benchmark-v1.0/{fn}",tmp)
        base=tmp/"controlled-fault-benchmark-v1.0"
        need=FEATURES+["anomaly","anomaly_type","event_id","segment_id"]
        station_frames={}
        for st,fn in STATION_FILES.items():
            df=pd.read_csv(base/fn,usecols=need)
            station_frames[st]=df
        return station_frames
    finally:
        shutil.rmtree(tmp,ignore_errors=True)

def build_event_sets(station_frames,duration_hours=None,raw=False):
    frames=[]
    audit=[]
    for st,df in station_frames.items():
        segs=sorted(df["segment_id"].dropna().unique())
        first=segs[0]
        day1=df[df["segment_id"].eq(first)].copy()
        evaldf=df[~df["segment_id"].eq(first)].copy()

        if raw:
            transformed=evaldf
            ref_rows=0
            ref_anom=0
        else:
            n=min(len(day1),int(duration_hours*3600))
            ref=day1.iloc[:n].copy()
            med,scale=robust_reference(ref)
            transformed=apply_rlr(evaldf,med,scale)
            ref_rows=len(ref)
            ref_anom=int(ref["anomaly"].eq(1).sum())

        transformed=transformed[transformed["anomaly"].eq(1)].copy()
        transformed["station"]=st
        transformed["event_uid"]=st+"::"+transformed["event_id"].astype(str)
        ev=transformed.groupby(["station","event_uid","anomaly_type"],as_index=False)[FEATURES].median()
        frames.append(ev)
        audit.append({
            "station":st,
            "duration_h":"RAW" if raw else duration_hours,
            "reference_rows":ref_rows,
            "reference_anomaly_rows":ref_anom,
            "evaluation_rows":len(evaldf),
            "evaluation_anomaly_rows":int(evaldf["anomaly"].eq(1).sum()),
            "overlap_rows":0
        })
    data=pd.concat(frames,ignore_index=True)
    all_types=sorted(data["anomaly_type"].dropna().unique())
    main13=sorted([c for c in all_types if c not in SPARSE_TYPES])
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
        n_estimators=trees,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=1,
        max_features="sqrt"
    )
    rf.fit(train[FEATURES],train["anomaly_type"])
    return rf

def entropy_from_model(model,test):
    P=model.predict_proba(test[FEATURES])
    eps=1e-12
    ent=-(P*np.log(np.clip(P,eps,1))).sum(axis=1)/np.log(P.shape[1])
    pred=np.asarray(model.classes_,dtype=object)[P.argmax(axis=1)]
    return ent,pred

def make_meta_entropy(outer_train,heldout,trees):
    sts=sorted(outer_train["station"].unique())
    pseudo_types=sorted([c for c in outer_train["anomaly_type"].unique() if c!=heldout])
    rows=[]
    for src,tgt in [(sts[0],sts[1]),(sts[1],sts[0])]:
        source=outer_train[outer_train["station"].eq(src)]
        target=outer_train[outer_train["station"].eq(tgt)]
        for pseudo in pseudo_types:
            fit=source[source["anomaly_type"]!=pseudo].copy()
            if fit["anomaly_type"].nunique()<2:
                continue
            fit,tgt_imp=impute(fit,target)
            rf=fit_rf(fit,trees)
            ent,_=entropy_from_model(rf,tgt_imp)
            rows.append(pd.DataFrame({
                "Entropy":ent,
                "meta_y":(target["anomaly_type"].to_numpy()==pseudo).astype(int)
            }))
    return pd.concat(rows,ignore_index=True)

def optimize_threshold(scorev,y):
    scorev=np.asarray(scorev,float); y=np.asarray(y,int)
    best=None
    for tau in np.unique(np.quantile(scorev,np.linspace(.40,.995,150))):
        ka=float(np.mean(scorev[y==0]<=tau))
        if ka+1e-12<MIN_KNOWN_ACCEPT:
            continue
        ur=float(np.mean(scorev[y==1]>tau))
        h=0 if ka+ur==0 else 2*ka*ur/(ka+ur)
        key=(ur,h,-tau)
        if best is None or key>best[0]:
            best=(key,float(tau),ka,ur,h)
    if best is None:
        tau=float(np.quantile(scorev[y==0],MIN_KNOWN_ACCEPT))
        return tau
    return best[1]

def eval_open(ytrue,base_pred,scorev,tau,heldout,known_classes):
    ytrue=np.asarray(ytrue,dtype=object)
    base_pred=np.asarray(base_pred,dtype=object)
    is_u=ytrue==heldout
    is_k=~is_u
    reject=np.asarray(scorev)>tau
    out=base_pred.copy(); out[reject]="UNKNOWN"
    target=ytrue.copy(); target[is_u]="UNKNOWN"

    ka=float(np.mean(~reject[is_k]))
    kc=float(np.mean(out[is_k]==target[is_k]))
    ur=float(np.mean(reject[is_u]))
    h=0 if kc+ur==0 else 2*kc*ur/(kc+ur)
    auc=float(roc_auc_score(is_u.astype(int),scorev))
    mf1=float(f1_score(target,out,labels=list(known_classes)+["UNKNOWN"],average="macro",zero_division=0))
    return ka,kc,ur,h,auc,mf1

def run_one(data,main13,test_station,label,inner_trees,outer_trees):
    otr_all=data[data["station"]!=test_station]
    ote=data[data["station"]==test_station]
    rows=[]
    for heldout in main13:
        otr=otr_all[otr_all["anomaly_type"]!=heldout].copy()
        otr,ote_imp=impute(otr,ote)

        meta=make_meta_entropy(otr,heldout,inner_trees)
        tau=optimize_threshold(meta["Entropy"].to_numpy(),meta["meta_y"].to_numpy(int))

        rf=fit_rf(otr,outer_trees)
        ent,pred=entropy_from_model(rf,ote_imp)
        known_classes=sorted(otr["anomaly_type"].unique())
        vals=eval_open(ote["anomaly_type"],pred,ent,tau,heldout,known_classes)

        rows.append({
            "test_station":test_station,
            "heldout_type":heldout,
            "reference_duration_h":label,
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
    station_frames=load_data(zip_path)
    all_results=[]; audits=[]

    raw,main13,audit=build_event_sets(station_frames,raw=True)
    audits.append(audit)
    all_results.append(run_one(raw,main13,test_station,"RAW",inner_trees,outer_trees))
    print(test_station,"RAW done")

    for h in DURATIONS:
        data,main13,audit=build_event_sets(station_frames,duration_hours=h,raw=False)
        audits.append(audit)
        all_results.append(run_one(data,main13,test_station,h,inner_trees,outer_trees))
        print(test_station,h,"h done")

    rr=pd.concat(all_results,ignore_index=True)
    rr.to_csv(out/f"reference_duration_results_{test_station}.csv",index=False)
    pd.concat(audits,ignore_index=True).drop_duplicates().to_csv(
        out/f"reference_duration_audit_{test_station}.csv",index=False
    )

    summ=rr.groupby("reference_duration_h")[[
        "unknown_auroc","unknown_recall","known_acceptance",
        "known_correct_rate","h_score","open_macro_f1"
    ]].mean()
    print("\n",test_station)
    print(summ.to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--station",choices=["S1","S2","S3"],required=True)
    ap.add_argument("--inner-trees",type=int,default=8)
    ap.add_argument("--outer-trees",type=int,default=30)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    main(a.zip,a.out,a.station,a.inner_trees,a.outer_trees)
