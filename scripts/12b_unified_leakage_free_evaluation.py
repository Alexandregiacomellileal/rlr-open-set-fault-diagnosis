#!/usr/bin/env python3
from pathlib import Path
import argparse, zipfile, tempfile, shutil, warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

FEATURES=[
    "voltage_V","current_A","power_W","frequency_Hz","power_factor",
    "PE_m3","PE_m4","PE_m5","WPE","MSPE","CV","delta_PF","SII"
]
SCORES=["MSP","Entropy","Margin","Euclidean","Mahalanobis","kNN"]
STATION_FILES={
    "S1":"station1_normal_features.csv",
    "S2":"station2_medium_features.csv",
    "S3":"station3_high_features.csv",
}
SPARSE_TYPES={"inrush_surge","spike_transient","overload"}
SEED=42
MIN_KNOWN_ACCEPT=0.85
ALPHAS=np.linspace(0,1,21)

def robust_reference(df, first_segment):
    ref=df[df["segment_id"].eq(first_segment)]
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
    return med,scale,len(ref)

def apply_rlr(df,med,scale):
    out=df.copy()
    for f in FEATURES:
        out[f]=(pd.to_numeric(out[f],errors="coerce")-med[f])/scale[f]
    return out

def load_events(zip_path):
    tmp=Path(tempfile.mkdtemp(prefix="step12b_"))
    try:
        with zipfile.ZipFile(zip_path) as z:
            for fn in STATION_FILES.values():
                z.extract(f"controlled-fault-benchmark-v1.0/{fn}",tmp)
        base=tmp/"controlled-fault-benchmark-v1.0"
        need=FEATURES+["anomaly","anomaly_type","event_id","segment_id"]
        raw_frames=[]; rlr_frames=[]; audits=[]
        for st,fn in STATION_FILES.items():
            df=pd.read_csv(base/fn,usecols=need)
            segs=sorted(df["segment_id"].dropna().unique())
            first=segs[0]
            med,scale,nref=robust_reference(df,first)

            ref=df[df["segment_id"].eq(first)]
            evaldf=df[~df["segment_id"].eq(first)].copy()

            audits.append({
                "station":st,
                "reference_segment":first,
                "reference_rows":len(ref),
                "reference_anomaly_rows":int(ref["anomaly"].eq(1).sum()),
                "evaluation_rows":len(evaldf),
                "evaluation_anomaly_rows":int(evaldf["anomaly"].eq(1).sum()),
                "reference_evaluation_overlap_rows":0
            })

            evaldf=evaldf[evaldf["anomaly"].eq(1)].copy()
            evaldf["station"]=st
            evaldf["event_uid"]=st+"::"+evaldf["event_id"].astype(str)

            raw=evaldf.copy()
            rlr=apply_rlr(evaldf,med,scale)

            for rep,src,holder in [("RAW",raw,raw_frames),("RLR",rlr,rlr_frames)]:
                ev=src.groupby(["station","event_uid","anomaly_type"],as_index=False)[FEATURES].median()
                ev["representation"]=rep
                holder.append(ev)

        raw_events=pd.concat(raw_frames,ignore_index=True)
        rlr_events=pd.concat(rlr_frames,ignore_index=True)
        all_types=sorted(raw_events["anomaly_type"].dropna().unique())
        main13=sorted([c for c in all_types if c not in SPARSE_TYPES])

        raw_events=raw_events[raw_events["anomaly_type"].isin(main13)].reset_index(drop=True)
        rlr_events=rlr_events[rlr_events["anomaly_type"].isin(main13)].reset_index(drop=True)
        return raw_events,rlr_events,main13,pd.DataFrame(audits)
    finally:
        shutil.rmtree(tmp,ignore_errors=True)

def impute_train_test(train,test):
    med=train[FEATURES].median(numeric_only=True)
    tr=train.copy(); te=test.copy()
    tr[FEATURES]=tr[FEATURES].fillna(med)
    te[FEATURES]=te[FEATURES].fillna(med)
    return tr,te

def fit_base(train,trees):
    rf=RandomForestClassifier(
        n_estimators=trees,class_weight="balanced_subsample",
        random_state=SEED,n_jobs=-1,max_features="sqrt"
    )
    rf.fit(train[FEATURES],train["anomaly_type"])
    return rf

def fit_geo(train):
    scaler=StandardScaler()
    X=scaler.fit_transform(train[FEATURES])
    y=train["anomaly_type"].to_numpy()
    classes=sorted(np.unique(y))
    centroids={c:X[y==c].mean(axis=0) for c in classes}
    residuals=np.vstack([X[i]-centroids[y[i]] for i in range(len(y))])
    precision=LedoitWolf().fit(residuals).precision_
    nn=NearestNeighbors(n_neighbors=min(5,len(X))).fit(X)
    return scaler,classes,centroids,precision,nn

def score(model,geo,test):
    P=model.predict_proba(test[FEATURES])
    classes=np.asarray(model.classes_,dtype=object)
    ss=np.sort(P,axis=1); p1=ss[:,-1]; p2=ss[:,-2] if P.shape[1]>1 else np.zeros_like(p1)
    eps=1e-12
    z={
        "MSP":1-p1,
        "Entropy":-(P*np.log(np.clip(P,eps,1))).sum(axis=1)/np.log(P.shape[1]),
        "Margin":1-(p1-p2)
    }
    scaler,gclasses,centroids,precision,nn=geo
    X=scaler.transform(test[FEATURES])
    eu=[];ma=[]
    for x in X:
        de=[];dm=[]
        for c in gclasses:
            d=x-centroids[c]
            de.append(np.linalg.norm(d))
            dm.append(np.sqrt(max(0,float(d@precision@d))))
        eu.append(min(de)); ma.append(min(dm))
    z["Euclidean"]=np.asarray(eu); z["Mahalanobis"]=np.asarray(ma)
    kd,_=nn.kneighbors(X)
    z["kNN"]=kd.mean(axis=1)
    pred=classes[P.argmax(axis=1)]
    return pd.DataFrame(z),pred

def make_meta_rows(outer_train,heldout,trees):
    sts=sorted(outer_train["station"].unique())
    pseudo_types=sorted([c for c in outer_train["anomaly_type"].unique() if c!=heldout])
    blocks=[]
    for src,tgt in [(sts[0],sts[1]),(sts[1],sts[0])]:
        source=outer_train[outer_train["station"].eq(src)]
        target=outer_train[outer_train["station"].eq(tgt)]
        for pseudo in pseudo_types:
            fit=source[source["anomaly_type"]!=pseudo].copy()
            if fit["anomaly_type"].nunique()<2:
                continue
            fit,tgt_imp=impute_train_test(fit,target)
            base=fit_base(fit,trees)
            geo=fit_geo(fit)
            Z,_=score(base,geo,tgt_imp)
            b=Z.copy()
            b["meta_y"]=(target["anomaly_type"].to_numpy()==pseudo).astype(int)
            b["pseudo"]=pseudo
            b["direction"]=f"{src}->{tgt}"
            blocks.append(b)
    return pd.concat(blocks,ignore_index=True)

def fit_meta(df):
    X=df[SCORES].replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median(numeric_only=True))
    y=df["meta_y"].to_numpy()
    model=Pipeline([
        ("scaler",StandardScaler()),
        ("lr",LogisticRegression(max_iter=2000,class_weight="balanced",C=1.0,random_state=SEED))
    ])
    model.fit(X,y)
    return model

def crossfit_puf(meta_rows):
    pseudo_types=sorted(meta_rows["pseudo"].unique())
    foldmap={p:i%2 for i,p in enumerate(pseudo_types)}
    p_oof=np.full(len(meta_rows),np.nan)
    for fold in [0,1]:
        vm=meta_rows["pseudo"].map(foldmap).to_numpy()==fold
        tr=meta_rows.loc[~vm].copy()
        va=meta_rows.loc[vm].copy()
        model=fit_meta(tr)
        Xv=va[SCORES].replace([np.inf,-np.inf],np.nan)
        Xv=Xv.fillna(tr[SCORES].median(numeric_only=True))
        p_oof[vm]=model.predict_proba(Xv)[:,1]
    return p_oof

def optimize_threshold(scorev,y,min_accept=MIN_KNOWN_ACCEPT):
    scorev=np.asarray(scorev,float); y=np.asarray(y,int)
    best=None
    for tau in np.unique(np.quantile(scorev,np.linspace(.40,.995,150))):
        ka=float(np.mean(scorev[y==0]<=tau))
        if ka+1e-12<min_accept:
            continue
        ur=float(np.mean(scorev[y==1]>tau))
        h=0 if ka+ur==0 else 2*ka*ur/(ka+ur)
        key=(ur,h,-tau)
        if best is None or key>best[0]:
            best=(key,float(tau),ka,ur,h)
    if best is None:
        tau=float(np.quantile(scorev[y==0],min_accept))
        ka=float(np.mean(scorev[y==0]<=tau)); ur=float(np.mean(scorev[y==1]>tau))
        h=0 if ka+ur==0 else 2*ka*ur/(ka+ur)
        return tau,ka,ur,h
    _,tau,ka,ur,h=best
    return tau,ka,ur,h

def optimize_adaptive(meta_rows,p_oof):
    ent=meta_rows["Entropy"].to_numpy(float)
    y=meta_rows["meta_y"].to_numpy(int)
    best=None
    for a in ALPHAS:
        s=a*ent+(1-a)*p_oof
        tau,ka,ur,h=optimize_threshold(s,y)
        key=(ur,h,-tau)
        if best is None or key>best[0]:
            best=(key,float(a),tau,ka,ur,h)
    _,a,tau,ka,ur,h=best
    return a,tau,ka,ur,h

def eval_open(ytrue,base_pred,scorev,tau,heldout,known_classes):
    ytrue=np.asarray(ytrue,dtype=object); base_pred=np.asarray(base_pred,dtype=object)
    is_u=ytrue==heldout; is_k=~is_u
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

def run_representation(events,main13,test_station,rep,inner_trees,outer_trees):
    otr_all=events[events["station"]!=test_station]
    ote_all=events[events["station"]==test_station]
    results=[]

    for heldout in main13:
        otr=otr_all[otr_all["anomaly_type"]!=heldout].copy()
        ote=ote_all.copy()
        otr,ote_imp=impute_train_test(otr,ote)

        meta_rows=make_meta_rows(otr,heldout,inner_trees)
        p_oof=crossfit_puf(meta_rows)
        ymeta=meta_rows["meta_y"].to_numpy(int)

        base=fit_base(otr,outer_trees)
        geo=fit_geo(otr)
        Z,pred=score(base,geo,ote_imp)
        known_classes=sorted(otr["anomaly_type"].unique())

        tau_e,_,_,_=optimize_threshold(meta_rows["Entropy"].to_numpy(),ymeta)
        vals=eval_open(ote["anomaly_type"],pred,Z["Entropy"].to_numpy(),tau_e,heldout,known_classes)
        results.append(dict(test_station=test_station,heldout_type=heldout,representation=rep,
                            method=f"{rep}+Entropy",tau=tau_e,alpha_entropy=1.0,
                            known_acceptance=vals[0],known_correct_rate=vals[1],
                            unknown_recall=vals[2],h_score=vals[3],
                            unknown_auroc=vals[4],open_macro_f1=vals[5]))

        meta_final=fit_meta(meta_rows)
        Xo=Z[SCORES].replace([np.inf,-np.inf],np.nan)
        Xo=Xo.fillna(meta_rows[SCORES].median(numeric_only=True))
        puf_test=meta_final.predict_proba(Xo)[:,1]
        tau_p,_,_,_=optimize_threshold(p_oof,ymeta)
        vals=eval_open(ote["anomaly_type"],pred,puf_test,tau_p,heldout,known_classes)
        results.append(dict(test_station=test_station,heldout_type=heldout,representation=rep,
                            method=f"{rep}+PUF",tau=tau_p,alpha_entropy=0.0,
                            known_acceptance=vals[0],known_correct_rate=vals[1],
                            unknown_recall=vals[2],h_score=vals[3],
                            unknown_auroc=vals[4],open_macro_f1=vals[5]))

        a,tau_a,_,_,_=optimize_adaptive(meta_rows,p_oof)
        final_score=a*Z["Entropy"].to_numpy()+(1-a)*puf_test
        vals=eval_open(ote["anomaly_type"],pred,final_score,tau_a,heldout,known_classes)
        results.append(dict(test_station=test_station,heldout_type=heldout,representation=rep,
                            method=f"{rep}+AdaptiveGate",tau=tau_a,alpha_entropy=a,
                            known_acceptance=vals[0],known_correct_rate=vals[1],
                            unknown_recall=vals[2],h_score=vals[3],
                            unknown_auroc=vals[4],open_macro_f1=vals[5]))

        for meth in ["MSP","Margin","Euclidean","Mahalanobis","kNN"]:
            tau_s,_,_,_=optimize_threshold(meta_rows[meth].to_numpy(),ymeta)
            vals=eval_open(ote["anomaly_type"],pred,Z[meth].to_numpy(),tau_s,heldout,known_classes)
            results.append(dict(test_station=test_station,heldout_type=heldout,representation=rep,
                                method=f"{rep}+{meth}",tau=tau_s,alpha_entropy=np.nan,
                                known_acceptance=vals[0],known_correct_rate=vals[1],
                                unknown_recall=vals[2],h_score=vals[3],
                                unknown_auroc=vals[4],open_macro_f1=vals[5]))
        print(test_station,rep,heldout,"done")

    return pd.DataFrame(results)

def main(zip_path,out,test_station,inner_trees,outer_trees):
    raw,rlr,main13,audit=load_events(zip_path)
    audit.to_csv(out/f"leakage_audit_{test_station}.csv",index=False)

    r1=run_representation(raw,main13,test_station,"RAW",inner_trees,outer_trees)
    r2=run_representation(rlr,main13,test_station,"RLR",inner_trees,outer_trees)
    rr=pd.concat([r1,r2],ignore_index=True)
    rr.to_csv(out/f"unified_results_{test_station}.csv",index=False)

    summary=rr.groupby("method")[["unknown_auroc","unknown_recall","known_acceptance","known_correct_rate","h_score","open_macro_f1"]].mean()
    print("\n",test_station)
    print(summary.sort_values("h_score",ascending=False).to_string())

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--zip",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--station",choices=["S1","S2","S3"],required=True)
    ap.add_argument("--inner-trees",type=int,default=20)
    ap.add_argument("--outer-trees",type=int,default=80)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    main(a.zip,a.out,a.station,a.inner_trees,a.outer_trees)
