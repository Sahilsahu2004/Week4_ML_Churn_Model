import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict, cross_validate, learning_curve, validation_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix,roc_curve,precision_recall_curve,ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
SEED=42; np.random.seed(SEED)
sns.set_theme(style="whitegrid")
URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw=pd.read_csv(URL); R={"raw_shape":raw.shape}
# ---------------- cleaning
df=raw.copy()
R["blank_total"]=int((df.TotalCharges.str.strip()=="").sum())
R["blank_tenure"]=df.loc[df.TotalCharges.str.strip()=="","tenure"].unique().tolist()
df["TotalCharges"]=pd.to_numeric(df.TotalCharges.str.strip().replace("",np.nan))
df.loc[df.TotalCharges.isna(),"TotalCharges"]=0.0
R["dups"]=int(df.duplicated().sum()); R["missing_after"]=int(df.isnull().sum().sum())
df=df.drop(columns="customerID")
df["Churn"]=(df.Churn=="Yes").astype(int)
df["SeniorCitizen"]=df.SeniorCitizen.map({0:"No",1:"Yes"})
R["churn_rate"]=round(float(df.Churn.mean()*100),1); R["churn_counts"]=df.Churn.value_counts().to_dict()
num=["tenure","MonthlyCharges","TotalCharges"]; cat=[c for c in df.columns if c not in num+["Churn"]]
R["n_cat"]=len(cat); R["cat"]=cat
R["churn_by_contract"]=(df.groupby("Contract").Churn.mean()*100).round(1).to_dict()
R["churn_by_internet"]=(df.groupby("InternetService").Churn.mean()*100).round(1).to_dict()
R["churn_by_pay"]=(df.groupby("PaymentMethod").Churn.mean()*100).round(1).to_dict()
R["tenure_med"]=df.groupby("Churn").tenure.median().to_dict(); R["monthly_mean"]=df.groupby("Churn").MonthlyCharges.mean().round(1).to_dict()
# ---------------- EDA fig
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
vc=df.Churn.value_counts().sort_index(); ax[0].bar(["Stayed","Churned"],vc.values,color=["#2E86AB","#D1495B"])
for i,v in enumerate(vc.values): ax[0].text(i,v+80,f"{v:,}\n({v/len(df)*100:.1f}%)",ha="center",fontweight="bold")
ax[0].set_ylim(0,6200); ax[0].set_title("Target is imbalanced",fontweight="bold")
cc=df.groupby("Contract").Churn.mean()*100; cc=cc.reindex(["Month-to-month","One year","Two year"])
ax[1].bar(cc.index,cc.values,color=["#D1495B","#E6AA68","#2E86AB"]); [ax[1].text(i,v+.8,f"{v:.1f}%",ha="center",fontweight="bold") for i,v in enumerate(cc.values)]
ax[1].set_title("Churn rate by contract type",fontweight="bold"); ax[1].set_ylabel("% churned")
sns.kdeplot(data=df,x="tenure",hue=df.Churn.map({0:"Stayed",1:"Churned"}),fill=True,common_norm=False,palette={"Stayed":"#2E86AB","Churned":"#D1495B"},ax=ax[2],alpha=.4)
ax[2].set_title("New customers churn most",fontweight="bold"); ax[2].set_xlabel("Tenure (months)")
plt.tight_layout(); plt.savefig("m1_eda.png",dpi=160); plt.close()
# ---------------- split
X=df.drop(columns="Churn"); y=df.Churn
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
R["split"]=dict(train=len(Xtr),test=len(Xte),churn_train=round(float(ytr.mean()*100),1),churn_test=round(float(yte.mean()*100),1))
pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
Xt_ = pre.fit(Xtr); R["n_features_after"]=int(pre.transform(Xtr).shape[1])
# ---------------- models
spec={
 "Baseline (majority class)":(DummyClassifier(strategy="most_frequent"),{}),
 "Logistic Regression":(LogisticRegression(max_iter=2000,class_weight="balanced",random_state=SEED),{"clf__C":[.01,.03,.1,.3,1,10]}),
 "Decision Tree":(DecisionTreeClassifier(class_weight="balanced",random_state=SEED),{"clf__max_depth":[3,4,5,6,8,12],"clf__min_samples_leaf":[10,30,60]}),
 "Random Forest":(RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=SEED,n_jobs=-1),{"clf__max_depth":[6,10,16,None],"clf__min_samples_leaf":[3,10,25]}),
}
fitted={}; R["cv"]={}; R["best_params"]={}; R["test"]={}; probs={}; preds={}
for name,(clf,grid) in spec.items():
    pipe=Pipeline([("pre",pre),("clf",clf)])
    if grid:
        gs=GridSearchCV(pipe,grid,scoring="roc_auc",cv=cv,n_jobs=-1,refit=True).fit(Xtr,ytr); est=gs.best_estimator_; R["best_params"][name]={k.replace("clf__",""):v for k,v in gs.best_params_.items()}
    else: est=pipe.fit(Xtr,ytr)
    fitted[name]=est
    sc=cross_validate(est,Xtr,ytr,cv=cv,scoring=["roc_auc","recall","precision","f1","accuracy"],return_train_score=True,n_jobs=-1)
    R["cv"][name]={m:[round(float(sc[f"test_{m}"].mean()),3),round(float(sc[f"test_{m}"].std()),3)] for m in ["roc_auc","recall","precision","f1","accuracy"]}
    R["cv"][name]["train_auc"]=round(float(sc["train_roc_auc"].mean()),3)
    p=est.predict_proba(Xte)[:,1]; yh=est.predict(Xte); probs[name]=p; preds[name]=yh
    tn,fp,fn,tp=confusion_matrix(yte,yh).ravel()
    R["test"][name]=dict(acc=round(accuracy_score(yte,yh),3),prec=round(precision_score(yte,yh,zero_division=0),3),rec=round(recall_score(yte,yh),3),f1=round(f1_score(yte,yh),3),auc=round(roc_auc_score(yte,p),3),ap=round(average_precision_score(yte,p),3),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
      train_auc_full=round(float(roc_auc_score(ytr,est.predict_proba(Xtr)[:,1])),3))
best=max([n for n in spec if n!="Baseline (majority class)"],key=lambda n:R["cv"][n]["roc_auc"][0]); R["best"]=best
# ---------------- fig2 confusion matrices
names=["Logistic Regression","Decision Tree","Random Forest"]
fig,ax=plt.subplots(1,3,figsize=(14,4.4))
for a,n in zip(ax,names):
    ConfusionMatrixDisplay(confusion_matrix(yte,preds[n]),display_labels=["Stayed","Churned"]).plot(ax=a,cmap="Blues",colorbar=False,values_format="d")
    t=R["test"][n]; a.set_title(f"{n}\nrecall {t['rec']:.2f} | precision {t['prec']:.2f}",fontweight="bold",fontsize=11); a.grid(False)
plt.tight_layout(); plt.savefig("m2_confusion.png",dpi=160); plt.close()
# ---------------- fig3 ROC + PR
cols={"Baseline (majority class)":"#999","Logistic Regression":"#2E86AB","Decision Tree":"#E6AA68","Random Forest":"#6D597A"}
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
for n in spec:
    p=probs[n]; fpr,tpr,_=roc_curve(yte,p); ax[0].plot(fpr,tpr,color=cols[n],lw=2.5 if n!="Baseline (majority class)" else 1.5,ls="--" if n.startswith("Base") else "-",label=f"{n} (AUC {R['test'][n]['auc']:.3f})")
    if not n.startswith("Base"):
        pr,rc,_=precision_recall_curve(yte,p); ax[1].plot(rc,pr,color=cols[n],lw=2.5,label=f"{n} (AP {R['test'][n]['ap']:.3f})")
ax[0].plot([0,1],[0,1],color="#bbb",ls=":"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate (recall)"); ax[0].set_title("ROC curves (test set)",fontweight="bold"); ax[0].legend(loc="lower right",fontsize=9)
ax[1].axhline(yte.mean(),color="#999",ls="--",label=f"Baseline (churn rate {yte.mean():.2f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall curves (test set)",fontweight="bold"); ax[1].legend(fontsize=9); ax[1].set_ylim(0,1.02)
plt.tight_layout(); plt.savefig("m3_roc_pr.png",dpi=160); plt.close()
# ---------------- fig4 overfitting
def tree_pipe(): return Pipeline([("pre",pre),("clf",DecisionTreeClassifier(class_weight="balanced",min_samples_leaf=10,random_state=SEED))])
depths=[1,2,3,4,5,6,8,10,12,16,20]
trs,vas=validation_curve(tree_pipe(),Xtr,ytr,param_name="clf__max_depth",param_range=depths,cv=cv,scoring="roc_auc",n_jobs=-1)
R["depth_curve"]={"depths":depths,"train":trs.mean(1).round(3).tolist(),"val":vas.mean(1).round(3).tolist()}
sizes,trl,val=learning_curve(fitted["Logistic Regression"],Xtr,ytr,cv=cv,scoring="roc_auc",train_sizes=np.linspace(.1,1,8),n_jobs=-1)
sizes_rf,trr,vlr=learning_curve(fitted["Random Forest"],Xtr,ytr,cv=cv,scoring="roc_auc",train_sizes=np.linspace(.1,1,6),n_jobs=-1)
R["lc"]={"lr":[round(float(trl[-1].mean()),3),round(float(val[-1].mean()),3)],"rf":[round(float(trr[-1].mean()),3),round(float(vlr[-1].mean()),3)]}
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
ax[0].plot(depths,trs.mean(1),"o-",color="#D1495B",label="Training AUC"); ax[0].plot(depths,vas.mean(1),"o-",color="#2E86AB",label="Cross-validated AUC")
bd=depths[int(np.argmax(vas.mean(1)))]; ax[0].axvline(bd,color="#888",ls="--"); ax[0].text(bd+.3,.935,f"best depth = {bd}",fontsize=9)
ax[0].fill_between(depths[5:],vas.mean(1)[5:],trs.mean(1)[5:],color="#D1495B",alpha=.12); ax[0].text(12,.855,"overfitting\ngap",color="#D1495B",fontweight="bold",ha="center")
ax[0].text(1.1,.745,"underfitting",color="#2E86AB",fontweight="bold",fontsize=9); ax[0].set_ylim(.7,.95)
ax[0].set_xlabel("Decision tree max depth"); ax[0].set_ylabel("ROC-AUC"); ax[0].set_title("Decision tree: complexity vs fit",fontweight="bold"); ax[0].legend(loc="lower right")
for a,(s,tr,va,t) in zip(ax[1:],[(sizes,trl,val,"Logistic regression learning curve"),(sizes_rf,trr,vlr,"Random forest learning curve")]):
    a.plot(s,tr.mean(1),"o-",color="#D1495B",label="Training"); a.plot(s,va.mean(1),"o-",color="#2E86AB",label="Cross-validated"); a.fill_between(s,va.mean(1)-va.std(1),va.mean(1)+va.std(1),color="#2E86AB",alpha=.15)
    a.set_xlabel("Training set size"); a.set_ylabel("ROC-AUC"); a.set_title(t,fontweight="bold"); a.legend(loc="upper right"); a.set_ylim(.7,1.0)
plt.tight_layout(); plt.savefig("m4_fit.png",dpi=160); plt.close()
R["best_depth"]=bd
# ---------------- fig5 interpretability (logistic coefficients + permutation importance for best)
lr=fitted["Logistic Regression"]; fn_=lr.named_steps["pre"].get_feature_names_out(); coef=lr.named_steps["clf"].coef_[0]
cs=pd.Series(coef,index=[f.replace("num__","").replace("cat__","") for f in fn_]).sort_values()
R["top_pos"]={k:round(float(np.exp(v)),2) for k,v in cs.tail(8).iloc[::-1].items()}; R["top_neg"]={k:round(float(np.exp(v)),2) for k,v in cs.head(8).items()}
sel=pd.concat([cs.head(8),cs.tail(8)])
fig,ax=plt.subplots(1,2,figsize=(14,5.5))
ax[0].barh(sel.index,sel.values,color=["#2E86AB" if v<0 else "#D1495B" for v in sel.values]); ax[0].axvline(0,color="k",lw=.8); ax[0].set_title("Logistic regression: strongest drivers\n(red raises churn risk, blue lowers it)",fontweight="bold",fontsize=11); ax[0].set_xlabel("Coefficient (log-odds, standardised numerics)")
pi=permutation_importance(fitted[best],Xte,yte,scoring="roc_auc",n_repeats=15,random_state=SEED,n_jobs=-1)
ps=pd.Series(pi.importances_mean,index=X.columns).sort_values().tail(10)
ax[1].barh(ps.index,ps.values,xerr=pd.Series(pi.importances_std,index=X.columns)[ps.index].values,color="#6D597A"); ax[1].set_title(f"Permutation importance ({best}, test set)\n(drop in AUC when the feature is shuffled)",fontweight="bold",fontsize=11); ax[1].set_xlabel("Decrease in ROC-AUC")
plt.tight_layout(); plt.savefig("m5_features.png",dpi=160); plt.close()
R["perm_top"]=ps.iloc[::-1].round(4).to_dict()
# ---------------- threshold tuning on OUT-OF-FOLD train predictions
oof=cross_val_predict(fitted[best],Xtr,ytr,cv=cv,method="predict_proba")[:,1]
ths=np.linspace(.05,.95,91); COST_FN,COST_FP=500,50
def met(y_,p_,t):
    yh=(p_>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y_,yh).ravel(); return dict(prec=precision_score(y_,yh,zero_division=0),rec=recall_score(y_,yh),f1=f1_score(y_,yh),cost=fn*COST_FN+(fp+tp)*COST_FP,fn=fn,fp=fp,tp=tp,tn=tn)
oofm=pd.DataFrame([met(ytr,oof,t) for t in ths],index=ths)
t_f1=float(oofm.f1.idxmax()); t_cost=float(oofm.cost.idxmin())
R["thr"]={"f1":round(t_f1,2),"cost":round(t_cost,2),"cost_fn":COST_FN,"cost_fp":COST_FP}
pb=probs[best]
R["thr_test"]={}
for lab,t in [("default 0.50",.5),("F1-optimal",t_f1),("cost-optimal",t_cost)]:
    m=met(yte,pb,t); R["thr_test"][lab]={"t":round(t,2),"prec":round(m["prec"],3),"rec":round(m["rec"],3),"f1":round(m["f1"],3),"cost":int(m["cost"]),"fn":int(m["fn"]),"fp":int(m["fp"]),"tp":int(m["tp"]),"tn":int(m["tn"])}
# no-model cost: contact nobody (all churners lost) / contact everyone
R["cost_nomodel"]=int(yte.sum()*COST_FN); R["cost_all"]=int(len(yte)*COST_FP)
tm=pd.DataFrame([met(yte,pb,t) for t in ths],index=ths)
fig,ax=plt.subplots(1,2,figsize=(13,4.8))
ax[0].plot(ths,tm.prec,label="Precision",color="#E6AA68",lw=2.5); ax[0].plot(ths,tm.rec,label="Recall",color="#2E86AB",lw=2.5); ax[0].plot(ths,tm.f1,label="F1",color="#6D597A",lw=2.5,ls="--")
for t,l,c_ in [(.5,"default 0.50","#888"),(t_cost,f"cost-optimal {t_cost:.2f}","#D1495B")]: ax[0].axvline(t,color=c_,ls=":"); ax[0].text(t+.01,.05,l,rotation=90,color=c_,fontsize=9)
ax[0].set_xlabel("Decision threshold"); ax[0].set_title(f"Precision-recall trade-off ({best})",fontweight="bold"); ax[0].legend(loc="center left")
ax[1].plot(ths,tm.cost/1000,color="#D1495B",lw=2.5); ax[1].axhline(R["cost_nomodel"]/1000,color="#888",ls="--"); ax[1].text(.45,R["cost_nomodel"]/1000-8,f"No model (lose every churner): ${R['cost_nomodel']/1000:,.0f}k",fontsize=9,color="#555")
ax[1].axhline(R["cost_all"]/1000,color="#2E86AB",ls="--"); ax[1].text(.45,R["cost_all"]/1000+3,f"Offer everyone: ${R['cost_all']/1000:,.0f}k",fontsize=9,color="#2E86AB")
ax[1].scatter([t_cost],[tm.loc[ths[np.argmin(abs(ths-t_cost))],"cost"]/1000],s=90,color="k",zorder=3); ax[1].set_xlabel("Decision threshold"); ax[1].set_ylabel("Total cost on test set ($ thousands)"); ax[1].set_title(f"Business cost (missed churner \\${COST_FN}, each offer \\${COST_FP})",fontweight="bold",fontsize=11)
plt.tight_layout(); plt.savefig("m6_threshold.png",dpi=160); plt.close()
# ---------------- error analysis by segment at cost-optimal threshold
ea=Xte.copy(); ea["y"]=yte.values; ea["p"]=pb; ea["pred"]=(pb>=t_cost).astype(int)
ea["TenureBand"]=pd.cut(ea.tenure,[-1,12,24,48,100],labels=["0-12","13-24","25-48","49+"])
def seg(col):
    g=ea.groupby(col,observed=True).apply(lambda d:pd.Series({"n":len(d),"churners":int(d.y.sum()),"recall":(d[d.y==1].pred.mean() if d.y.sum() else np.nan),"precision":(d[d.pred==1].y.mean() if d.pred.sum() else np.nan)}),include_groups=False); return g
sc_=seg("Contract"); st_=seg("TenureBand"); si_=seg("InternetService")
R["seg"]={"Contract":sc_.round(3).to_dict("index"),"Tenure":st_.round(3).to_dict("index"),"Internet":si_.round(3).to_dict("index")}
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
for a,(g,t) in zip(ax,[(sc_,"Contract type"),(st_,"Tenure (months)"),(si_,"Internet service")]):
    x=np.arange(len(g)); a.bar(x-.18,g.recall,.36,color="#2E86AB",label="Recall"); a.bar(x+.18,g.precision,.36,color="#E6AA68",label="Precision")
    a.set_xticks(x); a.set_xticklabels([f"{i}\n(n={int(n)})" for i,n in zip(g.index,g.n)],fontsize=9); a.set_ylim(0,1.05); a.set_title(t,fontweight="bold")
    for i,(r_,p_) in enumerate(zip(g.recall,g.precision)):
        a.text(i-.18,r_+.02,f"{r_:.2f}",ha="center",fontsize=8.5); a.text(i+.18,(0 if np.isnan(p_) else p_)+.02,"n/a" if np.isnan(p_) else f"{p_:.2f}",ha="center",fontsize=8.5)
ax[0].legend(loc="upper right",fontsize=9); ax[0].set_ylabel("Score at cost-optimal threshold")
plt.tight_layout(); plt.savefig("m7_errors.png",dpi=160); plt.close()

# ---------- bootstrap uncertainty on test set
rng=np.random.RandomState(SEED); yv=yte.values; n_=len(yv); aucs={k:[] for k in ["Logistic Regression","Decision Tree","Random Forest"]}; diffs=[]
for _ in range(1000):
    idx=rng.randint(0,n_,n_)
    if yv[idx].sum() in (0,n_): continue
    for k in aucs: aucs[k].append(roc_auc_score(yv[idx],probs[k][idx]))
    diffs.append(aucs["Random Forest"][-1]-aucs["Logistic Regression"][-1])
R["boot_auc"]={k:[round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)] for k,v in aucs.items()}
R["boot_diff_rf_lr"]=[round(float(np.mean(diffs)),4),round(float(np.percentile(diffs,2.5)),4),round(float(np.percentile(diffs,97.5)),4)]
# calibration quick stat
from sklearn.metrics import brier_score_loss
R["brier"]={n:round(float(brier_score_loss(yte,probs[n])),3) for n in names}
R["n_fn_noncontract"]=None
json.dump(R,open("res.json","w"),indent=1,default=str)
import joblib; joblib.dump(fitted[best],"best_model.joblib")
print(json.dumps({k:v for k,v in R.items() if k not in ["cat","seg"]},indent=1,default=str)); print(json.dumps(R["seg"],indent=1,default=str))
