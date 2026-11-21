"""Rebuild the result figures of the paper (Figs. 3 and 4) from results/*.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":8.5,"axes.edgecolor":"#52514e","axes.labelcolor":"#0b0b0b",
                     "xtick.color":"#52514e","ytick.color":"#52514e","pdf.fonttype":42})
import os
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"..","results"); OUT=os.path.join(HERE,"figures"); os.makedirs(OUT,exist_ok=True)
r=pd.read_csv(os.path.join(RES,"main_results.csv"))
fx=pd.read_csv(os.path.join(RES,"factorial_effects.csv")).rename(columns={"ci_low":"lo","ci_high":"hi","n_seeds":"n"})
MODELS=[("xlm-roberta-base","XLM-R"),("SI2M-Lab/DarijaBERT","DarijaBERT"),("SI2M-Lab/DarijaBERT-mix","DarijaBERT-mix")]
PANELS=[("myc","arabic","(a) MYC, Arabic → Latin"),("myc","latin","(b) MYC, Latin → Arabic"),("offensive","latin","(c) Offensive, Latin → Arabic")]
METH=[("vanilla","Vanilla","#a9a8a3"),("vanilla+TTN","Vanilla + TTN","#eb6834"),("aug_sample","Aug (sampled T)","#1baf7a"),("scriptbridge","ScriptBridge","#2a78d6")]
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5),sharey=True); w=0.19
for ax,(task,src,title) in zip(axes,PANELS):
    for gi,(mid,mn) in enumerate(MODELS):
        for mi,(meth,lab,col) in enumerate(METH):
            q=r[(r.task==task)&(r.src==src)&(r.model==mid)&(r.method==meth)]
            if q.empty: continue
            x=gi+(mi-1.5)*w*1.08; ax.bar(x,q.f1_cross.iloc[0],w,color=col,label=lab if gi==0 else None,linewidth=0)
            ax.errorbar(x,q.f1_cross.iloc[0],yerr=q.f1_cross_sd.iloc[0],color="#52514e",lw=0.7,capsize=1.2)
        u=r[(r.task==task)&(r.src==src)&(r.model==mid)&(r.method=="upper_dual")]
        if not u.empty: ax.hlines(u.f1_cross.iloc[0],gi-0.45,gi+0.45,colors="#0b0b0b",linestyles=(0,(3,2)),lw=0.9)
    ax.set_xticks(range(3)); ax.set_xticklabels(["XLM-R","Darija-\nBERT","Darija-\nBERT-mix"],fontsize=7.2); ax.set_title(title,fontsize=8)
    ax.set_ylim(30,90); ax.grid(axis="y",color="#e4e3df",lw=0.6); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Cross-script macro-F1 (%)")
h,l=axes[0].get_legend_handles_labels(); h.append(plt.Line2D([],[],color="#0b0b0b",ls=(0,(3,2)),lw=0.9)); l.append("Supervised dual-script reference")
fig.legend(h,l,ncol=5,loc="upper center",bbox_to_anchor=(0.5,1.07),frameon=False,fontsize=7.4,handlelength=1.4)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(os.path.join(OUT,"fig_main.pdf"),bbox_inches="tight"); fig.savefig(os.path.join(OUT,"fig_main.png"),dpi=200,bbox_inches="tight")

# factorial effects forest plot
SET=[("myc","arabic→latin","MYC Ar→La"),("myc","latin→arabic","MYC La→Ar"),("offensive","latin→arabic","Off. La→Ar")]
EFF=[("sampling","Sampling","#1baf7a"),("KL","KL","#eda100"),("InfoNCE","InfoNCE","#2a78d6")]
fig,axes=plt.subplots(1,2,figsize=(7.4,4.3))
for ax,(metric,xl) in zip(axes,[("f1_cross","(a) Effect on cross-script macro-F1 (points)"),("r1_raw","(b) Effect on retrieval R@1 (points)")]):
    rows=[]; y=0; yt=[]; yl=[]
    for mid,mn in MODELS[:2]:
        for task,d,sl in SET:
            for ei,(e,el,col) in enumerate(EFF):
                q=fx[(fx.model==mid)&(fx.task==task)&(fx.direction==d)&(fx.metric==metric)&(fx.effect==e)]
                if q.empty: continue
                q=q.iloc[0]; yy=y+(1-ei)*0.26
                ax.errorbar(q["mean"],yy,xerr=[[q["mean"]-q.lo],[q.hi-q["mean"]]],fmt="o",color=col,ms=5,lw=1.4,capsize=2,label=el if (y==0) else None)
            yt.append(y); yl.append(f"{mn}\n{sl}"); y-=1
    ax.axvline(0,color="#52514e",lw=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=8.5)
    ax.set_xlabel(xl,fontsize=9); ax.grid(axis="x",color="#e4e3df",lw=0.6); ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,ncol=3,loc="upper center",bbox_to_anchor=(0.5,1.04),frameon=False,fontsize=9.5)
fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(os.path.join(OUT,"fig_factorial.pdf"),bbox_inches="tight"); fig.savefig(os.path.join(OUT,"fig_factorial.png"),dpi=200,bbox_inches="tight")
