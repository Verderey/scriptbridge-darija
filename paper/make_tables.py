"""Rebuild the LaTeX tables of the paper from results/*.csv.

The tables use an lrbox called \\tblbox and a \\fitbox macro, defined in the paper preamble:
    \\newsavebox{\\tblbox}
    \\newcommand{\\fitbox}[1]{\\ifdim\\wd\\tblbox>#1\\resizebox{#1}{!}{\\usebox{\\tblbox}}\\else\\usebox{\\tblbox}\\fi}
"""
import pandas as pd, numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"..","results"); OUT=os.path.join(HERE,"tables"); os.makedirs(OUT,exist_ok=True)
r=pd.read_csv(os.path.join(RES,"main_results.csv"))
sig=pd.read_csv(os.path.join(RES,"contrasts.csv")).rename(columns={"ci_low":"lo","ci_high":"hi","n_seeds":"n","significant":"sig"})
M=[("xlm-roberta-base","XLM-R"),("SI2M-Lab/DarijaBERT","DarijaBERT"),("SI2M-Lab/DarijaBERT-mix","DarijaBERT-mix")]
S=[("myc","arabic","MYC","Ar$\\to$La"),("myc","latin","MYC","La$\\to$Ar"),("offensive","latin","Off.","La$\\to$Ar")]
N={"vanilla":"Vanilla","vanilla+TTN":"Vanilla + TTN","vanilla+TTN_uroman":"Vanilla + TTN (uroman)","aug_det":"Aug, argmax $T$","det_kl":"Aug + KL, argmax $T$",
   "augcon_det":"Aug + InfoNCE, argmax $T$","sb_det":"Full objective, argmax $T$","aug_sample":"Aug, sampled $T$","sb_noCon":"Aug + KL, sampled $T$",
   "sb_noKL":"Aug + InfoNCE, sampled $T$","scriptbridge":"\\textbf{ScriptBridge} (full, sampled $T$)","scriptbridge+TTN":"ScriptBridge + TTN",
   "augcon_uroman":"Aug + InfoNCE, uroman","rdrop":"R-Drop (no transliteration)","charnoise":"Char.\\ noise + full objective","upper_dual":"Supervised dual-script reference$^{\\ddagger}$",
   "aug_sample+TTN":"Aug (sampled) + TTN"}
GROUPS=[["vanilla","vanilla+TTN","vanilla+TTN_uroman"],["aug_det","det_kl","augcon_det","sb_det"],["aug_sample","sb_noCon","sb_noKL","scriptbridge"],
        ["augcon_uroman","rdrop","charnoise"],["scriptbridge+TTN"],["upper_dual"]]
def g(task,model,src,meth,col):
    q=r[(r.task==task)&(r.model==model)&(r.src==src)&(r.method==meth)]
    return np.nan if q.empty else q.iloc[0][col]
def metric_table(col,label,caption):
    cols=[(m,t,s) for m,_ in M for t,s,_,_ in S]
    zs=[m for g_ in GROUPS for m in g_ if m!="upper_dual"]
    best={c: np.nanmax([g(c[1],c[0],c[2],m,col) for m in zs]) for c in cols}
    L=["\\begin{table*}[t]","\\caption{"+caption+"}","\\label{"+label+"}","\\centering\\footnotesize","\\setlength{\\tabcolsep}{3.2pt}",
       "\\begin{fittable}[\\textwidth]","\\begin{tabular}{@{}l"+"ccc"*3+"@{}}","\\toprule",
       " & \\multicolumn{3}{c}{XLM-R} & \\multicolumn{3}{c}{DarijaBERT} & \\multicolumn{3}{c}{DarijaBERT-mix$^{\\S}$}\\\\",
       "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
       "Method"+"".join(f" & \\shortstack{{{a}\\\\{b}}}" for _ in M for _,_,a,b in S)+"\\\\","\\midrule"]
    for gi,grp in enumerate(GROUPS):
        if gi: L.append("\\midrule")
        for m in grp:
            row=[N[m]]
            for c in cols:
                v=g(c[1],c[0],c[2],m,col)
                row.append("--" if np.isnan(v) else (f"\\textbf{{{v:.1f}}}" if (m!="upper_dual" and abs(v-best[c])<1e-9) else f"{v:.1f}"))
            L.append(" & ".join(row)+"\\\\")
    L+=["\\bottomrule","\\end{tabular}\\end{fittable}","\\end{table*}"]
    open(os.path.join(OUT,label.split(":")[1]+".tex"),"w").write("\n".join(L))
note="The nine prespecified clean settings are MYC in both directions and offensive Latin$\\to$Arabic for three encoders; offensive Arabic$\\to$Latin is analysed separately in Table~\\ref{tab:clean}. Values are means over five seeds for XLM-R and DarijaBERT and over three seeds for DarijaBERT-mix, which was trained on a reduced grid (marked $^{\\S}$). Per-seed standard deviations are released with the code. Predictions use the argmax decision rule. The best zero-shot system in each column is in bold, and a dash marks a system that was not run in that configuration. The row marked $^{\\ddagger}$ is a diagnostic reference that uses target-script labels and is not a zero-shot system."
metric_table("f1_cross","tab:f1","Cross-script macro-F1 (\\%) in the clean settings. "+note)
metric_table("auc_cross","tab:auc","Cross-script AUROC (\\%) in the clean settings. "+note)

# contrasts table
C=[("C1","Training-time vs.\\ test-time transliteration","aug\\_sample $-$ vanilla+TTN"),("C2","Learned vs.\\ universal transliterator","augcon\\_det $-$ augcon\\_uroman"),
   ("C3","Sampling vs.\\ argmax (CE only)","aug\\_sample $-$ aug\\_det"),("C4","Sampling vs.\\ argmax (full objective)","scriptbridge $-$ sb\\_det"),
   ("C5","Added value of KL + InfoNCE","scriptbridge $-$ aug\\_sample"),("C6","Transliteration vs.\\ generic perturbation","scriptbridge $-$ charnoise")]
DIRS={"arabic":"arabic→latin","latin":"latin→arabic"}
def cell(task,model,src,cid,metric):
    q=sig[(sig.task==task)&(sig.model==model)&(sig.direction==DIRS[src])&(sig.contrast==cid)&(sig.metric==metric)&(sig.family=="primary")]
    if q.empty: return "--"
    q=q.iloc[0]; s=f"{q.delta:+.1f}" if abs(q.delta)>=0.05 else "0.0"
    if q.p_holm<0.05: s+=("\\rlap{$^{*}$}" if q.delta>0 else "\\rlap{$^{\\circ}$}")
    return s
cols=[(m,t,s) for m,_ in M for t,s,_,_ in S]
L=["\\begin{table*}[t]","\\caption{Pre-declared primary contrasts in the nine clean settings, in points of cross-script macro-F1 and AUROC. Each entry is the mean over seeds of the per-seed difference between the two systems that define the contrast in Section~\\ref{sec:stats}; seeds are five for XLM-R and DarijaBERT and three for DarijaBERT-mix. A star marks a significantly positive and a circle a significantly negative difference, meaning a Holm-adjusted $p$-value below 0.05. Each Holm family contains all primary contrasts of one backbone and one metric over the clean settings, and the unadjusted two-sided $p$-values come from a hierarchical bootstrap over seeds and test examples (1{,}000 resamples). A dash marks a contrast that is not defined, either because uroman exists only for Arabic$\\to$Latin or because the controls were not run for DarijaBERT-mix.}","\\label{tab:contrasts}","\\centering\\footnotesize","\\setlength{\\tabcolsep}{4pt}",
   "\\begin{fittable}[\\textwidth]","\\begin{tabular}{@{}lll"+"ccc"*3+"@{}}","\\toprule",
   " & & & \\multicolumn{3}{c}{XLM-R} & \\multicolumn{3}{c}{DarijaBERT} & \\multicolumn{3}{c}{DarijaBERT-mix}\\\\","\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}",
   "& Contrast & Metric"+"".join(f" & \\shortstack{{{a}\\\\{b}}}" for _ in M for _,_,a,b in S)+"\\\\","\\midrule"]
for cid,name,form in C:
    for k,metric in enumerate(["F1","AUROC"]):
        lab=(f"{cid} & {name}" if k==0 else f" & \\quad {{\\scriptsize\\texttt{{{form}}}}}") if False else (f"{cid} & {name} & F1" if k==0 else f" & & AUROC")
        L.append(lab+" & "+" & ".join(cell(t,m,s,cid,metric) for m,t,s in cols)+"\\\\")
    L.append("\\addlinespace[2pt]")
L+=["\\bottomrule","\\end{tabular}\\end{fittable}","\\end{table*}"]
open(os.path.join(OUT,"contrasts.tex"),"w").write("\n".join(L))

# invariance table (XLM-R and DarijaBERT, MYC Ar->La & La->Ar)
IM=["vanilla","aug_det","aug_sample","det_kl","sb_noCon","augcon_det","sb_noKL","sb_det","scriptbridge","augcon_uroman","rdrop","charnoise","upper_dual"]
L=["\\begin{table}[t]","\\caption{Script invariance of the MYC models, measured on 1{,}000 human-written DoDa sentence pairs that have the same content in both scripts and no task labels. Agr. is the percentage of pairs whose two versions receive the same predicted class, and R@1 is the cross-script retrieval accuracy (\\%) with mean-pooled encoder embeddings. Values are means over five seeds.}","\\label{tab:invariance}","\\centering\\footnotesize","\\setlength{\\tabcolsep}{3pt}",
   "\\begin{fittable}[\\linewidth]","\\begin{tabular}{@{}lcccccccc@{}}","\\toprule"," & \\multicolumn{4}{c}{XLM-R} & \\multicolumn{4}{c}{DarijaBERT}\\\\","\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
   " & \\multicolumn{2}{c}{Ar$\\to$La} & \\multicolumn{2}{c}{La$\\to$Ar} & \\multicolumn{2}{c}{Ar$\\to$La} & \\multicolumn{2}{c}{La$\\to$Ar}\\\\",
   "Method & Agr. & R@1 & Agr. & R@1 & Agr. & R@1 & Agr. & R@1\\\\","\\midrule"]
for m in IM:
    row=[N[m].replace("$^{\\ddagger}$","")]
    for mid in ["xlm-roberta-base","SI2M-Lab/DarijaBERT"]:
        for src in ["arabic","latin"]:
            for col in ["doda_agree","r1_raw"]:
                v=g("myc",mid,src,m,col); row.append("--" if np.isnan(v) else f"{v:.1f}")
    L.append(" & ".join(row)+"\\\\")
L+=["\\bottomrule","\\end{tabular}\\end{fittable}","\\end{table}"]
open(os.path.join(OUT,"invariance.tex"),"w").write("\n".join(L))

# contaminated vs cleaned
CM=["vanilla","vanilla+TTN","aug_det","aug_sample","sb_det","scriptbridge","scriptbridge+TTN","charnoise","upper_dual"]
L=["\\begin{table}[t]","\\caption{Offensive Arabic$\\to$Latin before and after removing the leak by stripping Latin-script letters from Arabic-script items. Each cell gives cross-script macro-F1 / AUROC (\\%), averaged over five seeds on the original task and three seeds on the cleaned task. This is a separate diagnostic and is not one of the nine clean settings. Controls and the supervised reference were not run on the cleaned task.}","\\label{tab:clean}","\\centering\\footnotesize","\\setlength{\\tabcolsep}{3pt}",
   "\\begin{fittable}[\\linewidth]","\\begin{tabular}{@{}lcccc@{}}","\\toprule"," & \\multicolumn{2}{c}{Original (contaminated)} & \\multicolumn{2}{c}{Cleaned}\\\\","\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
   "Method & XLM-R & DarijaBERT & XLM-R & DarijaBERT\\\\","\\midrule"]
for m in CM:
    row=[N[m]]
    for task in ["offensive","offensive_clean"]:
        for mid in ["xlm-roberta-base","SI2M-Lab/DarijaBERT"]:
            f=g(task,mid,"arabic",m,"f1_cross"); a=g(task,mid,"arabic",m,"auc_cross")
            row.append("--" if np.isnan(f) else f"{f:.1f} / {a:.1f}")
    L.append(" & ".join(row)+"\\\\")
L+=["\\bottomrule","\\end{tabular}\\end{fittable}","\\end{table}"]
open(os.path.join(OUT,"clean.tex"),"w").write("\n".join(L))

# calibration under prior shift (offensive La->Ar)
CB=["vanilla","vanilla+TTN","aug_det","aug_sample","sb_det","scriptbridge","upper_dual"]
L=["\\begin{table}[t]","\\caption{Offensive Latin$\\to$Arabic under label-prior shift, with 75.6\\% offensive items in the Latin-script training data and 29.8\\% in the Arabic-script test data. Each cell gives XLM-R / DarijaBERT, averaged over five seeds. F1 is macro-F1 under the argmax decision rule, which for two classes equals a threshold of 0.5 on the predicted probability of the offensive class. F1$_{\\text{bal}}$ is macro-F1 on a class-balanced test subsample, BA is balanced accuracy and ECE is expected calibration error, for which lower is better. F1$_{\\text{orc}}$ uses a threshold tuned post hoc on the target test set and is a diagnostic comparison only, not a zero-shot result.}","\\label{tab:calib}","\\centering\\footnotesize","\\setlength{\\tabcolsep}{2.6pt}",
   "\\begin{fittable}[\\linewidth]","\\begin{tabular}{@{}lccccc@{}}","\\toprule","Method & F1 & F1$_{\\text{bal}}$ & BA & ECE & F1$_{\\text{orc}}$\\\\","\\midrule"]
for m in CB:
    row=[N[m]]
    for col in ["f1_cross","f1_cross_bal","bacc_cross","ece_cross","f1_cross_oracle"]:
        a=g("offensive","xlm-roberta-base","latin",m,col); b=g("offensive","SI2M-Lab/DarijaBERT","latin",m,col)
        row.append(f"{a:.1f} / {b:.1f}")
    L.append(" & ".join(row)+"\\\\")
L+=["\\bottomrule","\\end{tabular}\\end{fittable}","\\end{table}"]
open(os.path.join(OUT,"calib.tex"),"w").write("\n".join(L))
import glob, re
def transform(s):
    out=[]; pos=0
    pat=re.compile(r"\\begin\{(table\*?)\}(\[[^\]]*\])?(.*?)\\end\{\1\}", re.S)
    for m in pat.finditer(s):
        body=m.group(3)
        if "\\begin{fittable}" not in body: continue
        out.append(s[pos:m.start()])
        fm=re.search(r"\\begin\{fittable\}\[(\\\w+)\]\n?(.*?)\\end\{fittable\}", body, re.S)
        width, tab = fm.group(1), fm.group(2)
        pre=body[:fm.start()]
        # font/tabcolsep commands that precede the tabular inside the float
        fonts="".join(re.findall(r"\\(?:footnotesize|scriptsize|small|tiny)\b", pre))
        seps="".join(re.findall(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", pre))
        boxed="\\begin{lrbox}{\\tblbox}"+fonts+seps+"%\n"+tab.rstrip()+"%\n\\end{lrbox}\n"
        newbody=pre+"\\fitbox{"+width+"}"+body[fm.end():]
        out.append(boxed+"\\begin{"+m.group(1)+"}"+(m.group(2) or "")+newbody+"\\end{"+m.group(1)+"}")
        pos=m.end()
    out.append(s[pos:]); return "".join(out)
for f in glob.glob(os.path.join(OUT,"*.tex")):
    t=transform(open(f).read())
    t=re.sub(r"\\begin\{table\}",r"\\begin{table*}",t); t=re.sub(r"\\end\{table\}",r"\\end{table*}",t)
    open(f,"w").write(t)
print("ok")
