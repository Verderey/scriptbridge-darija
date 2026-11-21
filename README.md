# ScriptBridge

Code and results for the paper **Benchmark shortcuts and transliteration consistency in zero-shot cross-script classification of Moroccan Darija**.

Soufiane Hourri<sup>1</sup>, Abdelhay Zoizou<sup>2</sup>, Ilham El Ouariachi<sup>3</sup>, Brahim Ait Skourt<sup>4</sup>, Moad Hicham Safhi<sup>5</sup> and Mohammed Bergui<sup>6</sup>

<sup>1</sup> Laboratory of Process, Industrial Signals and Computer Science, Cadi Ayyad University, Safi, Morocco  
<sup>2</sup> Laboratory of Sciences, Engineering and Management, Higher School of Technology, Sidi Mohamed Ben Abdellah University, Fez, Morocco  
<sup>3</sup> Computer Science Department, Faculty of Sciences, Moulay Ismail University, Meknes, Morocco  
<sup>4</sup> SPM Research Team, ENSIAS, Mohammed V University, Rabat, Morocco  
<sup>5</sup> C3S Laboratory, Higher School of Technology, Hassan II University, Casablanca, Morocco  
<sup>6</sup> Laboratory of Intelligent Systems and Applications, Sidi Mohamed Ben Abdellah University, Fez, Morocco

Corresponding author: Soufiane Hourri, s.hourri@uca.ac.ma

## About the project

Moroccan Darija is written online in Arabic script and in a Latin-script form called Arabizi, which has no standard orthography. Labelled data rarely cover both scripts. This repository studies zero-shot cross-script classification, where a model is trained and selected on labelled text in one script and tested on naturally written text in the other.

The pipeline is fully automatic and has four parts.

1. A model-free audit checks each benchmark for cues that are identical in both scripts. It found that MYC sentiment labels can be predicted across scripts from emojis alone, and that one direction of an offensive-language dataset is predominantly explained by a single Latin-script lexical cue inside Arabic-script items.
2. A stochastic Arabic to Arabizi transliterator is learned from the public DoDa parallel corpus by hard-EM alignment and evaluated against uroman, a ByT5-small model trained on the same pairs, and a multi-reference lexicon of human spellings.
3. Encoders (XLM-R, DarijaBERT and DarijaBERT-mix) are fine-tuned on labelled examples together with transliterated views, in a full 2×2×2 factorial design over sampling, a KL consistency term and an InfoNCE alignment term.
4. Six contrasts declared in advance are tested with a hierarchical bootstrap and Holm correction, and script invariance is measured on human-written parallel sentences.

## Repository layout

```
notebooks/ScriptBridge_Darija_CrossScript.ipynb   complete pipeline, from data download to tables and figures
results/main_results.csv                          per-system means and standard deviations over seeds
results/contrasts.csv                             primary and secondary contrasts with bootstrap intervals and Holm-adjusted p-values
results/factorial_effects.csv                     main effects of sampling, KL and InfoNCE
results/full/                                     complete outputs of the run reported in the paper
requirements.txt                                  library versions used for the paper
CITATION.cff                                      citation metadata
```

The three files directly in `results/` contain the aggregated values reported in the paper. The folder `results/full/` contains the complete outputs written by the notebook during the run reported in the paper.

| Files in `results/full/` | Content |
|---|---|
| `results_v5_<task>_noemoji_<model>.csv` | metrics of every run (method, direction, seed), with training time and peak GPU memory |
| `results_v5_<task>_noemoji_<model>_preds.pkl` | predicted class probabilities on the target test set, keyed by (method, source script, seed); load with Python's `pickle` |
| `results_v5_<task>_noemoji_<model>_lambdas.json` | loss weights selected on the source development set |
| `significance_v5.csv`, `factorial_effects.csv`, `retrieval_vs_f1.csv` | statistical tests, factorial main effects and the retrieval and F1 correlation |
| `audit.csv`, `audit_emoji_by_class.csv` | benchmark audit |
| `error_slices_<task>_<model>.csv` | error analysis on slices of the target test set |
| `split_manifest_<task>.csv` | split of every item as a text hash with its label, to rebuild the exact splits from the original data |
| `transliterator_params.json` | learned transliterator parameters |
| `tokenizer_coverage.csv`, `environment.json` | tokenizer check and library versions |

The prediction files follow the row order of the target test split, which the split manifests reproduce. The audit file listing the most influential n-grams is not included, because it contains offensive words from the dataset; the notebook regenerates it.

## Running the notebook

Open `notebooks/ScriptBridge_Darija_CrossScript.ipynb` in Google Colab, select a GPU runtime (an A100 is recommended, an L4 also works but is slower) and run all cells. The datasets are downloaded automatically from their original sources.

Set `QUICK_RUN=True` in the configuration cell for an end-to-end check on a subsample, which takes about ten minutes. The full run trains every configuration with several seeds and takes many hours of GPU time. Set `SAVE_TO_DRIVE=True` to write results to Google Drive. Results are saved after each run, so an interrupted session can continue from where it stopped.

## Data

| Dataset | Role | Source |
|---|---|---|
| MYC, Moroccan YouTube Corpus | sentiment task in both scripts | github.com/MouadJb/MYC |
| Moroccan Darija Offensive Language, version 3 | offensive-language task in both scripts | doi.org/10.17632/2y4m97b7dc.3 |
| DoDa, Darija Open Dataset | transliterator training and evaluation, temperature selection, invariance measurement | github.com/darija-open-dataset/dataset |

No dataset text is redistributed here. The split manifests written by the notebook contain only text hashes and labels, so the exact splits can be rebuilt from the original data. Please follow the licence terms of each dataset.

The offensive-language dataset contains vulgar and offensive content. The notebook prints examples and influential n-grams during the audit, so take care before sharing executed copies of it.

## Citation

If you use this code or its results, please cite the paper. Citation metadata is in `CITATION.cff`, and the full reference will be added here once the paper is published.

## Licence

The code is released under the MIT licence. See `LICENSE`.
