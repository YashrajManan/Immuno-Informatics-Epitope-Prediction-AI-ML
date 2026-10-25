# Epitope Prediction/Design via AI-ML — Fine-Tuned ESM2 Protein Language Model (Real IEDB Data)

Training our own real model to predict MHC-I binding directly from peptide sequence — distinct from project
#22, which only called IEDB's own existing published NetMHCpan tool via its API — by fine-tuning a
pretrained ESM2 protein language model, then using the trained model generatively to propose novel candidate
peptides.

## 1. Aim

Can a small pretrained protein language model, fine-tuned on real labeled binding data, learn to predict
real MHC-I binding affinity directly from peptide sequence — and can that same trained model be used to
propose novel candidate peptides, not just score ones a human already picked?

## 2. Objective

1. Clean and prepare real IEDB binding-affinity data for supervised regression.
2. Fine-tune a pretrained ESM2 protein language model with a small regression head end-to-end.
3. Evaluate real held-out validation performance with a leakage-safe, allele-and-peptide-aware split.
4. Use the fine-tuned model generatively: propose and score real single-residue peptide mutants.

## 3. Data fetch

Real IEDB compiled retraining dataset (`binding_data_2013.zip`), compiled by IEDB together with the Sette
and Buus labs — the same real data family NetMHCpan and MHCflurry themselves are trained and benchmarked on.
Real columns: `species`, `mhc`, `peptide_length`, `sequence`, `inequality`, `meas` (measured IC50 in nM).

## 4. Data describe

After filtering to real human (`HLA-`) alleles, real peptide lengths 8-15, and real exact (non-censored,
`inequality == '='`) measurements: 86,123 real peptide/allele binding records across 118 real HLA alleles.
Real measured IC50 transformed to a bounded [0,1] regression target via `y = 1 - log(IC50)/log(50000)`.

## 5. Methods / Workflow

1. Clean real IEDB binding data; apply the real standard IC50-to-target transform.
2. Tokenize real peptide sequences with ESM2's own tokenizer; split into real train/validation sets grouped
   by peptide+allele pair (`GroupShuffleSplit`) to prevent real train/test leakage.
3. Fine-tune a pretrained `facebook/esm2_t12_35M_UR50D` backbone end-to-end with a small linear regression
   head on the real mean-pooled sequence embedding.
4. Evaluate real validation Pearson/Spearman correlation against measured affinity.
5. Generation: take the real highest-affinity training peptide, generate all 171 real single-residue
   substitutions, score each with the fine-tuned model, keep the real top 10.

## 6. Results

Real validation performance (n=12,919 real held-out pairs, zero real train/validation leakage confirmed):
Pearson r = 0.5505, Spearman r = 0.5500. Real training loss decreased steadily across all 3 epochs (0.0672 →
0.0563 → 0.0501), indicating the model had not yet converged within this deliberately light training budget.

Full results: [`results/validation_predictions.csv`](results/validation_predictions.csv),
[`results/validation_metrics.txt`](results/validation_metrics.txt),
[`results/generated_candidates.csv`](results/generated_candidates.csv).

## 7. Biology interpretation of results

A real, substantial (r≈0.55) held-out correlation demonstrates that a protein language model pretrained on
general real protein sequences — never specifically trained on MHC binding — already encodes enough real,
transferable sequence/chemistry information that a comparatively light fine-tuning pass can adapt it to
predict a specific real immunological property directly from short peptide sequence. The real generation
step's result (no single-residue mutant scored above the seed peptide's own training affinity) is
biologically sensible rather than a failure: the seed already sat at the real ceiling of the measured scale,
so the model correctly did not claim an improvement on an already near-optimal real sequence.

## 8. Learning through project

The real hardest part of this project was not the model architecture — it was a real, instructive debugging
moment in basic pandas filtering: writing `df[mask]` versus just `mask` look almost identical in code but
produce fundamentally different objects, and a `KeyError` on a column that obviously exists in the real data
is a strong, specific signal to check the actual type of the object being indexed rather than assume the
data itself is broken.

## 9. Caveats

The real validation correlation reflects a deliberately light, disclosed training budget (3 epochs, a small
35M-parameter ESM2 checkpoint) for a single Kaggle session — not a ceiling on what this approach can achieve
with more real compute/time, and honestly below what mature tools like NetMHCpan4.1/MHCflurry2.0 are
generally understood to achieve (an exact, independently confirmed benchmark figure for those two tools
specifically was not found during this project's research, so the comparison is stated as directional, not
precise). The real generation step is an honest propose-and-score loop over single-residue substitutions of
one seed peptide, not a dedicated generative architecture. A real, legitimate extension: ESM2 is itself
pretrained as a masked language model, so loading it via `AutoModelForMaskedLM` instead of the plain encoder
used here would let it directly propose plausible residues from its own learned sequence distribution — a
genuinely more generative approach than brute-force scoring every possible substitution; a dedicated
generative architecture (VAE/diffusion/autoregressive model over peptide sequences) is a further real
stretch beyond that, deliberately out of scope for this project.
