# Backdooring the Guard
**Security and Privacy for Machine Learning (Honors) — Individual Research Project**
Aishwarya Nitin Gawali · Roll No. 16014223007 · Batch B1

A data-poisoning **backdoor attack on a CAN-bus intrusion-detection system**, and
a **layered privacy defence** (central DP-SGD, local DP, and compressive privacy)
studied for both its protective effect and its privacy–utility cost — all on the
HCRL Car-Hacking dataset.

## Layout
```
data/            place RPM_dataset.csv in data/raw/; PROVENANCE.md; scripts/hash_data.py
src/canids/      hcrl (parse+window), detector (CNN/RF), backdoor, localdp,
                 compress, defense, membership, metrics
tests/           pytest pipeline checks (windowing, trigger causality, LDP, ...)
IA1/, IA2/       preliminary + full study: scripts, results/, report/ (PDF), paper/
```

## Get the data
The HCRL Car-Hacking RPM file is distributed from the HCRL site
(https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) via a Google-Drive
folder that automated downloaders cannot enumerate. Repository: https://github.com/aish475/MLCS_IA

Download it manually and place
`RPM_dataset.csv` at `data/raw/`, then `python scripts/hash_data.py`.

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -q      # pipeline correctness (synthetic)
python IA2/scripts/run_all.py                  # full study on real data
python IA2/make_tables_ia2.py                  # tables + figures
```
`USE_SYNTHETIC=1` runs the pipeline on a synthetic HCRL-format sample for
verification only (those numbers are never reported).

## Modules covered
M1 backdoor/poisoning + membership inference · M2 DP-SGD + DP k-means · M3 local DP
on CAN features + LDP-NB · M4 output modification + k-anonymity (analytical) ·
M5 compressive PCA/DCA — all anchored to the same CAN-bus IDS.
