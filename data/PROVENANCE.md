# Dataset provenance — HCRL CAN Intrusion Dataset (OTIDS family, repackaged)

| Field | Value |
|---|---|
| Dataset name | CAN HCRL OTIDS (Kaggle repackaging of HCRL in-vehicle CAN intrusion captures) |
| Source | Kaggle dataset `bikashkundu/can-hcrl-otids`, derived from the Hacking and Countermeasure Research Lab (HCRL), Korea University |
| Access | `kagglehub.dataset_download("bikashkundu/can-hcrl-otids")` (public; no credentials required) |
| Origin | HCRL in-vehicle CAN-bus captures (OTIDS / Car-Hacking family), logged via the OBD-II port of a real vehicle |
| License / usage terms | HCRL datasets are released for academic/research use; redistribution subject to the original HCRL terms. State as unknown where not explicitly provided by the Kaggle mirror. |
| Download date (UTC) | 2026-10-23T17:45:12Z |
| Files used | dataset.csv (normal/attack-free), dataset1.csv / dataset2.csv / dataset3.csv (three attack captures) |
| Columns | TS, ID1, ID0, LEN, DLC0..DLC7, target (0 = normal, 1/2/3 = attack type) |
| Label used | Binary intrusion label: 0 (normal capture) vs 1 (any attack capture) |
| Frame features | [dt, CAN ID (ID1), DLC, 8 payload bytes] = 11 features per frame |
| Windowing | 100 consecutive frames (stride 50), time-contiguous split inside each capture (no leakage); window labelled by its capture's class |
| Trigger | A CAN ID verified absent from every capture (0x7FF), stamped into k=4 frames of a window — a genuinely rare, injectable pattern |

## Original / processed sizes
| File | Rows | SHA-256 |
|---|---|---|
| dataset.csv  | 2,369,398 | 30744d16c956df4da119924ccedbe5ca4b4f35b3e7d7597368c3268a49523e1e |
| dataset1.csv |   656,579 | a6367fc94b4e9d6fa9f59f015388e55cc0981d4839e7f5a524103d5e4ba058d9 |
| dataset2.csv |   591,990 | 288919641a97bbc215ef9c4d10b4d57e98e8877e37bcaadfe2194b65b24b53d2 |
| dataset3.csv |   995,472 | b23c3a6bf85ac31649768179a23d2d7757b0078bc2b138f83deffc0830478d63 |

For tractability the experiments use the first 500,000 normal frames and 150,000
frames from each attack capture (configurable in `IA2/scripts/run_all.py`);
every run is seeded.

## Known limitations
- A repackaged/relabelled mirror: the exact per-file attack type (DoS / fuzzing /
  impersonation) is not named in the mirror, so the task is framed as a **binary**
  normal-vs-attack intrusion detector, which is sufficient for the backdoor study.
- Attack captures are entirely attack-labelled (each is a separate capture), so the
  normal/attack balance is set by how many frames of each are included.
