"""Record size + SHA-256 of the placed HCRL file into the provenance log."""
import hashlib, os, sys
p = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "RPM_dataset.csv")
if not os.path.exists(p):
    sys.exit("RPM_dataset.csv not found at data/raw/ — place it first.")
h = hashlib.sha256()
with open(p, "rb") as f:
    for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
n = sum(1 for _ in open(p, "r", errors="ignore"))
print(f"file: {p}\nrows: {n}\nsize_bytes: {os.path.getsize(p)}\nsha256: {h.hexdigest()}")
