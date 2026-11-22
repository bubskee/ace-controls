"""Print the held-out two-site comparison from the five archived GPU runs."""
import json
from pathlib import Path

results = Path(__file__).resolve().parents[1] / "results" / "ioi"
models = ("gpt2", "gpt2-medium", "gpt2-large", "llama-3.1-8b", "gemma-2-9b")
methods = ("ACE", "pair-difference PCA", "PCA, unpaired (paper)", "gradient only")

print("Held-out joint removal AUC at the published S2 and END ranks")
print(f"{'Model':<17} {'Ranks':>7} {'ACE':>7} {'Pair PCA':>9} {'PCA':>7} {'Grad':>7}")
for model in models:
    data = json.loads((results / f"ioi_ladder_{model}.json").read_text())
    ranks = "+".join(map(str, data["ranks"]))
    scores = [data["rows"][method]["held-out"]["remove"] for method in methods]
    print(f"{model:<17} {ranks:>7} {scores[0]:7.3f} {scores[1]:9.3f} {scores[2]:7.3f} {scores[3]:7.3f}")
