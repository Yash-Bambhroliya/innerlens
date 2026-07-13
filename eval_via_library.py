"""Honest AUROC through the ACTUAL library scoring (WorkspaceResult.confidence =
weakest entity-token internal-confidence). This is the number the README quotes."""
import re, string
import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from innerlens import InnerLens

il = InnerLens.load("Qwen/Qwen3.5-4B")


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation[:200]")
conf, meanconf, correct = [], [], []
for ex in ds:
    q = ex["question"]
    gold = [ex["answer"]["value"]] + list(ex["answer"].get("aliases", []))
    r = il.generate(f"{q}\nReply with only the answer, no explanation.", max_tokens=16)
    na = norm(r.text)
    ok = any(norm(g) and norm(g) in na for g in gold)
    conf.append(r.confidence); meanconf.append(r.mean_entity_confidence); correct.append(int(ok))

correct = np.array(correct)
print(f"n={len(correct)}  model accuracy={correct.mean():.1%}  "
      f"(correct={correct.sum()}, wrong={len(correct)-correct.sum()})")
if 0 < correct.sum() < len(correct):
    print(f"AUROC  confidence (weakest entity-token)  -> correct : {roc_auc_score(correct, conf):.3f}")
    print(f"AUROC  mean entity-token confidence       -> correct : {roc_auc_score(correct, meanconf):.3f}")
