"""Check results/claims.json: recompute every headline number from its source file, and scan the docs for stale wording.

python3 -I scripts/check_claims.py [ROOT]   prints each claim, where the docs quote it, and any drift or stale hit; exit 1 on either.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = "results/claims.json"
DOCS = ("README.md", "docs/JUDGING.md", "docs/SUBMISSION.md", "demo/index.html", "demo/*.js")


def lookup(data, path):
    """Dotted path; `[i]` indexes a list, `[key=value]` picks the one list item whose key equals value."""
    for part in path.split("."):
        name, *subs = re.findall(r"[^\[\]]+", part)
        data = data[name]
        for sub in subs:
            if "=" in sub:
                key, want = sub.split("=", 1)
                hits = [x for x in data if str(x.get(key)) == want]
                if len(hits) != 1:
                    raise KeyError(f"{path}: {len(hits)} items with {key}={want}")
                data = hits[0]
            else:
                data = data[int(sub)]
    return data


def render(value, rule):
    kind, _, n = rule.partition(":")
    if kind == "eq":
        return str(value)
    if kind == "fixed" or kind == "ratio":
        return f"{value:.{int(n)}f}"
    if kind == "pct":
        return f"{100 * value:.{int(n)}f}"
    raise ValueError(f"unknown rule {rule}")


def drift(root=ROOT):
    """Every (claim id, what, expected, got) that no longer matches its source file."""
    ledger = json.loads((root / LEDGER).read_text())
    out = []
    for c in ledger["claims"]:
        src = json.loads((root / c["source"]).read_text())
        if (got := str(lookup(src, c["label_path"]))) != c["label"]:
            out.append((c["id"], c["label_path"], c["label"], got))
        for v in c["values"]:
            if "ratio" in v:
                a, b = (lookup(src, p) for p in v["ratio"])
                got, what = render(a / b, v["rule"]), "/".join(v["ratio"])
            else:
                got, what = render(lookup(src, v["path"]), v["rule"]), v["path"]
            if got != v["expect"]:
                out.append((c["id"], what, v["expect"], got))
    return out


def doc_files(root=ROOT):
    return [p for pat in DOCS for p in sorted(root.glob(pat))]


def stale(root=ROOT):
    """Every (file:line, pattern, why) where the docs or demo page still carry a superseded value or wording."""
    ledger = json.loads((root / LEDGER).read_text())
    out = []
    for p in doc_files(root):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for s in ledger["stale"]:
                if re.search(s["pattern"], line):
                    out.append((f"{p.relative_to(root)}:{i}", s["pattern"], s["why"]))
    return out


def quoted(root=ROOT):
    """claim id -> the doc files that quote its `quoted_as` text (a report, not a check)."""
    ledger = json.loads((root / LEDGER).read_text())
    texts = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in doc_files(root)}
    return {c["id"]: [f for f, t in texts.items() if c["quoted_as"] in t] for c in ledger["claims"]}


def main(argv):
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    for cid, files in quoted(root).items():
        print(f"{cid}: quoted in {', '.join(files) or 'no doc yet'}")
    bad = drift(root)
    for cid, what, want, got in bad:
        print(f"DRIFT {cid}: {what} is {got} in the source, the ledger says {want}")
    old = stale(root)
    for where, pattern, why in old:
        print(f"STALE {where}: /{pattern}/ ({why})")
    print("ok" if not bad and not old else f"{len(bad)} drifted, {len(old)} stale")
    return 1 if bad or old else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
