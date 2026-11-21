"""Check results/claims.json: recompute every headline number from its source file, check each published quote
still carries those numbers verbatim, and scan the docs for stale wording.

python3 -I scripts/check_claims.py [ROOT]   prints each claim, the files it is bound to, and any drift or stale hit; exit 1 on either.
"""
import json
import re
import statistics
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
    if kind == "fixed":
        return f"{value:.{int(n)}f}"
    if kind == "pct":
        return f"{100 * value:.{int(n)}f}"
    if kind == "comma":
        return f"{value:,.{int(n)}f}" if n else f"{value:,}"
    if kind == "word":
        return value.split()[0]
    if kind == "md":  # a pipe inside a Markdown table cell is written \|
        return str(value).replace("|", "\\|")
    raise ValueError(f"unknown rule {rule}")


def compute(claim, src):
    """name -> (raw value from the source, rendered by its rule); `label` is the source's own label."""
    out = {"label": (lookup(src, claim["label_path"]),) * 2}
    for v in claim["values"]:
        if "ratio" in v:
            a, b = (lookup(src, p) for p in v["ratio"])
            raw = a / b
        elif "product" in v:
            a, b = (lookup(src, p) for p in v["product"])
            raw = a * b
        elif "median_ratio" in v:  # per group, the median over its items of num/den; then the min or max of those medians
            m, groups = v["median_ratio"], {}
            for run in lookup(src, m["over"]):
                for item in lookup(run, m["items"]):
                    groups.setdefault(run[m["group"]], []).append(lookup(item, m["num"]) / lookup(item, m["den"]))
            raw = {"min": min, "max": max}[m["pick"]](statistics.median(x) for x in groups.values())
        elif "distinct" in v:  # how many distinct values a key takes across a list
            items, key = v["distinct"]
            raw = len({x[key] for x in lookup(src, items)})
        elif "len" in v:
            raw = len(lookup(src, v["len"]))
        else:
            raw = lookup(src, v["path"])
        out[v["name"]] = (raw, render(raw, v["rule"]))
    return out


def drift(root=ROOT):
    """Every (claim id, what, expected, got): a value that no longer matches its source file,
    or a published quote that no longer carries the value it is bound to."""
    ledger = json.loads((root / LEDGER).read_text())
    out = []
    for c in ledger["claims"]:
        vals = compute(c, json.loads((root / c["source"]).read_text()))
        if (got := str(vals["label"][0])) != c["label"]:
            out.append((c["id"], c["label_path"], c["label"], got))
        for v in c["values"]:
            if (got := vals[v["name"]][1]) != v["expect"]:
                out.append((c["id"], v["name"], v["expect"], got))
        for q in c.get("quotes", []):
            doc = (root / q["file"]).read_text(encoding="utf-8")
            if "text" in q:
                want = q["text"].format_map({k: r for k, (_, r) in vals.items()})
                if want not in doc:
                    out.append((c["id"], q["file"], want, "not found verbatim"))
            else:
                raw, shown = vals[q["value"]]
                got = lookup(json.loads(doc), q["path"])
                if got != raw and str(got) != shown:
                    out.append((c["id"], f"{q['file']}:{q['path']}", shown, str(got)))
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


def main(argv):
    root = Path(argv[1]).resolve() if len(argv) > 1 else ROOT
    for c in json.loads((root / LEDGER).read_text())["claims"]:
        print(f"{c['id']}: bound to {', '.join(sorted({q['file'] for q in c['quotes']})) or 'no published text yet'}")
    bad = drift(root)
    for cid, what, want, got in bad:
        print(f"DRIFT {cid}: {what}: ledger expects {want!r}, found {got!r}")
    old = stale(root)
    for where, pattern, why in old:
        print(f"STALE {where}: /{pattern}/ ({why})")
    print("ok" if not bad and not old else f"{len(bad)} drifted, {len(old)} stale")
    return 1 if bad or old else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
