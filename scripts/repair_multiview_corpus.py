"""Deterministic repair of the multiview corpus (no model calls).

Three passes per item/view:
  (a) strip stray control characters from assembled + granular files
  (b) strict outline repair: keep only the schema's required keys per entry,
      dropping extra keys (e.g. "tags") and empty/degenerate entries
  (c) outline reconstruction: when (b) leaves entry-count != granular-count (a
      corrupted outline), rebuild the outline with one derived entry per granular
      chapter so it is schema-valid and count-matched. Content is left untouched.

Validates each touched view with utils.multiview_recovery.validate_view and reports
before/after. Run with --apply to write; default is dry-run.
"""
import glob, json, os, re, sys
sys.path.insert(0, "/vast/projects/myatskar/design-documents/joseph/fine-tuning-or-retrieval")
from utils.multiview_recovery import validate_view, GRANULAR_LAYOUTS

APPLY = "--apply" in sys.argv
ASSEMBLED = {"stackexchange": "stackexchange.txt", "textbook": "textbook.txt", "blog": "blogs.txt"}
OUTLINES = {"stackexchange": "stack_exchange_outline.json", "textbook": "textbook_outline.json", "blog": "blog_outline.json"}

def strip_ctrl(text):
    return "".join(c for c in text if ord(c) >= 32 or c in "\n\r\t")

def outline_spec(view, domain):
    """(top-level key, [required entry keys]) for this view+domain."""
    if view == "textbook":
        return ("sections", ["section_title", "description", "subtopics"]) if domain == "medical" \
               else ("outline", ["chapter_title", "description", "subtopics"])
    if view == "blog":
        return ("blogs", ["title", "description"])
    return ("questions", ["question", "category"]) if domain == "medical" \
           else ("questions", ["title", "question_body"])

def _nonempty(v):
    return (isinstance(v, str) and v.strip()) or (isinstance(v, list) and v and all(isinstance(x, str) and x.strip() for x in v))

def strict_entries(data, key, req):
    out = []
    for e in (data.get(key, []) if isinstance(data, dict) else []):
        if isinstance(e, dict) and all(k in e and _nonempty(e[k]) for k in req):
            out.append({k: e[k] for k in req})
    return out

def _strip_preamble(text):
    body = text.strip()
    for _ in range(3):
        if "\n\n" in body and body.split("\n\n", 1)[0].lstrip().startswith(("\\title", "Title:")):
            body = body.split("\n\n", 1)[1].strip(); continue
        m = re.match(r"(?:Chapter|Section)\s+\d+\s*:\s*", body)
        if m: body = body[m.end():].strip(); continue
        break
    return body

def derive_entry(granular_text, req):
    """Build a schema-valid entry from a granular chapter/blog/answer."""
    body = _strip_preamble(granular_text)
    lines = [l.rstrip() for l in body.splitlines()]
    nonempty = [l for l in lines if l.strip()]
    title = re.sub(r"^#+\s*", "", nonempty[0]).strip() if nonempty else "Section"
    title = title or "Section"
    # description: first substantive non-heading paragraph line, else the title
    desc = next((l.strip() for l in nonempty[1:] if not l.lstrip().startswith("#") and len(l.strip()) >= 20), "")
    desc = (desc or title)[:800]
    # subtopics: markdown subheadings inside the chapter, else a single fallback
    subs = [re.sub(r"^#+\s*", "", l).strip() for l in nonempty if l.lstrip().startswith("#")]
    subs = [s for s in subs if s][1:] or [s for s in subs if s]  # drop the title heading if reused
    subs = [s for s in dict.fromkeys(subs) if s] or ["Overview"]
    title_key = req[0]  # chapter_title / section_title / title / question
    entry = {title_key: title}
    if "description" in req: entry["description"] = desc
    if "subtopics" in req: entry["subtopics"] = subs
    if "question_body" in req: entry["question_body"] = desc
    if "category" in req: entry["category"] = "General"
    return entry

def repair(item_dir, view, domain):
    key, req = outline_spec(view, domain)
    gdir, _ = GRANULAR_LAYOUTS[view]
    granular = sorted(glob.glob(f"{item_dir}/{gdir}/*.txt"))
    changed = []
    # (a) strip control chars from assembled + granular
    for p in [f"{item_dir}/{ASSEMBLED[view]}"] + granular:
        if not os.path.exists(p): continue
        t = open(p, encoding="utf-8", errors="replace").read()
        s = strip_ctrl(t)
        if s != t:
            changed.append(f"strip:{os.path.basename(p)}")
            if APPLY: open(p, "w", encoding="utf-8").write(s)
    # (b)+(c) outline
    opath = f"{item_dir}/{OUTLINES[view]}"
    if os.path.exists(opath):
        try: data = json.loads(open(opath, encoding="utf-8", errors="replace").read())
        except Exception: data = {}
        ents = strict_entries(data, key, req)
        if len(ents) != len(granular) or not ents:
            # (c) reconstruct one entry per granular unit from its text
            ents = [derive_entry(open(g, encoding="utf-8", errors="replace").read(), req) for g in granular]
            changed.append(f"reconstruct_outline:{len(ents)}")
        elif any(set(e) != set(req) for e in (data.get(key, []) if isinstance(data, dict) else [])) \
             or (isinstance(data, dict) and set(data) != {key}):
            changed.append("strict_outline")
        new = {key: ents}
        if APPLY: json.dump(new, open(opath, "w"), indent=2)
    return changed

from collections import Counter
tally = Counter(); after_valid = after_invalid = 0; still_bad = []
for dom in ["arxiv", "medical", "legal"]:
    for base in glob.glob(f"data/{dom}/explanations/qwen3_5_*_w16"):
        for item in glob.glob(f"{base}/*/"):
            for view in ("stackexchange", "textbook", "blog"):
                if not os.path.exists(f"{item}{OUTLINES[view]}") and not os.path.exists(f"{item}{ASSEMBLED[view]}"):
                    continue
                ch = repair(item, view, dom)
                for c in ch: tally[c.split(":")[0]] += 1
                if APPLY:
                    r = validate_view(item, view)
                    if r["valid"]: after_valid += 1
                    else: after_invalid += 1; still_bad.append((f"{os.path.relpath(item,'data')}{view}", r["reasons"][:2]))
print("MODE:", "APPLY" if APPLY else "DRY-RUN")
print("change tally:", dict(tally))
if APPLY:
    print(f"AFTER repair: valid={after_valid} invalid={after_invalid}")
    print("still invalid:")
    for v, r in still_bad[:40]: print("   ", v, "->", r)
