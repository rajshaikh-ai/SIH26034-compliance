"""Explainable rule engine: extracts the mandatory declarations from OCR lines
with OCR-tolerant regular expressions and checks them against the Legal
Metrology (Packaged Commodities) Rules. Used both as a baseline classifier
and as a source of features for the ML models."""
from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

from src import config as C

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
UNIT_ALT = r"(fl\.?\s*oz|kgs|kg|gms|gm|grams?|mg|g|ml|ltrs?|litres?|liters?|l|oz|lbs?|cc)"
QTY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*" + UNIT_ALT + r"(?![a-z])", re.I)
STANDARD_UNITS = {"g": 1, "kg": 1000, "mg": 0.001, "ml": 1, "l": 1000,
                  "gram": 1, "grams": 1, "litre": 1000, "litres": 1000, "liter": 1000, "liters": 1000}
NONSTANDARD_UNITS = {"gm": 1, "gms": 1, "kgs": 1000, "oz": 28.35, "lb": 453.6, "lbs": 453.6, "cc": 1,
                     "floz": 29.57, "ltr": 1000, "ltrs": 1000}

NETQTY_KW = re.compile(r"\bnet\s*\.?\s*(wt|weight|qty|quantity|vol|volume|content|contents|mass)\b", re.I)
MRP_KW = re.compile(r"\b(m\s*\.?\s*r\s*\.?\s*p|max(imum)?\.?\s*retail\s*price)\b", re.I)
PRICE_RE = re.compile(r"(rs\.?|₹|inr|r5|z)\s*[:.]?\s*(\d+(?:[.,]\d{1,2})?)", re.I)
# "Incl. of all taxes" - OCR often garbles "Incl." ("Ind.", "lncl", "Inc1"), so any
# word starting with "in" before "(of) all tax" counts; "excl..." never does.
TAX_RE = re.compile(r"(?<![a-z])[il1|]n[a-z0-9]{0,8}\.?\s*(of\s*)?all\s*tax", re.I)
MFG_KW = re.compile(r"\b(mfg|mfd|manufactur\w*|pkd|packed|packing|date\s*of\s*(mfg|manufacture|packing))\b", re.I)
MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*[-/']?\s*\d{2,4}"
DATE_RE = re.compile(r"(\b\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}\b|\b\d{1,2}\s*[/.\-]\s*\d{4}\b"
                     r"|\d{1,2}\s*-?\s*" + MONTH_RE + r"|\b" + MONTH_RE + r")", re.I)  # also 18OCT26 / 18-Oct-2026
MFG_DATE_KW = re.compile(r"((mfg|mfd|manufactur\w*|pkd|packed|packing)\s*\.?\s*(date|on|dt)|date\s*of\s*(mfg|manufacture|packing|pkg))", re.I)
IMPORT_RE = re.compile(r"\bimport(ed|er)\b", re.I)
NUTRITION_RE = re.compile(r"(p[ae]r\s*\d|p[ae]r\s*s[ae]rve|kcal|energy|protein|fat|carbo|sugar|sodium|\(g\)|usp|unit\s*sale)", re.I)
BEST_RE = re.compile(r"(best\s*before|use\s*by|expiry|exp\.?\s*date|\bexp\b)", re.I)
MANUF_KW = re.compile(r"(\b(mfd|manu\w*|mfg|mkd|marketed|packed|pkd|imported)\s*\.?\s*(&\s*\w+\.?\s*)*(by|for)\b"
                      r"|\bregd\.?\s*office)", re.I)
PIN_RE = re.compile(r"\b\d{3}\s?\d{3}\b")
CARE_KW = re.compile(r"(consumer\s*(care|complaint|feedback)|customer\s*care|complaints?|toll\s*free|helpline)", re.I)
PHONE_RE = re.compile(r"(1800[\s\-]*\d{3}[\s\-]*\d{4}|\+91[\s\-]*\d{5}[\s\-]*\d{5}|\b\d{10}\b)")
EMAIL_RE = re.compile(r"[\w.\-]+\s?@\s?[\w\-]+\.[\w.]+")
ORIGIN_RE = re.compile(r"(country\s*of\s*origin|made\s*in\s+[a-z]+|product\s*of\s+[a-z]+|origin\s*:)", re.I)

FUZZY_KEYWORDS = {
    "manufacturer": ["manufactured by", "mfd by", "marketed by", "packed by", "imported by"],
    "net_qty": ["net wt", "net weight", "net qty", "net quantity", "net vol", "net content"],
    "mrp": ["mrp", "m.r.p.", "maximum retail price"],
    "mrp_tax": ["inclusive of all taxes", "incl of all taxes"],
    "mfg_date": ["mfg date", "date of mfg", "packed on", "pkd on", "date of manufacture"],
    "best_before": ["best before", "use by", "expiry date"],
    "consumer_care": ["consumer care", "customer care", "consumer complaints"],
    "origin": ["country of origin", "made in india"],
}
FIELDS = list(FUZZY_KEYWORDS)


def _same_column(a: dict, b: dict) -> bool:
    return min(a.get("x1", 1e9), b.get("x1", 1e9)) - max(a.get("x0", 0), b.get("x0", 0)) > 0


def _window(lines: list[dict], i: int, n: int, above: int = 0) -> str:
    """Text of line i plus up to n-1 lines below (and `above` lines above) it
    in the SAME column, so a neighbouring panel of a photo is never mixed in."""
    a = lines[i]
    col = [j for j in range(len(lines)) if j == i or _same_column(a, lines[j])]
    k = col.index(i)
    return " ".join(lines[j]["text"] for j in col[max(0, k - above):k + n])


def _nearest(lines: list[dict], i: int, pred) -> int | None:
    """Index of the line closest to line i (favouring the same row, i.e. the
    value column of a 'KEY : value' table) whose text satisfies pred."""
    a = lines[i]
    h = max(a["height"], 1)
    ay, ax = (a["y0"] + a["y1"]) / 2, a.get("x1", 0)
    best, best_d = None, float("inf")
    for j, b in enumerate(lines):
        if j == i or not pred(b["text"]):
            continue
        by = (b["y0"] + b["y1"]) / 2
        dy, dx = abs(by - ay) / h, (b.get("x0", 0) - ax) / h
        # the value sits on (about) the same row, to the right, or just below
        if dy > 2.5 or dx < -2 or dx > 25:
            continue
        d = dy + 0.1 * abs(dx)
        if d < best_d:
            best, best_d = j, d
    return best


def fuzzy_score(tokens: list[str], keyword: str) -> float:
    n = len(keyword.split())
    best = 0.0
    for i in range(max(1, len(tokens) - n + 1)):
        cand = " ".join(tokens[i:i + n])
        if abs(len(cand) - len(keyword)) > len(keyword):
            continue
        best = max(best, SequenceMatcher(None, cand, keyword).ratio())
        if best == 1.0:
            break
    return best


OCR_UNIT_FIXES = [(re.compile(r"(\d)\s?kq\b", re.I), r"\1 kg"), (re.compile(r"(\d)\s?(rnl|mI|m1)\b"), r"\1 ml"),
                  (re.compile(r"(\d)\s?9m\b", re.I), r"\1 gm")]


def parse_quantity(text: str, ocr_fix: bool = False) -> dict | None:
    """Parse '<number> <unit>'. With ocr_fix=True (used right after a NET WT.
    keyword) common OCR confusions are repaired, e.g. '50g' read as '509'."""
    if ocr_fix:
        for rx, rep in OCR_UNIT_FIXES:
            text = rx.sub(rep, text)
    m = QTY_RE.search(text)
    if not m and ocr_fix:
        m9 = re.search(r"(?<![\d.])(\d{1,4})\s?9\b(?!\s*(%|kcal))", text)
        if m9:
            v = float(m9.group(1))
            return dict(value=v, unit="g", standard=True, base=v, ocr_corrected=True)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    unit_raw = m.group(2)
    unit = re.sub(r"[\s.]", "", unit_raw).lower()
    if unit in STANDARD_UNITS:
        return dict(value=value, unit=unit_raw, standard=True, base=value * STANDARD_UNITS[unit])
    factor = NONSTANDARD_UNITS.get(unit, 1)
    return dict(value=value, unit=unit_raw, standard=False, base=value * factor)


def _netqty_entry(lines: list[dict], i: int, q: dict, px_per_mm: float) -> dict:
    h = lines[i]["height"]
    numeral_mm = h * C.NUMERAL_TO_BOX_RATIO / px_per_mm
    return dict(found=True, value=lines[i]["text"], line=i, qty=q, numeral_mm=numeral_mm,
                required_mm=C.required_numeral_mm(q["base"]))


def check_label(lines: list[dict], px_per_mm: float, check_font: bool = True) -> dict:
    """Run all compliance checks. `lines` are OCR lines (text, height in
    original-image pixels). Returns per-field results and violation list.

    Handles both inline declarations ("Net Wt. 500 g") and the two-column
    table layout common on real packs ("NET WT. :" ... "50g" in another column).
    `check_font=False` skips the Rule 7 numeral-height check (e.g. for phone
    photos whose physical scale is unknown)."""
    # The rupee sign is often recognised as a random non-Latin glyph glued to
    # the previous word ("MRPぎ"): replace any such character by " ₹ ".
    lines = [{**l, "text": re.sub(r"[^\x00-\x7F₹]+", " ₹ ", l["text"]).strip()} for l in lines]
    texts = [l["text"] for l in lines]
    full = "\n".join(texts)
    tokens = re.findall(r"[a-z0-9.]+", full.lower())
    R: dict = {f: dict(found=False, value="", line=None) for f in FIELDS}

    unreadable: list[str] = []  # keyword visible but its value could not be read
    for i, t in enumerate(texts):
        # Manufacturer: keyword + PIN code within the next lines of the same column
        if not R["manufacturer"]["found"] and MANUF_KW.search(t):
            win = _window(lines, i, 4)
            if PIN_RE.search(win):
                R["manufacturer"] = dict(found=True, value=win[:160], line=i)
        # Net quantity
        if not R["net_qty"]["found"] and NETQTY_KW.search(t):
            q = parse_quantity(t[NETQTY_KW.search(t).end():], ocr_fix=True)
            if q:
                R["net_qty"] = _netqty_entry(lines, i, q, px_per_mm)
            else:
                R["net_qty"]["kw_line"] = i
        # MRP + tax clause
        if not R["mrp"]["found"] and MRP_KW.search(t):
            rest = t[MRP_KW.search(t).end():]
            pm = PRICE_RE.search(rest) or re.search(r"(\d+(?:[.,]\d{1,2})?)", rest)
            price = float(pm.group(pm.lastindex).replace(",", ".")) if pm else float("nan")
            R["mrp"] = dict(found=True, value=t, line=i, price=price)
            win = _window(lines, i, 3, above=1)
            if TAX_RE.search(win):
                R["mrp_tax"] = dict(found=True, value=TAX_RE.search(win).group(0), line=i)
        # Manufacturing / packing date
        if not R["mfg_date"]["found"] and MFG_KW.search(t) and not MANUF_KW.search(t) and DATE_RE.search(t):
            R["mfg_date"] = dict(found=True, value=t, line=i)
        # Best before
        if not R["best_before"]["found"] and BEST_RE.search(t):
            R["best_before"] = dict(found=True, value=t, line=i)
        # Consumer care: keyword + phone or email (contact may be printed just above)
        if not R["consumer_care"]["found"] and CARE_KW.search(t):
            win = _window(lines, i, 4, above=4)
            if PHONE_RE.search(win) or EMAIL_RE.search(win):
                R["consumer_care"] = dict(found=True, value=win[:160], line=i)
        # Origin
        if not R["origin"]["found"] and ORIGIN_RE.search(t):
            R["origin"] = dict(found=True, value=t, line=i)

    # ---- Table-layout fallbacks: keyword visible, value printed in another column
    if not R["net_qty"]["found"] and R["net_qty"].get("kw_line") is not None:
        k = R["net_qty"]["kw_line"]
        j = _nearest(lines, k, lambda x: not NUTRITION_RE.search(x) and parse_quantity(x, ocr_fix=True) is not None
                     and not NETQTY_KW.search(x))
        if j is not None:
            R["net_qty"] = _netqty_entry(lines, j, parse_quantity(texts[j], ocr_fix=True), px_per_mm)
            R["net_qty"]["value"] = f"{texts[k]} {texts[j]}"
        else:
            unreadable.append("net_qty")
            R["net_qty"]["value"] = texts[k] + "  (value not readable)"
    if R["mrp"]["found"]:
        i = R["mrp"]["line"]
        if R["mrp"]["price"] != R["mrp"]["price"]:  # NaN -> value in another column
            j = _nearest(lines, i, lambda x: PRICE_RE.search(x) is not None)
            if j is not None:
                R["mrp"]["price"] = float(PRICE_RE.search(texts[j]).group(2).replace(",", "."))
                R["mrp"]["value"] = f"{texts[i]} {texts[j]}"
            else:
                unreadable.append("mrp")
                R["mrp"]["value"] = texts[i] + "  (price not readable)"
    if not R["mfg_date"]["found"]:
        kw = next((i for i, t in enumerate(texts) if MFG_DATE_KW.search(t)), None)
        if kw is not None:
            j = _nearest(lines, kw, lambda x: DATE_RE.search(x) is not None)
            if j is not None:
                R["mfg_date"] = dict(found=True, value=f"{texts[kw]} {DATE_RE.search(texts[j]).group(0)}", line=kw)
            else:
                unreadable.append("mfg_date")
                R["mfg_date"]["value"] = texts[kw] + "  (date not readable)"
                R["mfg_date"]["kw_line"] = kw

    imported = bool(IMPORT_RE.search(full))
    for f in FIELDS:
        R[f]["fuzzy"] = max(fuzzy_score(tokens, kw) for kw in FUZZY_KEYWORDS[f])

    # Violations
    v = []
    if not R["manufacturer"]["found"]:
        v.append("MISSING_MANUFACTURER")
    if not R["net_qty"]["found"] and "net_qty" not in unreadable:
        v.append("MISSING_NET_QUANTITY")
    elif R["net_qty"]["found"]:
        nq = R["net_qty"]
        if not nq["qty"]["standard"]:
            v.append("NON_STANDARD_UNIT")
        if check_font and nq["numeral_mm"] < nq["required_mm"] * 0.97:
            v.append("NET_QTY_FONT_TOO_SMALL")
    if not R["mrp"]["found"]:
        v.append("MISSING_MRP")
    elif not R["mrp_tax"]["found"]:  # (also checked when the price itself is unreadable)
        v.append("MRP_TAX_CLAUSE_MISSING")
    if not R["mfg_date"]["found"] and "mfg_date" not in unreadable:
        v.append("MISSING_MFG_DATE")
    if not R["best_before"]["found"]:
        v.append("MISSING_BEST_BEFORE")
    if not R["consumer_care"]["found"]:
        v.append("MISSING_CONSUMER_CARE")
    # Country of origin is mandatory for imported packages (Rule 6(1)(aa))
    if imported and not R["origin"]["found"]:
        v.append("MISSING_COUNTRY_OF_ORIGIN")

    return dict(fields=R, violations=v, compliant=not v, full_text=full, imported=imported, unreadable=unreadable,
                has_pin=bool(PIN_RE.search(full)), has_phone=bool(PHONE_RE.search(full)),
                has_email=bool(EMAIL_RE.search(full)), n_dates=len(DATE_RE.findall(full)),
                has_currency=bool(PRICE_RE.search(full)))


def safe_log(x: float) -> float:
    return math.log1p(x) if x is not None and x == x and x > 0 else 0.0
