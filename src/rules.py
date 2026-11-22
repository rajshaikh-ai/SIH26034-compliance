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
TAX_RE = re.compile(r"incl\w*\.?\s*(of\s*)?all\s*tax", re.I)
MFG_KW = re.compile(r"\b(mfg|mfd|manufactur\w*|pkd|packed|packing|date\s*of\s*(mfg|manufacture|packing))\b", re.I)
MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*[-/']?\s*\d{2,4}"
DATE_RE = re.compile(r"(\b\d{1,2}\s*[/.\-]\s*\d{1,2}\s*[/.\-]\s*\d{2,4}\b|\b\d{1,2}\s*[/.\-]\s*\d{4}\b"
                     r"|\d{1,2}\s*-?\s*" + MONTH_RE + r"|\b" + MONTH_RE + r")", re.I)  # also 18OCT26 / 18-Oct-2026
MFG_DATE_KW = re.compile(r"((mfg|mfd|manufactur\w*|pkd|packed|packing)\s*\.?\s*(date|on|dt)|date\s*of\s*(mfg|manufacture|packing|pkg))", re.I)
IMPORT_RE = re.compile(r"\bimport(ed|er)\b", re.I)
NUTRITION_RE = re.compile(r"(per\s*\d|per\s*serve|kcal|energy|protein|fat|carbo|sugar|sodium|\(g\)|usp|unit\s*sale)", re.I)
BEST_RE = re.compile(r"(best\s*before|use\s*by|expiry|exp\.?\s*date|\bexp\b)", re.I)
MANUF_KW = re.compile(r"(\b(mfd|manufactured|mfg|mkd|marketed|packed|pkd|imported)\s*\.?\s*(&\s*\w+\.?\s*)*(by|for)\b"
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


def _window(lines: list[dict], i: int, n: int) -> str:
    return " ".join(l["text"] for l in lines[i:i + n])


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

    for i, t in enumerate(texts):
        # Manufacturer: keyword + PIN code within the next lines
        if not R["manufacturer"]["found"] and MANUF_KW.search(t):
            win = _window(lines, i, 4)
            if PIN_RE.search(win):
                R["manufacturer"] = dict(found=True, value=win[:160], line=i)
        # Net quantity
        if not R["net_qty"]["found"] and NETQTY_KW.search(t):
            q = parse_quantity(t[NETQTY_KW.search(t).end():], ocr_fix=True)
            if q:
                R["net_qty"] = _netqty_entry(lines, i, q, px_per_mm)
            elif parse_quantity(_window(lines, i + 1, 1)) and not NUTRITION_RE.search(_window(lines, i + 1, 1)):
                R["net_qty"] = _netqty_entry(lines, i + 1, parse_quantity(_window(lines, i + 1, 1)), px_per_mm)
            else:
                R["net_qty"]["kw_line"] = i
        # MRP + tax clause
        if not R["mrp"]["found"] and MRP_KW.search(t):
            win = _window(lines, i, 2)
            pm = PRICE_RE.search(t[MRP_KW.search(t).end():]) or re.search(r"(\d+(?:[.,]\d{1,2})?)", t[MRP_KW.search(t).end():])
            price = float(pm.group(pm.lastindex).replace(",", ".")) if pm else float("nan")
            R["mrp"] = dict(found=True, value=t, line=i, price=price)
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
            win = " ".join(l["text"] for l in lines[max(0, i - 4):i + 4])
            if PHONE_RE.search(win) or EMAIL_RE.search(win):
                R["consumer_care"] = dict(found=True, value=win[:160], line=i)
        # Origin
        if not R["origin"]["found"] and ORIGIN_RE.search(t):
            R["origin"] = dict(found=True, value=t, line=i)

    # ---- Table-layout fallbacks: keyword present, value printed in another column
    if not R["net_qty"]["found"] and R["net_qty"].get("kw_line") is not None:
        k = R["net_qty"]["kw_line"]
        cands = [(abs(j - k), j, parse_quantity(l["text"])) for j, l in enumerate(lines)
                 if j != k and not NUTRITION_RE.search(l["text"]) and parse_quantity(l["text"])]
        if cands:
            _, j, q = min(cands, key=lambda c: c[0])
            R["net_qty"] = _netqty_entry(lines, j, q, px_per_mm)
    if R["mrp"]["found"]:
        i = R["mrp"]["line"]
        if R["mrp"]["price"] != R["mrp"]["price"]:  # NaN -> value in another column
            pm = PRICE_RE.search(full)
            if pm:
                R["mrp"]["price"] = float(pm.group(2).replace(",", "."))
        if not R["mrp_tax"]["found"]:
            near = " ".join(l["text"] for l in lines[max(0, i - 1):i + 3])
            if TAX_RE.search(near):
                R["mrp_tax"] = dict(found=True, value=TAX_RE.search(near).group(0), line=i)
    if not R["mfg_date"]["found"]:
        kw = next((i for i, t in enumerate(texts) if MFG_DATE_KW.search(t)), None)
        dates = [(i, t) for i, t in enumerate(texts) if DATE_RE.search(t)]
        if kw is not None and dates:
            j, t = min(dates, key=lambda d: abs(d[0] - kw))
            R["mfg_date"] = dict(found=True, value=f"{texts[kw]} {DATE_RE.search(t).group(0)}", line=kw)

    imported = bool(IMPORT_RE.search(full))
    for f in FIELDS:
        R[f]["fuzzy"] = max(fuzzy_score(tokens, kw) for kw in FUZZY_KEYWORDS[f])

    # Violations
    v = []
    if not R["manufacturer"]["found"]:
        v.append("MISSING_MANUFACTURER")
    if not R["net_qty"]["found"]:
        v.append("MISSING_NET_QUANTITY")
    else:
        nq = R["net_qty"]
        if not nq["qty"]["standard"]:
            v.append("NON_STANDARD_UNIT")
        if check_font and nq["numeral_mm"] < nq["required_mm"] * 0.97:
            v.append("NET_QTY_FONT_TOO_SMALL")
    if not R["mrp"]["found"]:
        v.append("MISSING_MRP")
    elif not R["mrp_tax"]["found"]:
        v.append("MRP_TAX_CLAUSE_MISSING")
    if not R["mfg_date"]["found"]:
        v.append("MISSING_MFG_DATE")
    if not R["best_before"]["found"]:
        v.append("MISSING_BEST_BEFORE")
    if not R["consumer_care"]["found"]:
        v.append("MISSING_CONSUMER_CARE")
    # Country of origin is mandatory for imported packages (Rule 6(1)(aa))
    if imported and not R["origin"]["found"]:
        v.append("MISSING_COUNTRY_OF_ORIGIN")

    return dict(fields=R, violations=v, compliant=not v, full_text=full, imported=imported,
                has_pin=bool(PIN_RE.search(full)), has_phone=bool(PHONE_RE.search(full)),
                has_email=bool(EMAIL_RE.search(full)), n_dates=len(DATE_RE.findall(full)),
                has_currency=bool(PRICE_RE.search(full)))


def safe_log(x: float) -> float:
    return math.log1p(x) if x is not None and x == x and x > 0 else 0.0
