"""
Synthetic product-label dataset generator.

No public dataset of Indian packaged-commodity labels with compliance
annotations exists, so we generate a realistic, fully-annotated one:

* each image is a rendered principal display panel with brand, product name,
  the mandatory declarations of the Legal Metrology (Packaged Commodities)
  Rules 2011, decoy text (nutrition facts, ingredients, batch / FSSAI numbers)
  and a random layout, font, colour scheme;
* non-compliant labels get 1-3 violations sampled with realistic frequencies
  (field removed, wrong unit, MRP without tax clause, too-small numerals ...);
* camera-like degradations (rotation, blur, noise, JPEG, lighting) are applied
  independently of the compliance label so the models must be robust to them.

Usage:  python -m src.generate_dataset --n 1200
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from src import config as C

# ---------------------------------------------------------------------------
# Fonts (Windows names first, then common Linux / macOS fallbacks)
# ---------------------------------------------------------------------------
FONT_FAMILIES = [
    ("arial.ttf", "arialbd.ttf"),
    ("calibri.ttf", "calibrib.ttf"),
    ("verdana.ttf", "verdanab.ttf"),
    ("tahoma.ttf", "tahomabd.ttf"),
    ("segoeui.ttf", "segoeuib.ttf"),
    ("trebuc.ttf", "trebucbd.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]
RUPEE_FONTS = {"arial.ttf", "calibri.ttf", "segoeui.ttf"}


def available_families() -> list[tuple[str, str]]:
    fams = []
    for reg, bold in FONT_FAMILIES:
        try:
            ImageFont.truetype(reg, 20)
            ImageFont.truetype(bold, 20)
            fams.append((reg, bold))
        except OSError:
            continue
    if not fams:
        raise RuntimeError("No TrueType fonts found. Install DejaVu or Liberation fonts.")
    return fams


_FONT_CACHE: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(size))
    return _FONT_CACHE[key]


def numeral_ratio(path: str) -> float:
    """Height of digit glyphs relative to the font size (font-specific)."""
    f = font(path, 100)
    x0, y0, x1, y1 = f.getbbox("0123456789")
    return (y1 - y0) / 100.0


# ---------------------------------------------------------------------------
# Content vocabulary (all brand / company names are fictitious)
# ---------------------------------------------------------------------------
PRODUCTS = {
    "Food": [
        ("Premium Basmati Rice", "solid"), ("Whole Wheat Atta", "solid"), ("Assam Tea", "solid"),
        ("Cream Biscuits", "solid"), ("Roasted Peanuts", "solid"), ("Turmeric Powder", "solid"),
        ("Tomato Ketchup", "solid"), ("Instant Noodles", "solid"), ("Refined Sunflower Oil", "liquid"),
        ("Cow Ghee", "liquid"),
    ],
    "Beverages": [
        ("Mango Fruit Drink", "liquid"), ("Packaged Drinking Water", "liquid"), ("Instant Coffee", "solid"),
        ("Rose Sharbat", "liquid"),
    ],
    "Personal Care": [
        ("Herbal Shampoo", "liquid"), ("Neem Bathing Soap", "solid"), ("Fluoride Toothpaste", "solid"),
        ("Coconut Hair Oil", "liquid"), ("Moisturising Lotion", "liquid"),
    ],
    "Household": [
        ("Detergent Powder", "solid"), ("Dishwash Liquid", "liquid"), ("Floor Cleaner", "liquid"),
        ("Toilet Cleaner", "liquid"),
    ],
}
BRANDS = ["Velmora", "Annapurni", "Haritika", "Zestivo", "Nilgiro", "Swadika", "Purnam", "Oceanna",
          "Tarangi", "Grainora", "Bhoomika", "Sunveda", "Mitraa", "Rasika", "Kesarvan", "Dhanyam"]
COMPANY_SUFFIX = ["Consumer Products Pvt. Ltd.", "Foods Pvt. Ltd.", "Industries Ltd.", "Agro Foods LLP",
                  "Home Care Pvt. Ltd.", "Enterprises"]
CITIES = [("Mumbai", "Maharashtra", "400"), ("Pune", "Maharashtra", "411"), ("New Delhi", "Delhi", "110"),
          ("Bengaluru", "Karnataka", "560"), ("Chennai", "Tamil Nadu", "600"), ("Kolkata", "West Bengal", "700"),
          ("Ahmedabad", "Gujarat", "380"), ("Hyderabad", "Telangana", "500"), ("Jaipur", "Rajasthan", "302"),
          ("Lucknow", "Uttar Pradesh", "226"), ("Indore", "Madhya Pradesh", "452"), ("Nagpur", "Maharashtra", "440")]
AREAS = ["MIDC Industrial Area", "Phase II, Industrial Estate", "Sector 18", "GIDC Estate", "SIDCO Industrial Estate",
         "Udyog Vihar", "Peenya Industrial Area", "Kalyani Nagar", "Okhla Phase I"]
IMPORT_COUNTRIES = ["Thailand", "Malaysia", "Sri Lanka", "UAE", "Vietnam"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
TAGLINES = ["100% Natural", "No Added Preservatives", "Rich in Fibre", "Hygienically Packed", "Since 1985",
            "Premium Quality", "Farm Fresh", "Trusted by Millions"]

# Relative frequency of each violation among non-compliant labels.
VIOLATION_WEIGHTS = {
    "MISSING_CONSUMER_CARE": 0.17, "MRP_TAX_CLAUSE_MISSING": 0.14, "NON_STANDARD_UNIT": 0.13,
    "MISSING_MFG_DATE": 0.12, "NET_QTY_FONT_TOO_SMALL": 0.10, "MISSING_COUNTRY_OF_ORIGIN": 0.10,
    "MISSING_MANUFACTURER": 0.08, "MISSING_BEST_BEFORE": 0.07, "MISSING_MRP": 0.05, "MISSING_NET_QUANTITY": 0.04,
}
CONFLICTS = {
    "MISSING_MRP": {"MRP_TAX_CLAUSE_MISSING"},
    "MRP_TAX_CLAUSE_MISSING": {"MISSING_MRP"},
    "MISSING_NET_QUANTITY": {"NON_STANDARD_UNIT", "NET_QTY_FONT_TOO_SMALL"},
    "NON_STANDARD_UNIT": {"MISSING_NET_QUANTITY"},
    "NET_QTY_FONT_TOO_SMALL": {"MISSING_NET_QUANTITY"},
}


def sample_violations(rng: random.Random, p_compliant: float) -> list[str]:
    if rng.random() < p_compliant:
        return []
    k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    chosen: list[str] = []
    while len(chosen) < k:
        pool = [v for v in VIOLATION_WEIGHTS if v not in chosen
                and not any(v in CONFLICTS.get(c, set()) for c in chosen)]
        if not pool:
            break
        chosen.append(rng.choices(pool, weights=[VIOLATION_WEIGHTS[v] for v in pool])[0])
    return chosen


# ---------------------------------------------------------------------------
# Text builders for each declaration
# ---------------------------------------------------------------------------
def fmt_qty(base: float, kind: str) -> str:
    if kind == "solid":
        return f"{base / 1000:g} kg" if base >= 1000 else f"{base:g} g"
    return f"{base / 1000:g} L" if base >= 1000 else f"{base:g} ml"


def fmt_qty_nonstandard(base: float, kind: str, rng: random.Random) -> str:
    if kind == "solid":
        opts = [f"{base / 28.35:.1f} oz", f"{base / 453.6:.2f} lbs"]
        opts += [f"{base / 1000:g} Kgs"] if base >= 1000 else [f"{base:g} gms", f"{base:g} gm", f"{base:g} Gms"]
    else:
        opts = [f"{base:g} cc", f"{base / 29.57:.1f} fl oz"]
    return rng.choice(opts)


def make_spec(idx: int, rng: random.Random, p_compliant: float) -> dict:
    category = rng.choices(list(PRODUCTS), weights=[0.4, 0.15, 0.25, 0.2])[0]
    product, kind = rng.choice(PRODUCTS[category])
    brand = rng.choice(BRANDS)
    violations = sample_violations(rng, p_compliant)

    qty_opts = [20, 50, 100, 200, 250, 500, 1000, 2000, 5000] if kind == "solid" else [100, 200, 250, 500, 1000, 2000]
    if "NET_QTY_FONT_TOO_SMALL" in violations:
        qty_opts = [q for q in qty_opts if q > 200]
    qty = rng.choice(qty_opts)

    city, state, pin_prefix = rng.choice(CITIES)
    pin = pin_prefix + f"{rng.randint(1, 99):03d}"
    company = f"{brand} {rng.choice(COMPANY_SUFFIX)}"
    address = f"Plot No. {rng.randint(1, 250)}, {rng.choice(AREAS)}, {city}, {state} - {pin}"
    # Country of origin is only mandatory for imported goods, so that violation implies an import.
    imported = rng.random() < 0.15 or "MISSING_COUNTRY_OF_ORIGIN" in violations
    country = rng.choice(IMPORT_COUNTRIES) if imported else "India"
    price = round(qty ** 0.8 * rng.uniform(0.3, 1.2) + rng.randint(5, 40))
    year = rng.choice([2025, 2026])
    mon = rng.randint(1, 12)

    return dict(idx=idx, category=category, product=product, kind=kind, brand=brand, violations=violations,
                qty=qty, company=company, address=address, imported=imported, country=country,
                price=price, year=year, month=mon, phone=f"1800-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
                email=f"care@{brand.lower()}.{rng.choice(['in', 'co.in', 'com'])}",
                # layout style: two-column "KEY : value" table like many real packs
                table=rng.random() < 0.35, photo=rng.random() < 0.35)


def declaration_blocks(spec: dict, rng: random.Random, rupee_ok: bool) -> list[dict]:
    """Return the list of text blocks (excluding header and net quantity)."""
    v = set(spec["violations"])
    rs = "₹" if (rupee_ok and rng.random() < 0.5) else rng.choice(["Rs.", "Rs"])
    blocks = []

    mm, yy = spec["month"], spec["year"]
    if spec["table"]:
        return table_blocks(spec, rng, rs)

    # MRP
    if "MISSING_MRP" not in v:
        p = spec["price"]
        if "MRP_TAX_CLAUSE_MISSING" in v:
            txt = rng.choice([f"MRP {rs} {p:.2f}", f"M.R.P. {rs} {p}/-", f"MRP: {rs} {p:.2f} (+ taxes extra)",
                              f"Max. Retail Price {rs} {p:.2f}"])
        else:
            txt = rng.choice([f"MRP {rs} {p:.2f} (Incl. of all taxes)", f"M.R.P. {rs} {p}/- (Inclusive of all taxes)",
                              f"MRP: {rs} {p:.2f} incl. of all taxes",
                              f"Max. Retail Price {rs} {p:.2f} (Inclusive of all taxes)"])
        blocks.append(dict(key="mrp", text=txt, bold=rng.random() < 0.5))

    # Manufacturing date
    if "MISSING_MFG_DATE" not in v:
        txt = rng.choice([f"Mfg. Date: {mm:02d}/{yy}", f"Date of Mfg.: {rng.randint(1, 28):02d}/{mm:02d}/{yy}",
                          f"Pkd. On: {MONTHS[mm - 1]} {yy}", f"Packed on: {MONTHS[mm - 1]} {yy}"])
        blocks.append(dict(key="mfg_date", text=txt))

    # Best before
    if "MISSING_BEST_BEFORE" not in v:
        n = rng.choice([6, 9, 12, 18, 24])
        ey, em = yy + (mm - 1 + n) // 12, (mm - 1 + n) % 12 + 1
        txt = rng.choice([f"Best Before: {n} months from manufacture", f"Use By: {em:02d}/{ey}",
                          f"Expiry Date: {MONTHS[em - 1]} {ey}", f"Best before {n} months from packing"])
        blocks.append(dict(key="best_before", text=txt))

    blocks += _common_blocks(spec, rng)
    rng.shuffle(blocks)
    return blocks


def _common_blocks(spec: dict, rng: random.Random) -> list[dict]:
    """Manufacturer, consumer care, origin and decoy blocks (shared by both layouts)."""
    v = set(spec["violations"])
    blocks = []
    if "MISSING_MANUFACTURER" not in v:
        prefix = "Imported & Marketed by:" if spec["imported"] else rng.choice(
            ["Mfd. by:", "Manufactured by:", "Packed by:", "Marketed by:", "Mfd. & Pkd. by:", "MANUFACTURED BY :"])
        blocks.append(dict(key="manufacturer", text=f"{prefix} {spec['company']}, {spec['address']}"))
    if "MISSING_CONSUMER_CARE" not in v:
        txt = rng.choice([
            f"For consumer complaints contact: Consumer Care Manager, {spec['company']}, {spec['address']}. "
            f"Ph: {spec['phone']}, Email: {spec['email']}",
            f"Customer Care: {spec['phone']} | {spec['email']}",
            f"Consumer Care Cell: Toll free {spec['phone']}, e-mail: {spec['email']}",
            f"Email ID: {spec['email']} Ph. No.: {spec['phone']} Details of consumer care cell: in case of any "
            f"consumer complaints, contact our registered address provided",
        ])
        blocks.append(dict(key="consumer_care", text=txt))
    if "MISSING_COUNTRY_OF_ORIGIN" not in v:
        c = spec["country"]
        if spec["imported"] or rng.random() < 0.6:
            blocks.append(dict(key="origin", text=f"Country of Origin: {c}" if spec["imported"] or rng.random() < 0.7
                               else "Made in India"))
    decoys = [f"Batch No.: {rng.choice('ABCDEFGHKLMN')}{rng.randint(1000, 99999)}",
              "Store in a cool, dry and hygienic place.",
              f"Nutritional Information (per 100 g): Energy {rng.randint(50, 550)} kcal, "
              f"Protein {rng.uniform(0, 20):.1f} g, Fat {rng.uniform(0, 30):.1f} g"]
    if spec["category"] in ("Food", "Beverages"):
        decoys.append(f"FSSAI Lic. No. {rng.randint(10 ** 13, 10 ** 14 - 1)}")
        decoys.append("Ingredients: " + ", ".join(rng.sample(
            ["Wheat Flour", "Sugar", "Edible Vegetable Oil", "Salt", "Milk Solids", "Spices", "Water",
             "Fruit Pulp", "Acidity Regulator (330)", "Emulsifier (322)"], 4)))
    else:
        decoys.append("Keep out of reach of children.")
        decoys.append("Directions: Use as directed on the pack.")
    for d in rng.sample(decoys, rng.randint(2, min(4, len(decoys)))):
        blocks.append(dict(key="decoy", text=d))
    return blocks


def table_blocks(spec: dict, rng: random.Random, rs: str) -> list[dict]:
    """Two-column 'KEY : value' declaration table (values often inkjet-printed,
    slightly misaligned), as seen on many real Indian packs."""
    v = set(spec["violations"])
    mm, yy = spec["month"], spec["year"]
    dd = rng.randint(1, 28)
    table = []
    if "MISSING_MRP" not in v:
        table.append(dict(key="mrp", text=rng.choice(["MRP ₹ :", "MRP :", "M.R.P. :"]).replace("₹", rs.strip(".")),
                          value=rng.choice([f"Rs.{spec['price']}/-", f"{spec['price']:.2f}", f"₹{spec['price']}"
                                            if rs == "₹" else f"Rs {spec['price']}.00"])))
        if "MRP_TAX_CLAUSE_MISSING" not in v:
            table.append(dict(key="mrp", text=rng.choice(["(INCL. OF ALL TAXES)", "(Inclusive of all taxes)"]), small=True))
    table.append("NETQTY")  # placeholder, filled by render()
    if "MISSING_MFG_DATE" not in v:
        table.append(dict(key="mfg_date", text=rng.choice(["MFG DATE :", "MFD :", "PKD. ON :", "DATE OF MFG. :"]),
                          value=rng.choice([f"{dd:02d}{MONTHS[mm - 1]}{yy % 100}", f"{dd:02d}/{mm:02d}/{yy}",
                                            f"{MONTHS[mm - 1]} {yy}"])))
    if "MISSING_BEST_BEFORE" not in v:
        n = rng.choice([6, 9, 12, 18])
        ey, em = yy + (mm - 1 + n) // 12, (mm - 1 + n) % 12 + 1
        if rng.random() < 0.6:
            table.append(dict(key="best_before", text=rng.choice(["USE BY :", "EXPIRY :", "BEST BEFORE :"]),
                              value=f"{dd:02d}{MONTHS[em - 1]}{ey % 100}"))
        else:
            table.append(dict(key="best_before", text=f"BEST BEFORE {n} MONTHS FROM MANUFACTURE"))
    table.append(dict(key="decoy", text="BATCH NO. :", value=f"{rng.choice('ABHKM')}{rng.choice('DLX')}{rng.randint(100000, 999999)}"))
    head = _common_blocks(spec, rng)
    rng.shuffle(head)
    return head + [dict(table=table)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def wrap(text: str, f: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if f.getlength(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def render(spec: dict, rng: random.Random, families: list[tuple[str, str]]) -> tuple[Image.Image, dict]:
    W = C.LABEL_WIDTH_PX
    reg, bold = rng.choice(families)
    margin = rng.randint(30, 48)
    max_w = W - 2 * margin
    rupee_ok = Path(reg).name.lower() in RUPEE_FONTS

    bg = tuple(rng.randint(225, 255) for _ in range(3))
    ink = tuple(rng.randint(0, 60) for _ in range(3))
    accent = rng.choice([(180, 30, 40), (20, 90, 160), (30, 120, 60), (200, 110, 0), (110, 40, 130), (0, 110, 120)])

    rows: list[tuple] = []  # (key, line_text, font_path, size, color, value-or-None)
    table_mode = spec["table"]

    # Net quantity: font chosen so numeral height is (non-)compliant with Rule 7.
    v = set(spec["violations"])
    req_mm = C.required_numeral_mm(spec["qty"])
    nq_meta = dict(net_qty_text="", numeral_mm=np.nan, required_mm=req_mm)
    nq_rows = []
    if "MISSING_NET_QUANTITY" not in v:
        if table_mode:
            prefix = rng.choice(["NET WT. :", "NET QTY. :", "NET WEIGHT :"] if spec["kind"] == "solid"
                                else ["NET VOL. :", "NET QTY. :", "NET CONTENT :"])
        else:
            prefix = rng.choice(["Net Wt.", "Net Weight:", "Net Qty:", "Net Quantity:", "NET WT."] if spec["kind"] == "solid"
                                else ["Net Vol.", "Net Volume:", "Net Qty:", "Net Content:"])
        qty_txt = fmt_qty_nonstandard(spec["qty"], spec["kind"], rng) if "NON_STANDARD_UNIT" in v \
            else fmt_qty(spec["qty"], spec["kind"])
        nq_text = f"{prefix} {qty_txt}"
        ratio = numeral_ratio(bold)
        need_px = req_mm * C.PX_PER_MM
        mult = rng.uniform(0.45, 0.75) if "NET_QTY_FONT_TOO_SMALL" in v else rng.uniform(1.12, 1.6)
        size = max(18 if mult < 1 else 20, int(round(need_px * mult / ratio)))
        if mult >= 1:
            min_ok = int(np.ceil(need_px * 1.08 / ratio))
            while font(bold, size).getlength(nq_text) > max_w and size > min_ok:
                size -= 2
        numeral_px = ratio * size
        nq_meta = dict(net_qty_text=nq_text, numeral_mm=round(numeral_px / C.PX_PER_MM, 2), required_mm=req_mm)
        if table_mode:  # numerals carry the required height; key printed in the same size
            qty_txt = qty_txt.replace(" ", "") if rng.random() < 0.5 else qty_txt
            nq_rows.append(("net_qty", prefix, bold, size, ink, qty_txt))
            nq_meta["net_qty_text"] = f"{prefix} {qty_txt}"
        else:
            for ln in wrap(nq_text, font(bold, size), max_w):
                nq_rows.append(("net_qty", ln, bold, size, ink, None))

    # Header
    brand_size = rng.randint(54, 78)
    band = rng.random() < 0.6
    rows.append(("brand", spec["brand"].upper(), bold, brand_size, (255, 255, 255) if band else accent, None))
    rows.append(("product", spec["product"], bold, rng.randint(28, 38), ink, None))
    for t in rng.sample(TAGLINES, rng.randint(0, 2)):
        rows.append(("tagline", t, reg, rng.randint(20, 26), accent, None))

    body_size = rng.randint(17, 22)
    blocks = declaration_blocks(spec, rng, rupee_ok)
    nq_pos = rng.choice([0, 1, len(blocks)])  # net qty near top or at the bottom
    if table_mode:
        nq_pos = -1
    table_size = int(body_size * rng.uniform(1.3, 1.7))
    for i, b in enumerate(blocks):
        if i == nq_pos:
            rows.extend(nq_rows)
        if "table" in b:
            for item in b["table"]:
                if item == "NETQTY":
                    rows.extend(nq_rows)
                elif item.get("small"):
                    rows.append((item["key"], item["text"], bold, body_size, ink, None))
                else:
                    rows.append((item["key"], item["text"], bold, table_size, ink, item.get("value")))
            continue
        fpath = bold if b.get("bold") else reg
        for ln in wrap(b["text"], font(fpath, body_size), max_w):
            rows.append((b["key"], ln, fpath, body_size, ink, None))
    if nq_pos == len(blocks):
        rows.extend(nq_rows)

    # Layout
    gap = rng.randint(6, 12)
    heights = []
    for _, txt, fp, sz, _, _ in rows:
        x0, y0, x1, y1 = font(fp, sz).getbbox("Ag" + txt)
        heights.append(y1 + 4)
    H = int(sum(heights) + gap * len(rows) + 2 * margin + 40)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    y = margin
    if band:
        d.rectangle([0, 0, W, margin + heights[0] + 16], fill=accent)
        y += 4
    center_header = rng.random() < 0.5
    veg_mark = spec["category"] in ("Food", "Beverages") and rng.random() < 0.6
    row_boxes = []
    val_x = margin + int(max_w * rng.uniform(0.45, 0.55))
    val_dy = rng.choice([0.0, 0.0, -0.5, 0.4])  # inkjet-printed values are often misaligned
    for (key, txt, fp, sz, col, val), h in zip(rows, heights):
        f = font(fp, sz)
        x = margin
        if key in ("brand", "product", "tagline") and center_header:
            x = int((W - f.getlength(txt)) / 2)
        d.text((x, y), txt, font=f, fill=col)
        if val:
            vf = font(fp if key == "net_qty" else (reg if rng.random() < 0.5 else fp), sz)
            d.text((max(val_x, x + f.getlength(txt) + 25), y + int(val_dy * h)), val, font=vf, fill=col)
        row_boxes.append(dict(key=key, text=txt, y=y, h=h, size=sz))
        y += h + gap
        if key == "brand" and band:
            y += 16
    if veg_mark:  # green "veg" symbol
        s = 34
        x0, y0 = W - margin - s, H - margin - s
        d.rectangle([x0, y0, x0 + s, y0 + s], outline=(0, 140, 0), width=3)
        d.ellipse([x0 + 9, y0 + 9, x0 + s - 9, y0 + s - 9], fill=(0, 140, 0))

    meta = dict(font=Path(reg).name, bg=bg, **nq_meta, rows=row_boxes)
    return img, meta


def photo_composite(img: Image.Image, rng: random.Random) -> tuple[Image.Image, float]:
    """Place the label on a larger 'tabletop' background (fabric stripes / plain),
    like a phone photo. Returns the image and how many label-widths it spans."""
    f = rng.uniform(1.15, 1.6)
    W, H = int(img.width * f), int(img.height * rng.uniform(1.1, 1.4))
    c1 = tuple(rng.randint(150, 240) for _ in range(3))
    c2 = tuple(rng.randint(20, 120) for _ in range(3))
    bg = Image.new("RGB", (W, H), c1)
    d = ImageDraw.Draw(bg)
    if rng.random() < 0.7:  # diagonal stripes
        step = rng.randint(80, 200)
        for k in range(-H, W + H, step):
            d.polygon([(k, 0), (k + step // 2, 0), (k + step // 2 - H, H), (k - H, H)], fill=c2)
    x, y = rng.randint(0, W - img.width), rng.randint(0, H - img.height)
    bg.paste(img, (x, y))
    out_w = rng.randint(900, 1400)
    bg = bg.resize((out_w, int(H * out_w / W)), Image.LANCZOS)
    return bg, W / img.width


def degrade(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    p = dict(angle=0.0, blur=0.0, noise=0.0, jpeg=95, brightness=1.0, contrast=1.0)
    bg = img.getpixel((5, img.height - 5))
    if rng.random() < 0.6:
        p["angle"] = round(rng.uniform(-3, 3), 2)
        img = img.rotate(p["angle"], resample=Image.BICUBIC, expand=False, fillcolor=bg)
    if rng.random() < 0.4:
        p["blur"] = round(rng.uniform(0.3, 1.1), 2)
        img = img.filter(ImageFilter.GaussianBlur(p["blur"]))
    p["brightness"] = round(rng.uniform(0.85, 1.1), 2)
    p["contrast"] = round(rng.uniform(0.8, 1.15), 2)
    img = ImageEnhance.Brightness(img).enhance(p["brightness"])
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    if rng.random() < 0.4:
        p["noise"] = round(rng.uniform(3, 10), 1)
        arr = np.asarray(img).astype(np.float32)
        arr += np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, p["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    p["jpeg"] = rng.randint(45, 95)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=p["jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("RGB"), p


def generate(n: int, out_dir: Path = C.DATASET_DIR, p_compliant: float = 0.45, seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    families = available_families()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        spec = make_spec(i, rng, p_compliant)
        img, meta = render(spec, rng, families)
        span = 1.0
        if spec["photo"]:
            img, span = photo_composite(img, rng)
        img, deg = degrade(img, rng)
        fname = f"label_{i:05d}.jpg"
        img.save(img_dir / fname, quality=95)
        records.append(dict(
            image_id=f"label_{i:05d}", filename=f"images/{fname}", category=spec["category"],
            product=spec["product"], brand=spec["brand"], net_qty_base=spec["qty"], kind=spec["kind"],
            compliant=int(not spec["violations"]), violations="|".join(spec["violations"]),
            n_violations=len(spec["violations"]), imported=int(spec["imported"]), font=meta["font"],
            net_qty_text=meta["net_qty_text"], true_numeral_mm=meta["numeral_mm"],
            required_numeral_mm=meta["required_mm"], layout="table" if spec["table"] else "inline",
            photo=int(spec["photo"]), panel_width_mm=round(C.PANEL_WIDTH_MM * span, 2), **{f"deg_{k}": v for k, v in deg.items()},
        ))
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  generated {i + 1}/{n}")
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "labels.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1200, help="number of label images")
    ap.add_argument("--p-compliant", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    args = ap.parse_args()
    C.ensure_dirs()
    df = generate(args.n, p_compliant=args.p_compliant, seed=args.seed)
    print(f"Saved {len(df)} labels to {C.LABELS_CSV} ({df.compliant.mean():.0%} compliant)")


if __name__ == "__main__":
    main()
