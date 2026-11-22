"""Project-wide configuration: paths, constants and the compliance rule catalogue."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "synthetic"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_CSV = PROCESSED_DIR / "features.csv"
OCR_CACHE_DIR = PROCESSED_DIR / "ocr"

MODELS_DIR = ROOT / "models"
COMPLIANCE_MODEL_PATH = MODELS_DIR / "compliance_model.joblib"
VIOLATION_MODEL_PATH = MODELS_DIR / "violation_model.joblib"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "metrics.json"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Physical scale. A synthetic label is rendered 800 px wide and represents a
# 100 mm wide principal display panel, i.e. 8 px per mm. For real photos the
# user supplies the physical panel width so font heights can be converted to mm.
# ---------------------------------------------------------------------------
LABEL_WIDTH_PX = 800
PANEL_WIDTH_MM = 100.0
PX_PER_MM = LABEL_WIDTH_PX / PANEL_WIDTH_MM

# Ratio between the height of numerals and the height of the OCR text box that
# contains them (box includes ascenders/descenders + detector padding).
# Empirically calibrated on the synthetic set (see README).
NUMERAL_TO_BOX_RATIO = 0.68

# ---------------------------------------------------------------------------
# Violation catalogue (Legal Metrology (Packaged Commodities) Rules, 2011 -
# simplified for an academic prototype; always verify against the current
# gazette notification before using for enforcement).
# ---------------------------------------------------------------------------
VIOLATIONS = [
    "MISSING_MANUFACTURER",
    "MISSING_NET_QUANTITY",
    "NON_STANDARD_UNIT",
    "NET_QTY_FONT_TOO_SMALL",
    "MISSING_MRP",
    "MRP_TAX_CLAUSE_MISSING",
    "MISSING_MFG_DATE",
    "MISSING_BEST_BEFORE",
    "MISSING_CONSUMER_CARE",
    "MISSING_COUNTRY_OF_ORIGIN",
]

VIOLATION_INFO = {
    "MISSING_MANUFACTURER": (
        "Manufacturer / packer name & address missing",
        "Rule 6(1)",
        "Name and complete address (with PIN) of the manufacturer, packer or importer must be declared.",
    ),
    "MISSING_NET_QUANTITY": (
        "Net quantity missing",
        "Rule 6(1)",
        "Net quantity must be declared in terms of standard units of weight, measure or number.",
    ),
    "NON_STANDARD_UNIT": (
        "Net quantity in non-standard unit",
        "Rule 6(1) / SI units",
        "Units must be SI symbols such as g, kg, mg, ml, L (not gm, gms, Kgs, oz, lb, cc, fl oz).",
    ),
    "NET_QTY_FONT_TOO_SMALL": (
        "Net quantity numerals below minimum height",
        "Rule 7",
        "Numeral height must be >= 1 / 2 / 4 / 6 mm for quantities up to 50 / 200 / 1000 / above 1000 g or ml.",
    ),
    "MISSING_MRP": (
        "Retail sale price (MRP) missing",
        "Rule 6(1)",
        "Maximum retail price must be declared on the package.",
    ),
    "MRP_TAX_CLAUSE_MISSING": (
        "MRP without 'inclusive of all taxes'",
        "Rule 6(1)",
        "MRP must be declared as inclusive of all taxes.",
    ),
    "MISSING_MFG_DATE": (
        "Month & year of manufacture / packing missing",
        "Rule 6(1)",
        "Month and year in which the commodity is manufactured or pre-packed must be declared.",
    ),
    "MISSING_BEST_BEFORE": (
        "Best before / use-by date missing",
        "Rule 6(1)",
        "'Best before' or 'use by' date must be declared for commodities that may become unfit for consumption.",
    ),
    "MISSING_CONSUMER_CARE": (
        "Consumer care details missing",
        "Rule 6(1)",
        "Name, address, phone number and e-mail for consumer complaints must be declared.",
    ),
    "MISSING_COUNTRY_OF_ORIGIN": (
        "Country of origin missing",
        "Rule 6(1)",
        "Country of origin / manufacture must be declared.",
    ),
}

# Short labels used on chart axes.
VIOLATION_SHORT = {
    "MISSING_MANUFACTURER": "Manufacturer missing",
    "MISSING_NET_QUANTITY": "Net qty missing",
    "NON_STANDARD_UNIT": "Non-standard unit",
    "NET_QTY_FONT_TOO_SMALL": "Net qty font too small",
    "MISSING_MRP": "MRP missing",
    "MRP_TAX_CLAUSE_MISSING": "MRP tax clause missing",
    "MISSING_MFG_DATE": "Mfg date missing",
    "MISSING_BEST_BEFORE": "Best-before missing",
    "MISSING_CONSUMER_CARE": "Consumer care missing",
    "MISSING_COUNTRY_OF_ORIGIN": "Origin missing",
}


def required_numeral_mm(base_qty: float) -> float:
    """Minimum numeral height (mm) of the net-quantity declaration (Rule 7 table)."""
    if base_qty <= 50:
        return 1.0
    if base_qty <= 200:
        return 2.0
    if base_qty <= 1000:
        return 4.0
    return 6.0


def ensure_dirs() -> None:
    for d in (IMAGES_DIR, PROCESSED_DIR, OCR_CACHE_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
