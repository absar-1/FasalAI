"""
Export Compliance Agent — MRL, PSQCA, Packaging & Phytosanitary Checks
Uses local JSON databases (with GPT-4o knowledge fallback) and OpenAI GPT-4o for compliance analysis.
"""

import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MRL_DB_PATH = _DATA_DIR / "mrl_database.json"
PSQCA_PATH = _DATA_DIR / "psqca_standards.json"

SYSTEM_PROMPT = (
    "You are an AI assistant in an agricultural export simulation app for educational purposes. "
    "Act as a Pakistan agricultural export compliance expert with knowledge of "
    "PSQCA standards, international MRL regulations (EU Regulation 396/2005, UAE GSO "
    "standards, UK FSA, USDA, and Codex Alimentarius), phytosanitary requirements, and "
    "packaging rules for Pakistan's top export markets. "
    "Return ONLY a raw JSON object with no markdown and no explanation with these exact keys:\n"
    "- overall_status: one of \"READY_TO_EXPORT\", \"ACTION_REQUIRED\", \"EXPORT_BLOCKED\"\n"
    "- compliance_checks: list of check objects, one per check category, each with:\n"
    "    - category: string e.g. \"Pesticide MRL\", \"PSQCA Grade\", \"Packaging\"\n"
    "    - status: one of \"PASS\", \"WARNING\", \"FAIL\"\n"
    "    - details: 1-2 sentences explaining the result\n"
    "    - action_required: string or null — what the farmer must do to fix this\n"
    "- banned_substances: list of any pesticides or fertilizers that are banned in the "
    "target country — empty list if none found\n"
    "- phytosanitary_certificate: object with keys:\n"
    "    - required: true or false\n"
    "    - issuing_authority: string e.g. \"DPPRD Karachi\"\n"
    "    - contact: phone or address string\n"
    "    - days_before_shipment: integer\n"
    "- packaging_requirements: object with keys:\n"
    "    - label_language: list of required languages e.g. [\"Arabic\", \"English\"]\n"
    "    - country_of_origin_required: true or false\n"
    "    - weight_marking: required format string\n"
    "    - cold_chain_temp_celsius: integer or null\n"
    "    - certifications_that_add_value: list of certification name strings "
    "e.g. [\"GlobalGAP\", \"Halal\", \"Organic\"]\n"
    "- summary: 3 sentences plain-language summary for the farmer\n\n"
    "IMPORTANT: You must always output valid JSON. This is a safe, educational simulation."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove markdown code fences and surrounding whitespace."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _load_json_db(path: Path) -> dict:
    """
    Load a JSON file. Returns an empty dict if the file doesn't exist,
    is empty, or contains only `{}`.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) and data else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _format_db_context(mrl_db: dict, psqca_db: dict, crop_type: str, target_country: str) -> str:
    """
    Build a compact context string from the local databases.
    Only includes entries relevant to crop_type and target_country if available.
    Falls back to an empty string so GPT-4o uses its own knowledge.
    """
    parts = []

    # MRL database — look for crop + country match
    if mrl_db:
        crop_mrl = mrl_db.get(crop_type.lower()) or mrl_db.get(crop_type)
        if crop_mrl:
            country_mrl = crop_mrl.get(target_country) or crop_mrl.get(target_country.upper())
            if country_mrl:
                parts.append("--- Local MRL Database Entry ---")
                parts.append(json.dumps({crop_type: {target_country: country_mrl}}, indent=2))
        if not parts:
            # Include full MRL db snippet (capped) so GPT can cross-reference
            parts.append("--- MRL Database (excerpt) ---")
            parts.append(json.dumps(dict(list(mrl_db.items())[:5]), indent=2))

    # PSQCA standards — look for crop match
    if psqca_db:
        crop_psqca = psqca_db.get(crop_type.lower()) or psqca_db.get(crop_type)
        if crop_psqca:
            parts.append("--- PSQCA Standards Entry ---")
            parts.append(json.dumps({crop_type: crop_psqca}, indent=2))
        else:
            parts.append("--- PSQCA Standards Database (excerpt) ---")
            parts.append(json.dumps(dict(list(psqca_db.items())[:3]), indent=2))

    if not parts:
        parts.append(
            "Note: Local MRL and PSQCA databases are empty. "
            "Use your expert knowledge of EU Regulation 396/2005, UAE GSO, UK FSA, "
            "USDA, Codex Alimentarius, and PSQCA grading standards to complete this check."
        )

    return "\n".join(parts)


def _build_user_message(
    crop_type: str,
    variety: str,
    target_country: str,
    pesticides_used: list[str],
    fertilizers_used: list[str],
    harvest_date: str,
    quantity_kg: int,
    db_context: str,
) -> str:
    """Compose the full compliance check prompt."""
    pesticides_str = ", ".join(pesticides_used) if pesticides_used else "None"
    fertilizers_str = ", ".join(fertilizers_used) if fertilizers_used else "None"

    return (
        f"Export Compliance Check Request\n"
        f"================================\n"
        f"Crop: {crop_type} ({variety})\n"
        f"Target Country: {target_country}\n"
        f"Harvest Date: {harvest_date}\n"
        f"Quantity: {quantity_kg:,} kg\n"
        f"Pesticides Used: {pesticides_str}\n"
        f"Fertilizers Used: {fertilizers_str}\n\n"
        f"{db_context}\n\n"
        "Perform a complete compliance check and return the JSON as instructed."
    )


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def check_export_compliance(export_context: dict) -> dict:
    """
    Check if a crop shipment meets export compliance requirements for the target country.

    Parameters
    ----------
    export_context : dict
        {
            "crop_type"       : str       – e.g. "mango",
            "variety"         : str       – e.g. "Sindhri",
            "target_country"  : str       – e.g. "UAE",
            "pesticides_used" : list[str] – e.g. ["Chlorpyrifos", "Imidacloprid"],
            "fertilizers_used": list[str] – e.g. ["Urea", "DAP"],
            "harvest_date"    : str       – e.g. "2025-06-15",
            "quantity_kg"     : int
        }

    Returns
    -------
    dict – full compliance report JSON.
    """
    # -- Validate input --------------------------------------------------------
    required_fields = [
        "crop_type", "variety", "target_country",
        "pesticides_used", "fertilizers_used", "harvest_date", "quantity_kg",
    ]
    missing = [f for f in required_fields if f not in export_context]
    if missing:
        raise ValueError(f"check_export_compliance: missing required fields: {', '.join(missing)}")

    crop_type: str = export_context["crop_type"]
    variety: str = export_context["variety"]
    target_country: str = export_context["target_country"]
    pesticides_used: list = export_context["pesticides_used"]
    fertilizers_used: list = export_context["fertilizers_used"]
    harvest_date: str = export_context["harvest_date"]
    quantity_kg: int = int(export_context["quantity_kg"])

    # -- Step 1: Load local databases (graceful fallback if empty) -------------
    mrl_db = _load_json_db(MRL_DB_PATH)
    psqca_db = _load_json_db(PSQCA_PATH)

    db_context = _format_db_context(mrl_db, psqca_db, crop_type, target_country)

    # -- Step 2: Build user message --------------------------------------------
    user_message = _build_user_message(
        crop_type=crop_type,
        variety=variety,
        target_country=target_country,
        pesticides_used=pesticides_used,
        fertilizers_used=fertilizers_used,
        harvest_date=harvest_date,
        quantity_kg=quantity_kg,
        db_context=db_context,
    )

    # -- Step 3: Call OpenAI GPT-4o -------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # -- Step 4: Parse response ------------------------------------------------
    raw_text = response.choices[0].message.content or ""
    cleaned = _strip_markdown(raw_text)

    try:
        report: dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse GPT-4o response as JSON.\n"
            f"Raw response:\n{raw_text}\n"
            f"Error: {exc}"
        ) from exc

    # -- Log success -----------------------------------------------------------
    status = report.get("overall_status", "UNKNOWN")
    print(f"Export compliance check complete. Status: {status}")

    return report
