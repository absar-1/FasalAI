"""
Fasal AI — Orchestrator Pipeline
Wires all five agents into a single end-to-end run for testing and demonstration.
Run from the fasal-ai/ directory:  python src/main.py
"""

import sys
import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make `src/` importable regardless of working directory
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent          # .../fasal-ai/src
ROOT_DIR = SRC_DIR.parent                          # .../fasal-ai
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ["OPENAI_API_KEY", "PLANT_ID_API_KEY"]
missing_keys = [k for k in REQUIRED_KEYS if not os.getenv(k)]
if missing_keys:
    raise EnvironmentError(
        f"ERROR: Missing required API keys: {', '.join(missing_keys)}\n"
        f"Copy .env.example → .env and fill in the values."
    )

# ---------------------------------------------------------------------------
# Agent imports
# ---------------------------------------------------------------------------
from agents.vision_agent import analyze_crop
from agents.weather_agent import get_weather_advice
from agents.agronomy_agent import get_agronomy_advice
from agents.export_compliance_agent import check_export_compliance
from agents.language_agent import format_final_output

# ---------------------------------------------------------------------------
# Test scenario
# ---------------------------------------------------------------------------
test_scenario = {
    "crop_type": "wheat",
    "variety": "Punjab-11",
    "location": "Lahore, Punjab",
    "crop_stage": "tillering",
    "soil_type": "loamy",
    "area_acres": 5,
    "last_irrigation_date": "2025-05-08",
    "farmer_question": "My wheat leaves are turning yellow, what should I do?",
    "farmer_phone": None,
    "output_language": "english",
    "image_base64": None,
    "farmer_description": "Yellow spots appearing on lower leaves, started 5 days ago",
    "target_country": "UAE",
    "pesticides_used": ["Chlorpyrifos", "Imidacloprid"],
    "fertilizers_used": ["Urea", "DAP"],
    "harvest_date": "2025-06-20",
    "quantity_kg": 5000,
}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(scenario: dict) -> None:
    print("\n" + "=" * 60)
    print("  FASAL AI — ORCHESTRATOR PIPELINE")
    print("=" * 60 + "\n")

    vision_result = None
    weather_result = None
    agronomy_result = None
    export_result = None
    final_output = None

    try:
        # ── Step 1: Vision Agent (only if image provided) ──────────────────
        if scenario.get("image_base64"):
            print("▶ [1/5] Running Vision Agent...")
            vision_result = analyze_crop({
                "image_base64": scenario["image_base64"],
                "crop_type": scenario["crop_type"],
                "location": scenario["location"],
                "farmer_description": scenario["farmer_description"],
            })
            print(f"   ✔ Disease: {vision_result.get('disease_detected')} "
                  f"| Confidence: {vision_result.get('confidence')}")
        else:
            print("▶ [1/5] Vision Agent — skipped (no image provided)\n")

        # ── Step 2: Weather Agent ───────────────────────────────────────────
        print("▶ [2/5] Running Weather Agent...")
        weather_result = get_weather_advice({
            "location": scenario["location"],
            "crop_type": scenario["crop_type"],
            "crop_stage": scenario["crop_stage"],
            "last_irrigation_date": scenario["last_irrigation_date"],
        })
        alert_count = len(weather_result.get("proactive_alerts", []))
        print(f"   ✔ Conditions: {weather_result.get('current_conditions')}")
        print(f"   ✔ Proactive alerts: {alert_count}\n")

        # ── Step 3: Agronomy Agent ──────────────────────────────────────────
        print("▶ [3/5] Running Agronomy Agent...")
        agronomy_result = get_agronomy_advice({
            "crop_type": scenario["crop_type"],
            "crop_stage": scenario["crop_stage"],
            "disease_result": vision_result,
            "weather_result": weather_result,
            "farmer_question": scenario["farmer_question"],
            "soil_type": scenario["soil_type"],
            "area_acres": scenario["area_acres"],
        })
        fert = agronomy_result.get("fertilizer_recommendation", {})
        yield_pred = agronomy_result.get("yield_prediction", {})
        print(f"   ✔ Fertilizer: {fert.get('type')} — {fert.get('quantity_per_acre')}")
        print(f"   ✔ Yield prediction: {yield_pred.get('expected_yield_per_acre')} "
              f"({yield_pred.get('confidence')} confidence)\n")

        # ── Step 4: Export Compliance Agent ────────────────────────────────
        if scenario.get("target_country"):
            print("▶ [4/5] Running Export Compliance Agent...")
            export_result = check_export_compliance({
                "crop_type": scenario["crop_type"],
                "variety": scenario["variety"],
                "target_country": scenario["target_country"],
                "pesticides_used": scenario["pesticides_used"],
                "fertilizers_used": scenario["fertilizers_used"],
                "harvest_date": scenario["harvest_date"],
                "quantity_kg": scenario["quantity_kg"],
            })
            print(f"   ✔ Export status: {export_result.get('overall_status')}\n")
        else:
            print("▶ [4/5] Export Compliance Agent — skipped (no target country)\n")

        # ── Step 5: Language & Output Agent ────────────────────────────────
        print("▶ [5/5] Running Language Agent...")
        final_output = format_final_output({
            "vision_result": vision_result,
            "weather_result": weather_result,
            "agronomy_result": agronomy_result,
            "export_result": export_result,
            "output_language": scenario.get("output_language", "english"),
            "farmer_phone": scenario.get("farmer_phone"),
        })

        top_actions = final_output.get("top_actions", [])
        print("\n   ✔ Top Actions for the farmer:")
        for action in top_actions:
            print(f"      {action}")

        print()

        # ── Pipeline complete ───────────────────────────────────────────────
        print("=" * 60)
        print("  Fasal AI pipeline complete.")
        print("=" * 60 + "\n")

        return final_output

    except Exception as exc:
        print("\n" + "!" * 60)
        print(f"  PIPELINE ERROR: {exc}")
        print("!" * 60 + "\n")
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline(test_scenario)
