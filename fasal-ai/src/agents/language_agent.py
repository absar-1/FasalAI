"""
Language & Output Agent — Bilingual Formatting (Urdu / English)
Uses OpenAI GPT-4o to rewrite technical advice in plain farmer-friendly language.
WhatsApp delivery is disabled (React frontend handles messaging).
"""

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an AI assistant in an educational agricultural simulation. "
    "Act as a bilingual agricultural communication expert. Your job is to take expert "
    "technical advice and rewrite it in simple, clear language for a Pakistani farmer "
    "with limited education. "
    "Return ONLY a raw JSON object with no markdown and no explanation with these exact keys:\n"
    "- greeting: a warm 1-sentence greeting in the chosen language\n"
    "- disease_summary: 2 sentences about the crop diagnosis in plain language "
    "(omit this key entirely if vision_result is None)\n"
    "- weather_summary: 2 sentences about today's weather and what it means for the farm\n"
    "- top_actions: list of exactly 5 prioritized action strings, numbered, in plain simple "
    "language. Start each with a verb. "
    "e.g. \"1. Spray Dimethoate today before 9am\"\n"
    "- irrigation_today: single sentence — irrigate or skip and why\n"
    "- export_status: 1 sentence summary of export readiness "
    "(omit this key entirely if export_result is None)\n"
    "- whatsapp_message: a short WhatsApp-friendly version of the advice, "
    "under 300 characters, starting with \"Fasal AI:\"\n\n"
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


def _build_user_message(
    vision_result: dict | None,
    weather_result: dict,
    agronomy_result: dict,
    export_result: dict | None,
    output_language: str,
) -> str:
    """Compose the full message containing all agent results."""
    lang_instruction = (
        "Write ALL text values in Urdu script. Keys must remain in English."
        if output_language == "urdu"
        else "Write all text values in clear, simple English."
    )

    parts = [
        f"Target Language: {output_language.upper()}",
        f"Language Instruction: {lang_instruction}",
        "",
    ]

    if vision_result:
        parts.append("--- Crop Disease / Vision Result ---")
        parts.append(json.dumps(vision_result, indent=2, ensure_ascii=False))
    else:
        parts.append("--- Crop Disease / Vision Result ---")
        parts.append("None (no image was uploaded by the farmer)")

    parts.append("")
    parts.append("--- Weather Advisory Result ---")
    parts.append(json.dumps(weather_result, indent=2, ensure_ascii=False))

    parts.append("")
    parts.append("--- Agronomy Advisory Result ---")
    parts.append(json.dumps(agronomy_result, indent=2, ensure_ascii=False))

    if export_result:
        parts.append("")
        parts.append("--- Export Compliance Result ---")
        parts.append(json.dumps(export_result, indent=2, ensure_ascii=False))
    else:
        parts.append("")
        parts.append("--- Export Compliance Result ---")
        parts.append("None (no export check was requested)")

    parts.append("")
    parts.append(
        "Using all results above, generate the farmer-friendly output JSON as instructed. "
        "Keep language extremely simple — assume the farmer has minimal formal education."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def format_final_output(all_results: dict) -> dict:
    """
    Reformat all agent results into a farmer-friendly bilingual summary.

    Parameters
    ----------
    all_results : dict
        {
            "vision_result"   : dict | None  – output of analyze_crop(),
            "weather_result"  : dict         – output of get_weather_advice(),
            "agronomy_result" : dict         – output of get_agronomy_advice(),
            "export_result"   : dict | None  – output of check_export_compliance(),
            "output_language" : str          – "urdu" or "english",
            "farmer_phone"    : str | None   – WhatsApp number (unused; React handles this)
        }

    Returns
    -------
    dict – formatted output JSON with added 'sent_whatsapp' key.
    """
    # -- Validate input --------------------------------------------------------
    required_fields = [
        "vision_result", "weather_result", "agronomy_result",
        "export_result", "output_language",
    ]
    missing = [f for f in required_fields if f not in all_results]
    if missing:
        raise ValueError(f"format_final_output: missing required fields: {', '.join(missing)}")

    vision_result: dict | None = all_results["vision_result"]
    weather_result: dict = all_results["weather_result"]
    agronomy_result: dict = all_results["agronomy_result"]
    export_result: dict | None = all_results["export_result"]
    output_language: str = all_results.get("output_language", "english").lower()
    farmer_phone: str | None = all_results.get("farmer_phone")  # Reserved for React frontend

    if output_language not in ("urdu", "english"):
        raise ValueError(
            f"output_language must be 'urdu' or 'english', got: '{output_language}'"
        )

    # -- Step 1: Build user message --------------------------------------------
    user_message = _build_user_message(
        vision_result=vision_result,
        weather_result=weather_result,
        agronomy_result=agronomy_result,
        export_result=export_result,
        output_language=output_language,
    )

    # -- Step 2: Call OpenAI GPT-4o -------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in the environment.")

    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    # -- Step 3: Parse response ------------------------------------------------
    raw_text = response.choices[0].message.content or ""
    cleaned = _strip_markdown(raw_text)

    try:
        output: dict = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse GPT-4o response as JSON.\n"
            f"Raw response:\n{raw_text}\n"
            f"Error: {exc}"
        ) from exc

    # -- Step 4: WhatsApp delivery (handled by React frontend) -----------------
    # farmer_phone is accepted in the contract but messaging is not sent here.
    # The React frontend reads 'whatsapp_message' from this response and
    # dispatches it through its own messaging layer.
    sent_whatsapp = False
    output["sent_whatsapp"] = sent_whatsapp

    # -- Log success -----------------------------------------------------------
    print(
        f"Output formatted in {output_language}. "
        f"WhatsApp sent: {str(sent_whatsapp).lower()}"
    )

    return output
