"""
Part 4 - Track C: Model Prediction Explanation Pipeline

Loads best_model.pkl from Part 3, runs .predict() / .predict_proba() on three
hand-crafted feature-vector inputs, and asks an LLM (via Groq, OpenAI-
compatible HTTP API) to produce a structured JSON explanation for each
prediction. Validates the JSON against a schema and applies a PII guardrail
before every LLM call.

NOTE ON RUNNING THIS FILE:
Set MOCK_LLM = False and set the GROQ_API_KEY environment variable to
run this for real. MOCK_LLM = True uses a stand-in function so the rest of the
pipeline (encoding, prediction, schema validation, guardrail) can be tested
without a live network call -- useful for development, but the README must be
written from a MOCK_LLM = False run for actual submission.
"""

import os
import re
import json
import joblib
import jsonschema
import numpy as np
import pandas as pd
import requests

import time

MOCK_LLM = False  # <-- set to False and export GROQ_API_KEY before your real run

MODEL = "llama-3.1-8b-instant"  # Groq free tier - fast, reliable, no cost
CALL_DELAY_SECONDS = 1  # Groq's free tier rate limits are generous; light pacing is enough
URL = "https://api.groq.com/openai/v1/chat/completions"

# ---------------------------------------------------------------------------
# Same column order the model was trained on in Part 3 (documented there):
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "income_category", "ocean_proximity_INLAND", "ocean_proximity_ISLAND",
    "ocean_proximity_NEAR BAY", "ocean_proximity_NEAR OCEAN",
]
INCOME_ORDER_MAP = {"Low": 0, "Medium": 1, "High": 2, "VeryHigh": 3}
OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]  # baseline dropped: <1H OCEAN


def encode_record(features: dict) -> pd.DataFrame:
    """Turn a raw feature dict into the encoded row (as a DataFrame) the pipeline expects."""
    row = {}
    row["longitude"] = features["longitude"]
    row["latitude"] = features["latitude"]
    row["housing_median_age"] = features["housing_median_age"]
    row["total_rooms"] = features["total_rooms"]
    row["total_bedrooms"] = features["total_bedrooms"]
    row["population"] = features["population"]
    row["households"] = features["households"]
    row["median_income"] = features["median_income"]
    row["income_category"] = INCOME_ORDER_MAP[features["income_category"]]
    for cat in OCEAN_CATEGORIES[1:]:  # baseline "<1H OCEAN" has no dummy column
        col = f"ocean_proximity_{cat}"
        row[col] = 1 if features["ocean_proximity"] == cat else 0
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


# ---------------------------------------------------------------------------
# Task 1: LLM API connection
# ---------------------------------------------------------------------------
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    if MOCK_LLM:
        # Stand-in used only for local development/testing without network access.
        return json.dumps({
            "prediction_label": "MOCK: above median" if "class 1" in user_prompt else "MOCK: at/below median",
            "confidence_level": "medium",
            "top_reason": "median_income is the dominant driver in this model",
            "second_reason": "geographic location contributes secondary signal",
            "next_step": "verify with a local comparable-sales estimate"
        })

    api_key = os.environ["GROQ_API_KEY"]
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    max_retries = 4
    for attempt in range(max_retries):
        response = requests.post(URL, headers=headers, json=payload)
        if response.status_code == 200:
            time.sleep(CALL_DELAY_SECONDS)  # pace the next call too
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code == 429:
            wait = 20  # default backoff if the API doesn't tell us how long
            try:
                wait = response.json()["error"]["metadata"].get("retry_after_seconds", 20)
            except Exception:
                pass
            print(f"Rate limited (429), waiting {wait:.0f}s before retry {attempt+1}/{max_retries}...")
            time.sleep(wait + 2)  # small buffer on top of the provider's suggested wait
            continue
        # any other non-200 status: fail without retrying
        print("LLM call failed, status:", response.status_code, response.text[:200])
        return None

    print("LLM call failed after retries: still rate-limited.")
    return None


# Demonstrate the function with a simple test prompt
test_output = call_llm("You are a helpful assistant.", "Reply with only the word: hello")
print("=== Task 1: test call_llm output ===")
print(repr(test_output), "\n")

# ---------------------------------------------------------------------------
# Task 2: Prompt design
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a structured explanation assistant for a house-price classification "
    "model. Given feature values, a predicted class, and a predicted probability, "
    "output ONLY a valid JSON object explaining the prediction in plain language "
    "for a non-technical client. Do not include any text outside the JSON object."
)

USER_PROMPT_TEMPLATE = """Feature values: {features}
Predicted class: {predicted_class} (1 = above median house value, 0 = at/below median)
Predicted probability of class 1: {probability:.4f}

Return a JSON object with exactly these fields:
{{
  "prediction_label": "string, e.g. 'above median' or 'at or below median'",
  "confidence_level": "low|medium|high",
  "top_reason": "string, the most likely driver of this prediction",
  "second_reason": "string, a secondary contributing factor",
  "next_step": "string, a suggested next action for the client"
}}"""

# ---------------------------------------------------------------------------
# Task 3: JSON schema (Track C)
# ---------------------------------------------------------------------------
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
}

FALLBACK = {k: None for k in EXPLANATION_SCHEMA["required"]}


def get_validated_explanation(system_prompt, user_prompt, temperature=0.0):
    raw = call_llm(system_prompt, user_prompt, temperature=temperature)
    if raw is None:
        return dict(FALLBACK)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return dict(FALLBACK)
    try:
        jsonschema.validate(parsed, EXPLANATION_SCHEMA)
    except jsonschema.ValidationError as e:
        print("Schema validation error:", e.message)
        return dict(FALLBACK)
    return parsed


# ---------------------------------------------------------------------------
# Task 4: PII guardrail
# ---------------------------------------------------------------------------
def has_pii(text):
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


print("=== Task 4: PII guardrail tests ===")
print("Clean input blocked?", has_pii("Explain this prediction for a client in Fresno."))
print("PII input blocked?", has_pii("Contact john.doe@example.com about this prediction."), "\n")

# ---------------------------------------------------------------------------
# Load best model (Task: Track C only)
# ---------------------------------------------------------------------------
pipeline = joblib.load("best_model.pkl")

hand_crafted_inputs = [
    {
        "longitude": -122.25, "latitude": 37.85, "housing_median_age": 35.0,
        "total_rooms": 1500.0, "total_bedrooms": 300.0, "population": 900.0,
        "households": 280.0, "median_income": 7.5, "income_category": "VeryHigh",
        "ocean_proximity": "NEAR BAY",
    },
    {
        "longitude": -119.5, "latitude": 36.5, "housing_median_age": 20.0,
        "total_rooms": 2200.0, "total_bedrooms": 450.0, "population": 1400.0,
        "households": 420.0, "median_income": 2.8, "income_category": "Low",
        "ocean_proximity": "INLAND",
    },
    {
        "longitude": -121.9, "latitude": 37.35, "housing_median_age": 15.0,
        "total_rooms": 3200.0, "total_bedrooms": 600.0, "population": 1600.0,
        "households": 580.0, "median_income": 5.1, "income_category": "High",
        "ocean_proximity": "<1H OCEAN",
    },
]

# ---------------------------------------------------------------------------
# Task: Temperature A/B comparison
# ---------------------------------------------------------------------------
ab_rows = []
demo_rows = []

print("=== Task 5: End-to-end demonstration (Track C) ===")
for i, features in enumerate(hand_crafted_inputs, start=1):
    encoded = encode_record(features)
    pred_class = int(pipeline.predict(encoded)[0])
    pred_proba = float(pipeline.predict_proba(encoded)[0][1])

    user_input_text = USER_PROMPT_TEMPLATE.format(
        features=features, predicted_class=pred_class, probability=pred_proba
    )

    if has_pii(user_input_text):
        print(f"Input {i}: blocked by PII guardrail.")
        continue

    explanation_t0 = get_validated_explanation(SYSTEM_PROMPT, user_input_text, temperature=0.0)
    explanation_t7 = get_validated_explanation(SYSTEM_PROMPT, user_input_text, temperature=0.7)

    valid = all(explanation_t0.get(k) is not None for k in EXPLANATION_SCHEMA["required"])

    print(f"\n--- Input {i} ---")
    print("Features:", features)
    print("Predicted class:", pred_class, " Probability:", round(pred_proba, 4))
    print("Explanation (temp=0):", explanation_t0)
    print("Explanation (temp=0.7):", explanation_t7)
    print("Valid JSON (temp=0)?", valid)

    ab_rows.append({
        "input": f"Input {i}", "output_t0": explanation_t0, "output_t7": explanation_t7,
    })
    demo_rows.append({
        "input": f"Input {i}", "pred_class": pred_class, "pred_proba": round(pred_proba, 4),
        "explanation": explanation_t0, "valid": valid,
    })

print("\n=== Part 4 complete ===")
print("(Re-run with MOCK_LLM = False and GROQ_API_KEY set for real submission output)")
