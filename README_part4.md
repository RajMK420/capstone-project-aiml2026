# Part 4 — LLM-Powered Feature (Track C: Model Prediction Explanation Pipeline)

**Track chosen: C — Model Prediction Explanation Pipeline.** I picked this one
because it builds directly on `best_model.pkl` from Part 3 instead of starting
a new dataset from scratch — it takes the tuned Random Forest and explains its
predictions in plain language for a non-technical client.

## LLM provider
I used **Groq** (`api.groq.com`), not OpenRouter. I actually started with
OpenRouter since that's the example in the brief, but ran into a real problem
worth documenting honestly: OpenRouter's free-tier models were rate-limited
(HTTP 429) or unavailable (HTTP 404) across six different models I tried, and
their paid models need a purchased credit balance my account didn't have
(HTTP 402). Groq's free tier is more generous and worked reliably, and it uses
the exact same request format the brief describes (JSON body with `model` and
`messages`, Bearer token auth, OpenAI-compatible), so no logic had to change,
just the URL and model name. Model used: `llama-3.1-8b-instant`.

## call_llm function and test call
`call_llm(system_prompt, user_prompt, temperature, max_tokens)` builds the
JSON payload, sends it via `requests.post`, and returns
`response.json()["choices"][0]["message"]["content"]`. If the response isn't
200, it retries automatically (up to 4 times) if the failure is a 429 rate
limit, reading the provider's suggested wait time from the error response;
any other error returns `None` without retrying.

Test call: `call_llm("You are a helpful assistant.", "Reply with only the word: hello")`
returned: **`'hello'`** — confirms the connection works before using it for
anything real.

The API key is read from `os.environ["GROQ_API_KEY"]` — never hardcoded
anywhere in the script.

## Prompt design

**System prompt (verbatim):**
```
You are a structured explanation assistant for a house-price classification
model. Given feature values, a predicted class, and a predicted probability,
output ONLY a valid JSON object explaining the prediction in plain language
for a non-technical client. Do not include any text outside the JSON object.
```

**User prompt template (verbatim, with placeholders):**
```
Feature values: {features}
Predicted class: {predicted_class} (1 = above median house value, 0 = at/below median)
Predicted probability of class 1: {probability:.4f}

Return a JSON object with exactly these fields:
{
  "prediction_label": "string, e.g. 'above median' or 'at or below median'",
  "confidence_level": "low|medium|high",
  "top_reason": "string, the most likely driver of this prediction",
  "second_reason": "string, a secondary contributing factor",
  "next_step": "string, a suggested next action for the client"
}
```

I used **temperature=0** for the actual submission run. The task is a
structured-extraction-style task — I want the same feature values to always
produce a consistent explanation, not a different one every time you ask.
At temperature=0 the model always picks its single highest-probability next
token, which makes output deterministic and repeatable; at higher
temperatures it samples from a broader distribution over likely tokens, which
introduces variety but also inconsistency, which isn't what I want for a
reproducible, defensible explanation of a specific prediction.

## Temperature A/B comparison (temp=0 vs temp=0.7)

| Input | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| Input 1 (VeryHigh income, NEAR BAY) | top_reason: "High median income... strong local economy"; next_step: "consult a real estate agent" | top_reason: "Predicted high median income... associated with higher house prices"; next_step: "explore neighborhoods with a bay view" | Same conclusion and confidence, but the temp=0.7 version phrases the reasoning slightly differently and gives a more specific next step |
| Input 2 (Low income, INLAND) | top_reason: "predicted median house value is very low"; next_step: "verify property condition and recent sales data" | top_reason: "predicted probability... extremely low (0.0%)"; next_step: "explore renovating/upgrading to increase value" | temp=0.7 gave a materially different next_step — a more proactive suggestion instead of a verification step |
| Input 3 (High income, <1H OCEAN) | top_reason: "High median income... strong local economy"; next_step: "consult a real estate agent" | top_reason: "Predicted median income is high..."; next_step: "consult a real estate agent to discuss listings" | Very similar in substance, mostly rephrased |

Across all three inputs, temp=0 gave tighter, more repeatable phrasing, while
temp=0.7 varied the wording and occasionally suggested a different next step
for the same input (most visible in Input 2). This matches what temperature
controls: at 0 the model always takes the highest-probability path through
its response, so re-running the same prompt would give (close to) the same
answer every time; at 0.7 it's sampling from a wider set of plausible next
tokens, so the phrasing — and sometimes the specific recommendation — shifts
between runs even though the underlying prediction and confidence stayed the
same in all three cases.

## Structured output handling (Track C)

Schema (5 required scalar fields):
```python
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
```
After every LLM call: strip whitespace, `json.loads()` inside a
`try/except json.JSONDecodeError`, then `jsonschema.validate()` inside a
`try/except jsonschema.ValidationError`. On any failure, return a fallback
dict with all 5 fields set to `None` and log the error. All three real test
inputs produced valid JSON on the first try — no fallback was triggered in
the final run.

## PII guardrail
Before every LLM call, a regex check (`has_pii`) scans the outgoing prompt for
an email pattern or a phone-number pattern. Tested on two inputs:
- Clean input ("Explain this prediction for a client in Fresno.") →
  **not blocked** (correct — no PII present)
- Input containing an email ("Contact john.doe@example.com about this
  prediction.") → **blocked** (correct)

## End-to-end demonstration (real output, temp=0, Groq / llama-3.1-8b-instant)

| Input | Predicted class | Probability | LLM explanation (top_reason) | Valid JSON | Pass/Block |
|---|---|---|---|---|---|
| Input 1 — VeryHigh income, NEAR BAY | 1 (above median) | 0.975 | High median income in the area, indicating a strong local economy | Pass | Pass |
| Input 2 — Low income, INLAND | 0 (at/below median) | 0.000 | Predicted median house value is very low | Pass | Pass |
| Input 3 — High income, <1H OCEAN | 1 (above median) | 0.835 | High median income in the area, indicating a strong local economy | Pass | Pass |

All three predictions line up with what the model itself was already telling
us in Part 3 — `median_income` was consistently the top feature by importance
— so the LLM's stated top_reason matches the model's actual behavior rather
than just sounding plausible, which is a good sign the explanations are
grounded in the real prediction rather than hallucinated.

## Files
- `part4_llm_feature.py` — all code above; set `MOCK_LLM = True` for offline
  development/testing without hitting the network, `False` for the real run
  shown here
- Requires `GROQ_API_KEY` set as an environment variable before running with
  `MOCK_LLM = False`
