import os
import threading
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import streamlit as st
import json
import logging

# Attempt to import Watsonx client - if missing, code will raise informative error
try:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models import ModelInference
    _HAS_WATSONX_SDK = True
except Exception:
    # We still keep the rest of the app working; raise when the model is used.
    _HAS_WATSONX_SDK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drug-analyzer")

# -------------------------
# Watsonx.ai credentials (recommended: set these as environment variables)
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "7mYof6ZGhFSkk_3dZxmVHWSScQwHgYqttQpto9jE9L5k")
WATSONX_URL = os.getenv("WATSONX_URL", "https://api.au-syd.assistant.watson.cloud.ibm.com/instances/d64ee46d-bbb2-4ce1-a236-8e8fa2b4fb7b")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "REPLACE_WITH_YOUR_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2")

if "REPLACE_WITH_YOUR_KEY" in WATSONX_API_KEY:
    logger.warning("WATSONX_API_KEY not set. Please export WATSONX_API_KEY or edit the script before running.")

# -------------------------
# In-memory drug "database"
SAMPLE_DRUG_DB = {
    "aspirin": {
        "aliases": ["aspirin", "acetylsalicylic acid", "asa"],
        "typical_dose_mg": 300,
        "max_daily_mg": 4000,
        "interactions": {"warfarin": "high", "ibuprofen": "moderate"},
        "alternatives": ["paracetamol"],
        "side_effects": ["stomach upset", "bleeding risk"],
        "contraindications": {
            "kidney disease": False,
            "liver disease": False,
            "penicillin allergy": False,
            "elderly": False
        }
    },
    "warfarin": {
        "aliases": ["warfarin", "coumadin"],
        "typical_dose_mg": 5,
        "max_daily_mg": 15,
        "interactions": {"aspirin": "high", "amoxicillin": "moderate"},
        "alternatives": ["heparin (requires monitoring)"],
        "side_effects": ["bleeding", "bruising"],
        "contraindications": {
            "kidney disease": False,
            "liver disease": True,
            "penicillin allergy": False,
            "elderly": True
        }
    },
    "ibuprofen": {
        "aliases": ["ibuprofen", "advil", "motrin"],
        "typical_dose_mg": 400,
        "max_daily_mg": 2400,
        "interactions": {"aspirin": "moderate"},
        "alternatives": ["paracetamol"],
        "side_effects": ["nausea", "dizziness", "possible liver damage in high doses"],
        "contraindications": {
            "kidney disease": True,
            "liver disease": False,
            "penicillin allergy": False,
            "elderly": True
        }
    },
    "paracetamol": {
        "aliases": ["paracetamol", "acetaminophen", "tylenol"],
        "typical_dose_mg": 500,
        "max_daily_mg": 3000,
        "interactions": {},
        "alternatives": ["ibuprofen (if no contraindication)"],
        "side_effects": ["liver damage in overdose"],
        "contraindications": {
            "kidney disease": False,
            "liver disease": True,
            "penicillin allergy": False,
            "elderly": False
        }
    },
    "amoxicillin": {
        "aliases": ["amoxicillin"],
        "typical_dose_mg": 500,
        "max_daily_mg": 3000,
        "interactions": {"warfarin": "moderate"},
        "alternatives": ["doxycycline", "azithromycin"],
        "side_effects": ["allergic reactions", "diarrhea"],
        "contraindications": {
            "kidney disease": False,
            "liver disease": False,
            "penicillin allergy": True,
            "elderly": False
        }
    }
}

ALIAS_TO_CANON: Dict[str, str] = {}
for canon, meta in SAMPLE_DRUG_DB.items():
    for a in meta.get("aliases", []):
        ALIAS_TO_CANON[a.lower()] = canon

# -------------------------
# Initialize Watsonx model client if SDK available
watsonx_model = None
if _HAS_WATSONX_SDK:
    try:
        client = APIClient(apikey=WATSONX_API_KEY, service_url=WATSONX_URL)
        watsonx_model = ModelInference(client=client, project_id=WATSONX_PROJECT_ID, model_id=WATSONX_MODEL_ID)
        logger.info("Initialized Watsonx.ai SDK client.")
    except Exception as e:
        logger.exception("Failed to initialize Watsonx SDK client. Will attempt to continue and raise at inference time.")


# -------------------------
# LLM extraction using Watsonx.ai LLM
def llm_extract_drugs_from_text(text: str) -> List[str]:
    """
    Ask the Watsonx model to extract drug names from text.
    Returns canonical names using ALIAS_TO_CANON mapping.
    """
    if not text:
        return []

    # Construct clear extraction prompt with examples
    prompt = (
        "You are a careful medical assistant. "
        "Extract only the drug/medication names mentioned in the text. "
        "Return the drug names as a single comma-separated list, nothing else. "
        "If no drug names appear, return an empty string.\n\n"
        "Examples:\n"
        "Text: 'Patient was prescribed aspirin and paracetamol.'\n"
        "Drugs: aspirin, paracetamol\n\n"
        f"Text: {text}\n"
        "Drugs:"
    )

    # If SDK available, use it; otherwise raise with instructions
    if watsonx_model is None:
        raise RuntimeError(
            "Watsonx.ai model client not initialized. "
            "Ensure package 'ibm-watsonx-ai' is installed and credentials are set."
        )

    try:
        # Different SDK versions have different method names/response formats.
        # We defensively attempt common patterns and parse the result robustly.
        resp = watsonx_model.generate(prompt=prompt, max_tokens=256, temperature=0.0)

        # Possible response shapes: {'results':[{'generated_text': '...'}]} or {'output':[...]} etc.
        raw_text = ""
        if isinstance(resp, dict):
            # Try common keys in order
            if "results" in resp and isinstance(resp["results"], list) and resp["results"]:
                raw_text = resp["results"][0].get("generated_text", "")
            elif "output" in resp and isinstance(resp["output"], list) and resp["output"]:
                # output items may have 'text' or 'generated_text'
                item = resp["output"][0]
                raw_text = item.get("text") or item.get("generated_text") or ""
            elif "generated_text" in resp:
                raw_text = resp.get("generated_text", "")
            else:
                # fallback: stringify whole response
                raw_text = json.dumps(resp)
        elif isinstance(resp, str):
            raw_text = resp
        else:
            raw_text = str(resp)

        # Do a little cleanup
        raw_text = raw_text.strip()
        # Some models may prefix "Drugs:"; remove that
        if raw_text.lower().startswith("drugs:"):
            raw_text = raw_text[6:].strip()
        # Split by common delimiters
        candidates = []
        if raw_text:
            # Often the model returns a comma-separated list; fallback to whitespace-split if needed
            for part in raw_text.replace("\n", ",").split(","):
                p = part.strip().lower()
                if p:
                    # remove trailing punctuation
                    p = p.strip(" .;:").lower()
                    candidates.append(p)
        # Map aliases to canonical
        extracted: List[str] = []
        for alias in candidates:
            canon = ALIAS_TO_CANON.get(alias, alias)
            if canon not in extracted:
                extracted.append(canon)
        return extracted

    except Exception as e:
        logger.exception("Error calling Watsonx.ai model for extraction")
        return []


# -------------------------
# Core functions (same logic as yours)
def check_interactions(drugs: List[str]) -> List[Dict[str, Any]]:
    findings = []
    n = len(drugs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = drugs[i], drugs[j]
            meta_a = SAMPLE_DRUG_DB.get(a, {})
            meta_b = SAMPLE_DRUG_DB.get(b, {})
            sev = None
            if meta_a and meta_b:
                sev = meta_a.get("interactions", {}).get(b)
                if not sev:
                    sev = meta_b.get("interactions", {}).get(a)
            if sev:
                findings.append({
                    "drug_a": a,
                    "drug_b": b,
                    "severity": sev,
                    "message": f"{a} and {b} interaction: {sev}"
                })
    return findings


def recommend_dosage(drug: str, age: Optional[int]) -> Dict[str, Any]:
    meta = SAMPLE_DRUG_DB.get(drug)
    if not meta:
        return {"drug": drug, "error": "Drug not found in database"}
    base = meta["typical_dose_mg"]
    max_daily = meta["max_daily_mg"]
    note = "Standard adult dosing used (demo)."
    dose_mg = base
    if age is not None:
        if age < 12:
            dose_mg = max(1, int(base * (age / 12.0)))
            note = f"Pediatric adjustment applied for age {age}."
        elif age >= 65:
            dose_mg = int(base * 0.8)
            note = f"Elderly precaution applied for age {age}."
    return {
        "drug": drug,
        "recommended_single_dose_mg": dose_mg,
        "typical_adult_single_dose_mg": base,
        "max_daily_mg": max_daily,
        "note": note
    }


def suggest_alternatives(drug: str) -> List[str]:
    meta = SAMPLE_DRUG_DB.get(drug, {})
    return meta.get("alternatives", [])


def check_safety(drug: str, age: int, condition: str, allergy: str) -> Dict[str, Any]:
    meta = SAMPLE_DRUG_DB.get(drug)
    if not meta:
        return {"safe": True, "reason": "", "alternative": "", "side_effects": []}

    contraindications = meta.get("contraindications", {})
    side_effects = meta.get("side_effects", [])

    if age >= 65 and contraindications.get("elderly", False):
        reason = f"{drug.capitalize()} is not safe for elderly patients."
        alt_list = suggest_alternatives(drug)
        alternative = alt_list[0] if alt_list else ""
        return {"safe": False, "reason": reason, "alternative": alternative, "side_effects": side_effects}

    if condition and contraindications.get(condition.lower(), False):
        reason = f"{drug.capitalize()} is not safe for patients with {condition}."
        alt_list = suggest_alternatives(drug)
        alternative = alt_list[0] if alt_list else ""
        return {"safe": False, "reason": reason, "alternative": alternative, "side_effects": side_effects}

    # allergy might be provided as 'penicillin' -> look up 'penicillin allergy'
    if allergy and contraindications.get(allergy.lower() + " allergy", False):
        reason = f"{drug.capitalize()} is contraindicated due to {allergy} allergy."
        alt_list = suggest_alternatives(drug)
        alternative = alt_list[0] if alt_list else ""
        return {"safe": False, "reason": reason, "alternative": alternative, "side_effects": side_effects}

    return {"safe": True, "reason": "", "alternative": "", "side_effects": side_effects}


# -------------------------
# FastAPI backend
app = FastAPI(title="Drug Interaction & Dosage Analyzer (Watsonx.ai LLM)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: Optional[str] = None
    drugs: Optional[List[str]] = None
    age: Optional[int] = None
    condition: Optional[str] = None
    allergy: Optional[str] = None


class AnalyzeResponse(BaseModel):
    extracted_drugs: List[str]
    interactions: List[Dict[str, Any]]
    dosages: Dict[str, Any]
    alternatives: Dict[str, List[str]]
    safety: Dict[str, Any]
    warnings: List[str] = []


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    extracted: List[str] = []
    # If user provided explicit drugs
    if req.drugs:
        for d in req.drugs:
            key = d.strip().lower()
            canon = ALIAS_TO_CANON.get(key, key)
            if canon not in extracted:
                extracted.append(canon)

    # If free text is provided, call LLM extraction
    if req.text:
        try:
            from_text = llm_extract_drugs_from_text(req.text)
        except Exception as e:
            logger.exception("LLM extraction error")
            from_text = []

        for d in from_text:
            if d not in extracted:
                extracted.append(d)

    warnings = []
    if not extracted:
        warnings.append("No known drugs detected in input.")

    interactions = check_interactions(extracted)
    dosages = {d: recommend_dosage(d, req.age) for d in extracted}
    alternatives = {d: suggest_alternatives(d) for d in extracted}
    safety = {}
    if extracted:
        drug = extracted[0]
        safety = check_safety(drug, req.age or 30, (req.condition or "").strip(), (req.allergy or "").strip())

    return AnalyzeResponse(
        extracted_drugs=extracted,
        interactions=interactions,
        dosages=dosages,
        alternatives=alternatives,
        safety=safety,
        warnings=warnings
    )


# -------------------------
# Start API in thread (for local dev convenience)
_api_thread_started = False


def start_api_in_thread():
    global _api_thread_started
    if _api_thread_started:
        return

    def _run():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # small wait so the server comes up before Streamlit tries to call it
    time.sleep(0.7)
    _api_thread_started = True


# -------------------------
# Streamlit frontend (full UI)
start_api_in_thread()
st.set_page_config(page_title="Drug Interaction Analyzer (Watsonx.ai)", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .section-header { font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5rem; color: #111827; }
    .drug-name { font-weight: 700; color: #2563eb; }
    .result-box { background-color: #fef3c7; border-left: 6px solid #f59e0b; padding: 1rem; margin-bottom: 1rem; border-radius: 6px; }
    .safe { color: #16a34a; font-weight: 700; }
    .unsafe { color: #dc2626; font-weight: 700; }
    .footer { font-size: 0.8rem; color: #6b7280; margin-top: 2rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("ℹ About")
    st.write("Analyze drug interactions, dosages & safety. Powered by IBM Watsonx.ai LLM (prompt-based extraction).")
    st.markdown("---")
    st.write("*Usage tips:*")
    st.write("- Paste prescription text or enter drug names separated by commas.")
    st.write("- Specify patient age, condition, and allergy.")
    st.write("- Click 'Analyze' to see results.")
    st.markdown("---")
    st.write("⚠ This demo is for educational use — validate with a clinician before using for medical decisions.")

st.title("💊 Drug Interaction, Dosage & Safety Analyzer (Watsonx.ai LLM)")
col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_area("Prescription text", height=140, placeholder="e.g., Take aspirin 300 mg twice daily and paracetamol as needed.")
with col2:
    drugs_input = st.text_input("Or enter comma-separated drug names", placeholder="aspirin, warfarin")

age = st.number_input("Patient age (years)", min_value=0, max_value=120, value=30)
condition = st.text_input("Condition", placeholder="e.g., kidney disease")
allergy = st.text_input("Allergy", placeholder="e.g., penicillin")

col3, col4 = st.columns(2)
with col3:
    analyze_btn = st.button("Analyze", use_container_width=True)
with col4:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.experimental_rerun()

if analyze_btn:
    payload = {
        "text": text_input if text_input else None,
        "drugs": [d.strip() for d in drugs_input.split(",")] if drugs_input else None,
        "age": int(age),
        "condition": condition.strip() if condition else None,
        "allergy": allergy.strip() if allergy else None
    }

    with st.spinner("Analyzing..."):
        try:
            resp = requests.post("http://127.0.0.1:8000/analyze", json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()

                # Safety
                safety = data.get("safety", {})
                if safety:
                    if safety.get("safe", True):
                        # show safe box if everything ok
                        if data.get("extracted_drugs"):
                            st.markdown(
                                f'<div class="result-box safe">✅ {data["extracted_drugs"][0].capitalize()} appears safe (based on local rules).</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        alt = safety.get("alternative", "")
                        side_effects = ", ".join(safety.get("side_effects", [])) or "N/A"
                        st.markdown(f'<div class="result-box unsafe">❌ {safety["reason"]}</div>', unsafe_allow_html=True)
                        if alt:
                            st.markdown(f"👉 Alternative: *{alt}*")
                        st.markdown(f"⚠ Side Effects: {side_effects}")

                # Detected drugs
                st.markdown('<div class="section-header">🔍 Detected Drugs</div>', unsafe_allow_html=True)
                if data.get("extracted_drugs"):
                    st.write(", ".join(f"{d}" for d in data["extracted_drugs"]))
                else:
                    st.write("No drugs detected.")

                # Interactions
                with st.expander(f"⚠ Interactions ({len(data.get('interactions', []))})", expanded=True):
                    if data.get("interactions"):
                        for it in data["interactions"]:
                            st.error(f"{it['drug_a']}** ↔ *{it['drug_b']}*: {it['severity'].capitalize()} — {it['message']}")
                    else:
                        st.success("No known interactions found.")

                # Dosage
                with st.expander(f"💊 Dosage Recommendations ({len(data.get('dosages', {}))})", expanded=True):
                    for drug, ddata in data.get("dosages", {}).items():
                        st.markdown(f'<span class="drug-name">{drug}</span>', unsafe_allow_html=True)
                        if "error" in ddata:
                            st.error(ddata["error"])
                        else:
                            st.write(f"- Recommended dose: *{ddata['recommended_single_dose_mg']} mg*")
                            st.write(f"- Typical adult dose: *{ddata['typical_adult_single_dose_mg']} mg*")
                            st.write(f"- Max daily: *{ddata['max_daily_mg']} mg*")
                            st.caption(ddata["note"])

                # Alternatives
                with st.expander(f"🔄 Alternatives", expanded=False):
                    for drug, alts in data.get("alternatives", {}).items():
                        st.markdown(f'<span class="drug-name">{drug}</span>', unsafe_allow_html=True)
                        st.write(", ".join(alts) if alts else "No alternatives found.")

                if data.get("warnings"):
                    with st.expander(f"⚠ Warnings ({len(data.get('warnings'))})", expanded=True):
                        for w in data.get("warnings"):
                            st.warning(w)
            else:
                st.error(f"Backend error: HTTP {resp.status_code} — {resp.text}")
        except Exception as e:
            st.error(f"Could not call backend: {e}")

st.markdown('<div class="footer">&copy; 2025 Drug Interaction Analyzer — Powered by Watsonx.ai LLM (demo)</div>', unsafe_allow_html=True)