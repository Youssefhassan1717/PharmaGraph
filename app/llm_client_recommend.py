# app/llm_client_recommend.py

import json
import tempfile
import subprocess
import os

def query_llm_for_recommendation(diagnosis_prompt, forbidden_drugs):
    """
    Sends a diagnosis prompt and forbidden drug list to the local LLM API and streams the response word-by-word.
    """
    max_forbidden = 10
    trimmed_forbidden_drugs = forbidden_drugs[:max_forbidden]

    full_prompt = f"""
You are a certified medical assistant AI.

Patient Diagnosis:
{diagnosis_prompt}

Current Medications (Forbidden List):
{', '.join(trimmed_forbidden_drugs)}

IMPORTANT INSTRUCTIONS:
- Recommend exactly 1–2 safe FDA-approved drugs that can treat the given diagnosis.
- DO NOT recommend any drug from the forbidden list above.
- If no safe drug exists, clearly respond: "No safe drug found."
- Do NOT explain medical background. Do NOT tell stories.
- Only output the recommended drug names and a very brief reason (1 line maximum).
- Summarize your recommendation clearly and briefly.
- Always advise consulting a healthcare professional before starting any new medications.

TASK:
Please recommend a safe medication and briefly explain why it is appropriate.
"""

    data = {
        "model": "gemma:7b-instruct-v1.1-q4_0",
        "stream": True,
        "messages": [
            {"role": "user", "content": full_prompt}
        ]
    }

    json_data = json.dumps(data, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", encoding="utf-8") as temp_json:
        temp_json.write(json_data)
        temp_json_path = temp_json.name

    cmd = [
        "curl", "--no-buffer", "-X", "POST",
        "https://272b-34-34-1-154.ngrok-free.app/api/chat",
        "-H", "Content-Type: application/json",
        "-d", f"@{temp_json_path}"
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        word = data["message"]["content"]
                        yield word
                except json.JSONDecodeError:
                    continue

        process.stdout.close()
        process.wait()
        os.remove(temp_json_path)

    except Exception as e:
        os.remove(temp_json_path)
        yield f"Error: {e}"
