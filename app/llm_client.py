# app/llm_client.py

import json
import tempfile
import subprocess
import os

def query_llm(full_prompt):
    """
    Sends a prompt to the local LLM API and streams the response word-by-word.
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
