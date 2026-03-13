import subprocess
import re


def clean_text(text: str) -> str:
    """
    Clean extra spaces and formatting from generated text.
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def query_ollama(prompt, model="gemma:2b"):
    """
    Send prompt to Ollama and return generated response.
    """
    process = None
    try:
        command = ["ollama", "run", model]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        stdout, stderr = process.communicate(prompt, timeout=120)

        if process.returncode not in (0, None):
            print(f"[WARN] Ollama exited with code {process.returncode}: {stderr.strip()}")

        return (stdout or "").strip()

    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        print("[WARN] Ollama request timed out")
        return "LLM explanation unavailable."
    except Exception as e:
        print(f"[WARN] Failed to query Ollama: {e}")
        return "LLM explanation unavailable."


def llm_reasoning(prediction, confidence, visual_evidence, heatmap_summary):
    """
    Generate explanation using Ollama LLM.
    """
    confidence_pct = confidence * 100

    prompt = f"""
You are an AI forensic analyst.

Prediction: {prediction}
Confidence: {confidence_pct:.1f}%

Visual Evidence:
{visual_evidence}

Heatmap Summary:
{heatmap_summary}

Explain clearly and concisely why this image is classified as real or fake.
Mention visual cues and attention regions in your explanation.
"""

    explanation = query_ollama(prompt)
    explanation = clean_text(explanation)

    return explanation
