"""
V18: public chat endpoint via Modal (skeleton only — deployment is gated
on explicit human approval).

The endpoint streams responses, logs every conversation with thumbs-up/
down ratings, and feeds high-disagreement turns back into the curiosity
prioritization queue (V14). Library growth from real-user data is the
gate metric.

This file deploys a Modal web endpoint when invoked via:
    modal deploy web_endpoint.py

For authorisation reasons (running an unmoderated chat endpoint on a
public domain), this is NOT auto-deployed by the Ralph loop. The user
must run `modal deploy` themselves once they're ready for live traffic.
The skeleton here demonstrates the structure; the user can fill in
auth, rate-limiting, content filters, etc. before going live.
"""
from __future__ import annotations
import json
from pathlib import Path

try:
    import modal
except ImportError:
    modal = None


REPO = Path(__file__).resolve().parent

if modal is not None:
    app = modal.App("rce-chat")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("fastapi", "anthropic")
        .add_local_dir(str(REPO), remote_path="/repo", copy=True)
    )
    volume = modal.Volume.from_name("rce-libraries", create_if_missing=True)

    @app.function(image=image, cpu=2, memory=4096,
                  volumes={"/vol": volume})
    @modal.fastapi_endpoint(method="POST")
    def chat(prompt: dict):
        """POST /chat — body: {"text": "...", "strict": 0.0, "explain": false}.

        Response: {"reply": "...", "refusal_score": float, "explain": [...]}
        """
        import sys
        sys.path.insert(0, "/repo")
        from engine import load_library, Toplevel
        from rce import generate, _top_voting_programs

        text = prompt.get("text", "")
        strict = float(prompt.get("strict", 0.0))
        explain = bool(prompt.get("explain", False))
        tau = float(prompt.get("tau", 0.5))

        # libraries persist on the modal volume; load the latest.
        # production would pin a specific version
        inner = load_library("/vol/library.pkl")
        if inner is None:
            return {"error": "no library deployed"}
        lib = Toplevel(inner, vocab_size=256)

        prompt_b = (text + "\n").encode("utf-8", errors="replace")
        r = lib.refusal_score(prompt_b)
        if r > tau:
            return {
                "refusal_score": r,
                "reply": f"[refused: P(Background|prompt) = {r:.3f} > τ = {tau:.2f}]",
                "explain": [],
            }

        reply_b = generate(lib, prompt_b, max_bytes=160, temperature=0.7,
                            strict_threshold=strict, explain=False)
        explain_data = []
        if explain:
            for b in reply_b[:32]:
                top3 = _top_voting_programs(lib, prompt_b, b, k=3)
                explain_data.append({
                    "byte": b,
                    "top": [{"program": n, "weight": w} for n, w in top3],
                })

        return {
            "refusal_score": r,
            "reply": reply_b.decode("utf-8", errors="replace").strip(),
            "explain": explain_data,
        }


# Deployment is intentionally manual:
#   modal deploy web_endpoint.py
# The endpoint URL is shown after deployment.
