# main.py

from flask import Flask, request, render_template, redirect, url_for
from supabase import create_client, Client
import os
import numpy as np
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ────────── 1) Load environment variables ────────────────────────────────────
# (We will set these in Windows; see Section 5)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Hugging Face Inference API token
# (optionally check that they're present)
if not (SUPABASE_URL and SUPABASE_KEY and HF_API_TOKEN):
    raise RuntimeError("Missing one of: SUPABASE_URL, SUPABASE_KEY, HF_API_TOKEN")

# ────────── 2) Initialize Supabase client ────────────────────────────────────
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ────────── 3) Hugging Face Embedding Function ───────────────────────────────
# We will use the official HF embedding endpoint. You can choose any sentence-transformer model 
# that’s hosted on Hugging Face (e.g., "sentence-transformers/paraphrase-MiniLM-L3-v2").
HF_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
HF_EMBED_URL = f"https://api-inference.huggingface.co/embeddings/{HF_MODEL_ID}"

def get_hf_embedding(text: str) -> list[float]:
    """
    Calls the Hugging Face embeddings endpoint and returns a list of floats.
    Requires env var HF_API_TOKEN.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": text}
    
    response = requests.post(HF_EMBED_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")
    
    data = response.json()
    # For the embeddings endpoint, HF returns {"embedding": [..list of floats..]}
    if "embedding" in data:
        return data["embedding"]
    else:
        # Some endpoints return {"data": [ [..floats..] ]}, but in practice the HF Embedding API 
        # returns a top-level "embedding" key. If yours differs, adjust accordingly.
        raise RuntimeError(f"Unexpected response format from HF: {data}")

# ────────── 4) Flask routing ───────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    """
    On GET:
      - Fetch all entries from Supabase (ordered by created_at ASC)
      - Render the template with original_list only; sorted_list remains empty.

    On POST:
      - Read user_text from form
      - Compute HF embedding
      - Insert into Supabase table `entries`
      - Fetch ALL entries again, build:
          • original_list  (chronological)
          • sorted_list    (semantic order relative to the new entry)
      - Render the template with both lists.
    """
    # 1) Fetch all existing entries from Supabase
    #    We only need text + embedding; order by created_at ascending (oldest→newest).
    supa_resp = (
        supabase
        .table("entries")
        .select("id, text, embedding, created_at")
        .order("created_at", {"ascending": True})
        .execute()
    )
    data = supa_resp.data  # This is a list of dicts: { "id": "...", "text": "...", "embedding": [...], "created_at": "..." }

    original_list = [row["text"] for row in data]
    sorted_list   = []  # Only populate after POST

    if request.method == "POST":
        # 2) Read the form field. We'll have a single input named "user_text".
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            # 3) Compute HF embedding (list of floats)
            new_emb = get_hf_embedding(user_text)

            # 4) Insert into Supabase
            insert_payload = {
                "text": user_text,
                "embedding": new_emb,
                # created_at will default to now()
            }
            supabase.table("entries").insert(insert_payload).execute()

            # 5) Re-fetch ALL entries (so that the newest is at the end)
            supa_resp = (
                supabase
                .table("entries")
                .select("id, text, embedding, created_at")
                .order("created_at", {"ascending": True})
                .execute()
            )
            data = supa_resp.data
            original_list = [row["text"] for row in data]

            # 6) Build semantic sorting
            #    - The “query” is always the last row we just inserted
            query_record = data[-1]
            query_emb = np.array(query_record["embedding"], dtype=float)

            #    - All previous records:
            prev_records = data[:-1]
            if prev_records:
                prev_texts = [r["text"] for r in prev_records]
                prev_embs = np.vstack([np.array(r["embedding"], dtype=float) for r in prev_records])

                # Cosine similarity = (u⋅v) / (||u|| * ||v||)
                query_norm = np.linalg.norm(query_emb)
                prev_norms = np.linalg.norm(prev_embs, axis=1)
                sims = (prev_embs @ query_emb) / (prev_norms * query_norm + 1e-10)

                # Sort indices in descending order of similarity
                idxs = np.argsort(sims)[::-1]

                # Build sorted_list: [ newest text, then prev_texts in order of descending sim ]
                sorted_list = [query_record["text"]] + [prev_texts[i] for i in idxs]
            else:
                # No previous entries => only the new one
                sorted_list = [query_record["text"]]

    # 7) Render template with both lists (if sorted_list is empty, template will show placeholder)
    return render_template(
        "index.html",
        original_list=original_list,
        sorted_list=sorted_list
    )

if __name__ == "__main__":
    # Run Flask on http://127.0.0.1:5000
    app.run(debug=True)
