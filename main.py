import os
from flask import Flask, request, render_template, redirect, url_for
from sentence_transformers import SentenceTransformer
import numpy as np
from supabase import create_client
from dotenv import load_dotenv
import json

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = Flask(__name__)
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")


def fetch_all_records():
    resp = supabase.table("entries").select("id, text, embedding").order("id", desc=False).execute()
    data = None
    error = None
    if isinstance(resp, str):
        payload = json.loads(resp)
        data = payload.get("data", [])
        error = payload.get("error")
    elif isinstance(resp, dict):
        data = resp.get("data", [])
        error = resp.get("error")
    else:
        data = getattr(resp, "data", [])
        error = getattr(resp, "error", None)
    if error:
        raise RuntimeError(f"Supabase fetch error: {error}")
    return data


def insert_record(text: str, embedding: list[float]):
    resp = supabase.table("entries").insert({"text": text, "embedding": embedding}).execute()
    data = None
    error = None
    if isinstance(resp, str):
        payload = json.loads(resp)
        data = payload.get("data", [])
        error = payload.get("error")
    elif isinstance(resp, dict):
        data = resp.get("data", [])
        error = resp.get("error")
    else:
        data = getattr(resp, "data", [])
        error = getattr(resp, "error", None)
    if error:
        raise RuntimeError(f"Supabase insert error: {error}")
    if not data:
        raise RuntimeError("Insert succeeded but no data returned.")
    return data[0]


def delete_record(record_id: int):
    # Deletes by id
    resp = supabase.table("entries").delete().eq("id", record_id).execute()
    # similar response handling
    if hasattr(resp, 'error') and resp.error:
        raise RuntimeError(f"Supabase delete error: {resp.error}")
    if isinstance(resp, dict) and resp.get("error"):
        raise RuntimeError(f"Supabase delete error: {resp.get('error')}")


@app.route("/", methods=["GET", "POST"])
def index():
    data = fetch_all_records()
    original_list = data.copy()  # list of dicts
    sorted_list = []
    selected_id = request.values.get("query_id", type=int)

    # Handle new insertion
    if request.method == "POST" and request.form.get("part1"):
        part1 = request.form.get("part1", "").strip()
        part2 = request.form.get("part2", "").strip()
        if part1 and part2:
            user_text = f"I am {part1} and I LOVE {part2}"
            emb = model.encode([user_text])[0].tolist()
            new_record = insert_record(user_text, emb)
            return redirect(url_for('index', query_id=new_record['id']))

    # If a query_id is provided (or fallback to last)
    if selected_id is None and original_list:
        selected_id = original_list[-1]['id']

    if selected_id:
        # find the query record
        query_items = [r for r in data if r['id'] == selected_id]
        if query_items:
            query_rec = query_items[0]
            query_emb = np.array(query_rec['embedding'])
            # all others
            others = [r for r in data if r['id'] != selected_id]
            if others:
                texts = [r['text'] for r in others]
                embs = np.vstack([np.array(r['embedding']) for r in others])
                q_norm = np.linalg.norm(query_emb)
                norms = np.linalg.norm(embs, axis=1)
                sims = (embs @ query_emb) / (norms * q_norm + 1e-10)
                idxs = np.argsort(sims)[::-1]
                sorted_list = [query_rec] + [others[i] for i in idxs]
            else:
                sorted_list = [query_rec]

    return render_template(
        "index.html",
        original_list=original_list,
        sorted_list=sorted_list,
        selected_id=selected_id
    )

@app.route("/delete/<int:record_id>", methods=["POST"])
def delete(record_id):
    delete_record(record_id)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)