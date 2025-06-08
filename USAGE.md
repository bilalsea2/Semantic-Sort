# The Flask App

This guide shows how to get the Flask semantic-sorting playground running locally.

---

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/semantic-playground.git
cd semantic-playground
```

## 2. Create & Activate a Virtual Environment

On Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS/Linux (bash):

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Set Environment Variables

Create `.env` file in the project root with:

```env
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_KEY=eyJ...YourAnonKey...
```

*(Replace with your Supabase URL & anon key.)*

## 5. Run the App Locally

```bash
python main.py
```

By default, the app runs at `http://127.0.0.1:5000/`.

## 6. Use the Playground

1. Enter your text in the form: "I am \_\_\_ and I love \_\_\_".
2. Submit to see both the **Original Order** and the **Semantic Order** (closest matches).
3. Use the dropdown to pick any sentence as your similarity query.
4. Hover over a sentence to reveal the remove (✖) button and delete entries.

---

Enjoy!
