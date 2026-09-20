# 🏢 HR Policy Assistant

A document-grounded **RAG** app: upload your HR policy PDFs and ask questions. Answers come
**only** from your documents, with **page-level citations**.

**Stack:** Streamlit · FAISS · Sentence Transformers (`all-MiniLM-L6-v2`) · PyMuPDF · Groq (`openai/gpt-oss-20b`)

## Features

| Feature | Details |
|---|---|
| Multi-PDF upload | Several PDFs at once; per-file and total size limits |
| Page-level citations | Every claim cited as `[file.pdf p.12]`, plus retrieved-passage viewer |
| Streaming answers | Tokens appear as they are generated |
| Confidence indicator | High / Medium / Low, based on retrieval similarity |
| Quick insights | One-click cited summaries: leave, benefits, conduct, hours, pay, exit, remote work, grievance, training |
| Semantic search | Meaning-based search with relevance scores |
| Page viewer | Renders the actual PDF page, with optional text highlighting |
| Email drafter | Policy-aware emails to HR (placeholders for unknown details) |
| Analytics | Usage, confidence, latency and most-retrieved pages |
| Chat export | Markdown and JSON |
| Languages | English, اردو (Urdu), Roman Urdu |
| Themes | Works in light and dark mode |

## How it stays honest

1. Retrieval: pages are split into ~900-character passages, embedded, and searched with FAISS (cosine).
2. If the best passage scores below the **Minimum relevance** setting, the app answers "not found" without calling the model.
3. Otherwise the model receives only the retrieved passages and a strict prompt: use only the context, cite every claim, and emit `[[NOT_FOUND]]` if the answer is not there.
4. The confidence badge reflects **how well the retrieved passages match the question**, not proof that the answer is correct. Always verify important decisions with HR.

Because `all-MiniLM-L6-v2` is English-only, Urdu / Roman Urdu questions (and very short follow-ups) are first rewritten into an English search query by the LLM. You can turn this off in the sidebar.

## Limits (designed for Streamlit Community Cloud's ~1 GB RAM)

- 25 MB per PDF, 50 MB total, 600 pages, 5,000 passages per session.
- Scanned / image-only PDFs have no text layer and are rejected with a message. OCR them first.
- Nothing is stored permanently: after a restart or sleep, upload and process the PDFs again.

---

## Deploy with only a web browser (no terminal, Colab or VS Code)

### Step 0 - Get a free Groq API key
1. Go to <https://console.groq.com/keys> and sign in.
2. Click **Create API Key**, name it, and **copy the key** (starts with `gsk_`). You will not see it again.

### Step 1 - Create the GitHub repository
1. Sign in at <https://github.com> (create a free account if needed).
2. Click **+** (top right) → **New repository**.
3. Name it `hr-policy-assistant`. Choose **Public** (simplest for the free Streamlit plan) or Private.
4. Leave the other options as they are and click **Create repository**.

### Step 2 - Add the 4 files (in the repository root)
For each file: **Add file → Create new file**, type the exact file name, paste the contents, then click **Commit changes → Commit directly to main**.

1. `app.py`
2. `requirements.txt`
3. `README.md` (if GitHub already created one, open it and click the ✏️ pencil to replace its contents)
4. `.gitignore` (type the name with the leading dot)

Double-check the names: no `.txt` added by accident (e.g. `requirements.txt.txt`), and the files sit in the top level, not inside a folder.

> **Never** paste your API key into any file in the repository. It goes only into Streamlit Secrets (Step 4).

### Step 3 - Create the Streamlit app
1. Go to <https://share.streamlit.io> and **Continue with GitHub** (authorize access).
2. Click **Create app** → **Deploy a public app from GitHub** (or "Yup, I have an app").
3. Fill in:
   - **Repository:** `your-username/hr-policy-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name (optional)
4. Click **Advanced settings** before deploying (Step 4).

### Step 4 - Python version and Secrets (in Advanced settings)
1. **Python version:** choose **3.11** (best compatibility with FAISS, PyMuPDF and sentence-transformers). Avoid the newest release if wheels are not yet available for it.
2. **Secrets:** paste exactly (keep the quotes):

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

3. Click **Save**, then **Deploy**.

The first build takes roughly 5-10 minutes (PyTorch is large). On first launch the embedding model (~90 MB) downloads once and is cached.

### Step 5 - Use it
Open your app URL → upload PDFs in the sidebar → **Process documents** → ask questions.

### Updating later
- **Code:** open the file on GitHub → ✏️ pencil → edit → **Commit changes**. The app redeploys automatically.
- **Secrets:** in Streamlit Cloud open your app → **⋮ → Settings → Secrets**, edit, save, then **⋮ → Reboot app**.
- **Python version:** if it cannot be changed in Settings, delete the app and redeploy, choosing the version under Advanced settings.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` / "Error installing requirements" | Confirm `requirements.txt` is in the repo root with the exact name. Open the app's **Manage app → logs** to see which package failed. |
| Build fails on `faiss-cpu` or `PyMuPDF` | Use **Python 3.11** (delete and redeploy the app with that setting). |
| Build is very slow or times out on `torch` | Reboot and retry once. If it persists, add `--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt` and `torch` as a new line to fetch the smaller CPU build. |
| "This app has gone over its resource limits" / app restarts | Memory (~1 GB) exhausted. Upload fewer or smaller PDFs, keep total pages under ~300, then **Reboot app**. |
| "The main module file does not exist" | The **Main file path** must be exactly `app.py`, and the file must be in the repo root. |
| Sidebar says no API key | Secrets must contain `GROQ_API_KEY = "..."` (valid TOML, quoted). After editing Secrets, **Reboot app**. Or paste the key in the sidebar box for the current session. |
| "Groq rejected the API key (401)" | Key was copied incorrectly or revoked. Create a new one. |
| "Rate limit reached (429)" | Free tier limit. Wait a minute or lower "Passages to retrieve". |
| Model not found / 400 error | Pick the other model in the sidebar, or check the current model name at console.groq.com/docs/models. |
| "looks like a scanned/image-only PDF" | The PDF has no text layer. Run OCR (e.g. Adobe Acrobat, or upload to Google Drive → Open with Google Docs → download as PDF) and re-upload. |
| Urdu answers show as boxes | Your browser lacks an Arabic-script font, or it is offline; the app loads Noto Naskh Arabic from Google Fonts. |
| First load takes a long time | The model downloads once (~90 MB). Wait, then refresh. |
| App "went to sleep" | Free apps sleep when idle. Click **Yes, get this app back up**, then upload the PDFs again. |
| Answers say "not found" too often | Lower **Minimum relevance** (e.g. 0.20), raise **Passages to retrieve**, or check the PDF pages actually contain text (Page Viewer → "Extracted text"). |

## Project files

```
app.py             # the whole application
requirements.txt   # dependencies
README.md          # this file
.gitignore         # keeps secrets and PDFs out of Git
```
