# HR Dashboard + Chatbot (Integrated)

This is a merged Flask app combining the HR Dashboard from `Hr-Dashboard` and the resume summarizer chatbot widget from `Chatbot1`.

- HR features: candidate/job tracking, resume uploads, simple dashboards.
- Chatbot: bottom-right widget to summarize pasted text or uploaded resumes, plus batch summarization.

## Project structure

- `app.py` — main Flask app (HR routes + Chatbot API endpoints)
- `summarizer.py` — text extraction and summarization helpers (PDF/DOCX/Image/OCR fallback)
- `templates/` — HTML templates for HR and Candidate portals (chatbot auto-loaded in HR pages)
- `static/js/ats-embed.js` — loads widget assets and config
- `static/js/widget.js` — UI and client logic for summarizer (calls `/api/summarize*`)
- `static/css/styles.css` — widget styles
- `uploads/` — created at runtime for resume files
- `app.db` — SQLite DB (auto-created)

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

Seed credentials (auto-created on first run):
- HR: username `hr`, password `hr123`
- Candidate: username `cand`, password `cand123`

## Using the Chatbot Widget

On HR pages (e.g., `Dashboard`, `Candidates`, `Resume Screening`), you’ll see a circular button bottom-right. Click to open the widget. You can:
- Paste resume text and choose a style: `Pointwise`, `Crisp`, `Detailed`.
- Upload a file (`PDF`, `DOCX`, `TXT`, or an image for OCR if enabled) and summarize.
- Batch "Scan All" for multiple files at once.

Backend endpoints:
- `POST /api/summarize` — accepts `text` or `file`, optional `style`.
- `POST /api/summarize_batch` — accepts multiple `files`, optional `style`.

Authorization:
- Allowed when an HR user is logged in.
- Or set an env var `WIDGET_TOKEN` and include header `X-Widget-Token` for token-based access.

## Optional OCR for scanned PDFs/images

OCR is disabled by default. To enable OCR:

- Install Tesseract (Windows):
  - https://github.com/UB-Mannheim/tesseract/wiki
- (Optional) Install Poppler (for `pdf2image`), then set `POPPLER_PATH` to its `bin` folder.
- Set environment variables before running the app:

```powershell
$env:USE_OCR = "1"                # enable OCR fallback
$env:TESSERACT_CMD = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
$env:POPPLER_PATH = "C:\\poppler\\Library\\bin"
```

## Notes

- This is a dev setup. For production, change `SECRET_KEY`, run behind HTTPS, and secure file handling.
- The widget can be embedded on other websites by pointing the `ats-embed.js` `data-api-base` to this server and optionally passing `data-token` if `WIDGET_TOKEN` is set.
