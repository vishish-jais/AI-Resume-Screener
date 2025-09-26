import os
import re
import math
from collections import Counter
from typing import List

from pypdf import PdfReader
from docx import Document
from typing import Optional
try:
    # OCR and PDF-to-image (optional, enabled if installed)
    from pdf2image import convert_from_path  # requires Poppler on Windows
except Exception:
    convert_from_path = None
try:
    import pypdfium2 as pdfium  # cross-platform PDF rendering, no Poppler needed
except Exception:
    pdfium = None
try:
    import pytesseract  # requires Tesseract OCR installed
except Exception:
    pytesseract = None

# Toggle OCR via environment variable (default: disabled)
USE_OCR = os.environ.get('USE_OCR', '0') == '1'
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

# Basic English stopwords list (compact, no external downloads)
STOPWORDS = set('''a about above after again against all am an and any are aren't as at be because been before being below
between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll
i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves'''.split())

SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    # Split by sentence end regex, fallback if whole text becomes empty
    parts = SENTENCE_END_RE.split(text)
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences or ([text] if text else [])


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def sentence_scores(sentences: List[str]):
    # Compute TF-IDF-like scoring with simple IDF approximation
    docs = [set(tokenize(s)) for s in sentences]
    df = Counter()
    for d in docs:
        for w in d:
            if w not in STOPWORDS:
                df[w] += 1
    N = len(sentences) or 1
    scores = []
    for s in sentences:
        tokens = tokenize(s)
        tf = Counter([w for w in tokens if w not in STOPWORDS])
        score = 0.0
        for w, f in tf.items():
            idf = math.log((N + 1) / (1 + df[w])) + 1.0
            score += (f / max(1, len(tokens))) * idf
        scores.append(score)
    return scores


def summarize_text(text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    scores = sentence_scores(sentences)
    # Select top-k sentences, keep original order
    top_idx = sorted(sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:max_sentences])
    summary = " ".join([sentences[i] for i in top_idx])
    return summary

# --- Crisp, structured resume summarizer ---
_SKILL_ALIASES = {
    # Languages
    'python': ['python'],
    'java': ['java'],
    'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'node'],
    'typescript': ['typescript', 'ts'],
    'c#': ['c#', '.net', 'dotnet'],
    'c++': ['c++'],
    'go': ['go', 'golang'],
    'sql': ['sql'],
    'nosql': ['nosql', 'mongo', 'mongodb', 'dynamodb', 'cassandra', 'redis'],
    # Frameworks / Libraries
    'react': ['react', 'react.js', 'reactjs'],
    'angular': ['angular', 'angular.js', 'angularjs'],
    'vue': ['vue', 'vue.js', 'vuejs'],
    'django': ['django'],
    'flask': ['flask'],
    'spring': ['spring', 'spring boot', 'springboot'],
    # Cloud / DevOps
    'aws': ['aws', 'amazon web services'],
    'azure': ['azure'],
    'gcp': ['gcp', 'google cloud'],
    'docker': ['docker'],
    'kubernetes': ['kubernetes', 'k8s'],
    'terraform': ['terraform'],
    'ci/cd': ['ci/cd', 'cicd', 'jenkins', 'github actions', 'gitlab ci'],
    # Data / ML
    'pandas': ['pandas'],
    'numpy': ['numpy'],
    'scikit-learn': ['scikit-learn', 'sklearn'],
    'ml': ['machine learning', 'ml'],
    'nlp': ['nlp', 'natural language processing'],
}

def _approx_years_experience(text: str) -> float:
    """Estimate total years of experience from explicit mentions and employment ranges."""
    import datetime
    t = text.lower()
    years = []
    # Explicit patterns like "X years", "X+ years"
    for m in re.finditer(r"(\d{1,2})\s*\+?\s*(?:\+|plus)?\s*(?:yrs|years|year)\b", t):
        try:
            years.append(float(m.group(1)))
        except Exception:
            pass
    # Ranges like 2018-2022, or "2019 to 2023"
    this_year = datetime.datetime.utcnow().year
    for m in re.finditer(r"(20\d{2}|19\d{2})\s*(?:-|to|–|—)\s*(present|current|20\d{2}|19\d{2})", t):
        start = int(m.group(1))
        end_raw = m.group(2)
        if end_raw in ('present', 'current'):
            end = this_year
        else:
            try:
                end = int(end_raw)
            except Exception:
                end = start
        if end >= start:
            years.append(max(0.0, float(end - start)))
    if not years:
        return 0.0
    # Heuristic: cap outrageous totals, take max as conservative signal
    est = max(years)
    return round(est, 1)


def _extract_strong_skills(text: str, top_n: int = 6) -> list:
    t = text.lower()
    counts = {}
    for canon, aliases in _SKILL_ALIASES.items():
        c = 0
        for a in aliases:
            # word boundary match to avoid false positives
            c += len(re.findall(rf"\b{re.escape(a)}\b", t))
        if c > 0:
            counts[canon] = c
    # Sort by frequency desc, then alphabetically
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered[:top_n]]


def summarize_crisp(text: str, max_skills: int = 6) -> str:
    """
    Produce a concise, recruiter-friendly summary with years of experience and strong skills.
    Fallback to short extractive summary if we cannot detect signals.
    """
    yrs = _approx_years_experience(text)
    skills = _extract_strong_skills(text, top_n=max_skills)

    parts = []
    if yrs > 0:
        parts.append(f"Experience: ~{yrs} years")
    if skills:
        parts.append("Strong in: " + ", ".join(skills))

    if parts:
        return " | ".join(parts)
    # Fallback: extractive summary, short
    return summarize_text(text, max_sentences=3)

# --- Pointwise structured summary ---
_ROLE_KEYWORDS = [
    'software engineer', 'senior software', 'staff engineer', 'principal engineer', 'data scientist',
    'machine learning', 'ml engineer', 'full stack', 'frontend', 'backend', 'devops', 'site reliability', 'sre',
    'product manager', 'project manager', 'qa engineer', 'test engineer', 'security engineer', 'cloud engineer'
]

_DEGREE_PATTERNS = [
    r"\b(b\.?e\.?|btech|b\.tech|bachelor(?:'s)?|bs|bsc)\b",
    r"\b(m\.?e\.?|mtech|m\.tech|master(?:'s)?|ms|msc|mca|mba)\b",
    r"\b(phd|ph\.d\.|doctorate)\b"
]

def _find_roles(text: str, top_n: int = 3) -> list:
    t = text.lower()
    counts = {}
    for kw in _ROLE_KEYWORDS:
        c = len(re.findall(rf"\b{re.escape(kw)}\b", t))
        if c:
            counts[kw] = c
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ordered[:top_n]]


def _find_degrees(text: str, top_n: int = 2) -> list:
    t = text.lower()
    found = []
    for pat in _DEGREE_PATTERNS:
        if re.search(pat, t):
            found.append(re.sub(r"\\\\b|\\\\.", '', pat))
    return found[:top_n]


def _find_highlights(text: str, top_n: int = 4) -> list:
    # Pick sentences with action verbs and metrics
    sentences = split_sentences(text)
    action_verbs = r"(led|built|designed|implemented|developed|optimized|improved|reduced|increased|migrated|automated)"
    metric = r"(\b\d+\s*%|\b\d+\s*(?:k|m|million|billion)|\$\s*\d+)"
    candidates = []
    for s in sentences:
        sc = 0
        if re.search(action_verbs, s, flags=re.I):
            sc += 1
        if re.search(metric, s, flags=re.I):
            sc += 1
        if sc > 0:
            candidates.append((sc, len(s), s.strip()))
    # Sort by score desc, then shorter is better
    ordered = sorted(candidates, key=lambda x: (-x[0], x[1]))
    return [s for _, __, s in ordered[:top_n]]


def summarize_pointwise(text: str) -> str:
    """
    Return a concise bullet list covering the main recruiter signals.
    Bullets: Experience, Strong Skills, Roles, Education, Highlights (up to 4)
    """
    yrs = _approx_years_experience(text)
    skills = _extract_strong_skills(text, top_n=8)
    roles = _find_roles(text)
    degrees = _find_degrees(text)
    highlights = _find_highlights(text)

    bullets = []
    if yrs > 0:
        bullets.append(f"- Experience: ~{yrs} years")
    if skills:
        bullets.append("- Strong skills: " + ", ".join(skills))
    if roles:
        bullets.append("- Roles: " + ", ".join(roles))
    if degrees:
        bullets.append("- Education: " + ", ".join(degrees))
    if highlights:
        bullets.append("- Highlights:")
        for h in highlights:
            bullets.append(f"  • {h}")

    if not bullets:
        return summarize_text(text, max_sentences=3)
    return "\n".join(bullets)


def _extract_companies(text: str, top_n: int = 4) -> list:
    t = text
    # Patterns: "at Company", "@ Company", uppercase styled names, suffixes like Inc, LLC, Ltd, Technologies, Labs
    company_patterns = [
        r"(?:at|@)\s+([A-Z][A-Za-z0-9&\-\. ]{2,})",
        r"\b([A-Z][A-Za-z0-9&\-\. ]+\s+(?:Inc\.|Inc|LLC|Ltd\.|Ltd|Technologies|Labs|Systems|Solutions))\b",
    ]
    candidates = []
    for pat in company_patterns:
        for m in re.finditer(pat, t):
            cand = m.group(1).strip().rstrip('.,;:')
            if len(cand.split()) <= 6:
                candidates.append(cand)
    # Count frequency
    counts = Counter([c for c in candidates if c])
    ordered = [c for c, _ in counts.most_common()]
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for c in ordered:
        k = c.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq[:top_n]


_CERT_PATTERNS = [
    r"aws\s+certified[\w\s\-]*",
    r"azure\s+certified[\w\s\-]*",
    r"gcp\s+certified[\w\s\-]*",
    r"pmp",
    r"scrum\s*master",
    r"cissp",
    r"compTIA\s*(?:security\+|network\+)",
]

def _find_certifications(text: str, top_n: int = 3) -> list:
    t = text.lower()
    found = []
    for pat in _CERT_PATTERNS:
        for m in re.finditer(pat, t):
            found.append(m.group(0))
    # Normalize and dedupe
    norm = []
    seen = set()
    for f in found:
        v = re.sub(r"\s+", " ", f.strip())
        if v not in seen:
            seen.add(v)
            norm.append(v)
    return norm[:top_n]


def summarize_pointwise_structured(text: str) -> tuple:
    """Return (text_bullets, html_list) for pointwise summary with companies and qualifications."""
    yrs = _approx_years_experience(text)
    skills = _extract_strong_skills(text, top_n=8)
    roles = _find_roles(text)
    degrees = _find_degrees(text)
    highlights = _find_highlights(text)
    companies = _extract_companies(text)
    certs = _find_certifications(text)

    bullets = []
    if yrs > 0:
        bullets.append(f"- Experience: ~{yrs} years")
    if companies:
        bullets.append("- Companies: " + ", ".join(companies))
    if skills:
        bullets.append("- Strong skills: " + ", ".join(skills))
    if roles:
        bullets.append("- Roles: " + ", ".join(roles))
    quals = []
    if degrees:
        quals.append("Degrees: " + ", ".join(degrees))
    if certs:
        quals.append("Certifications: " + ", ".join(certs))
    if quals:
        bullets.append("- Qualifications: " + "; ".join(quals))
    if highlights:
        bullets.append("- Highlights:")
        for h in highlights:
            bullets.append(f"  • {h}")

    if not bullets:
        txt = summarize_text(text, max_sentences=3)
        html = f"<p>{re.sub(r'&', '&amp;', txt)}</p>"
        return txt, html

    # Build HTML
    def esc(s: str) -> str:
        return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
    html_lines = [
        '<ul class="ats-summary">'
    ]
    for b in bullets:
        if b.startswith("- Highlights:"):
            html_lines.append('<li><strong>Highlights:</strong><ul>')
        elif b.startswith("  • "):
            html_lines.append(f"<li>{esc(b[4:])}</li>")
        else:
            # Split label and value
            if ': ' in b:
                label, value = b[2:].split(': ', 1)
                html_lines.append(f"<li><strong>{esc(label)}:</strong> {esc(value)}</li>")
            else:
                html_lines.append(f"<li>{esc(b[2:])}</li>")
    # Close nested list if any highlights were added
    if any(b.startswith("  • ") for b in bullets):
        html_lines.append('</ul></li>')
    html_lines.append('</ul>')
    html = "".join(html_lines)
    txt = "\n".join(bullets)
    return txt, html


def _detect_tesseract_cmd() -> str:
    if pytesseract is None:
        return ''
    # 1) Respect environment variable first
    tess_cmd = os.environ.get('TESSERACT_CMD')
    if tess_cmd and os.path.exists(tess_cmd):
        return tess_cmd
    # 2) Try common Windows install location
    common = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    if os.path.exists(common):
        return common
    # 3) Try Program Files (x86)
    common86 = r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    if os.path.exists(common86):
        return common86
    return ''


def _detect_poppler_path() -> str:
    # 1) Respect environment variable first
    pp = os.environ.get('POPPLER_PATH', '')
    if pp and os.path.isdir(pp):
        return pp
    # 2) Try a couple of common locations
    guesses = [
        r"C:\\poppler\\Library\\bin",
        r"C:\\Program Files\\poppler\\Library\\bin",
    ]
    for g in guesses:
        if os.path.isdir(g):
            return g
    return ''


def _render_pdf_with_pdfium(path: str):
    """Render PDF pages to PIL Images using pypdfium2 if available."""
    if pdfium is None:
        return []
    try:
        pdf = pdfium.PdfDocument(path)
        images = []
        # Render at higher scale for better OCR
        for i in range(len(pdf)):
            page = pdf[i]
            pil_image = page.render(scale=2).to_pil()
            images.append(pil_image)
        return images
    except Exception:
        return []


def extract_text_from_pdf(path: str) -> str:
    # 1) Try PyPDF2 first
    pieces = []
    try:
        with open(path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                try:
                    pieces.append(page.extract_text() or '')
                except Exception:
                    continue
    except Exception:
        pieces = []

    text = "\n".join(pieces).strip()
    # 2) Fallback to pdfminer if PyPDF2 gave little or no text
    if (not text or len(text) < 20) and pdfminer_extract_text is not None:
        try:
            text = (pdfminer_extract_text(path) or '').strip()
        except Exception:
            pass

    # If still empty/very short, try OCR fallback (only if enabled)
    if USE_OCR and (not text or len(text) < 20) and pytesseract is not None:
        try:
            # Prefer pdfium rendering if available
            images = _render_pdf_with_pdfium(path)
            if not images and convert_from_path is not None:
                poppler_path = _detect_poppler_path() or None  # e.g., C:\\poppler-xx\\Library\\bin
                images = convert_from_path(path, dpi=300, poppler_path=poppler_path)
            ocr_chunks = []
            # Configure tesseract path if provided
            tess_cmd = _detect_tesseract_cmd()  # e.g., C:\\Program Files\\Tesseract-OCR\\tesseract.exe
            if tess_cmd and pytesseract is not None:
                pytesseract.pytesseract.tesseract_cmd = tess_cmd
            for img in images:
                try:
                    ocr_chunks.append(pytesseract.image_to_string(img) or '')
                except Exception:
                    continue
            text = "\n".join(ocr_chunks).strip()
        except Exception:
            pass

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text_from_image(path: str) -> str:
    """Extract text from a raster image using OCR if available."""
    try:
        from PIL import Image
        if pytesseract is None:
            return ''
        tess_cmd = _detect_tesseract_cmd()
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        img = Image.open(path)
        text = pytesseract.image_to_string(img) or ''
        return re.sub(r"\s+", " ", text.strip())
    except Exception:
        return ''


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(path)
    if ext in ('.docx', '.doc'):
        return extract_text_from_docx(path)
    if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'):
        return extract_text_from_image(path)
    # Fallback: treat as plain text
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return re.sub(r"\s+", " ", content)
    except Exception:
        return ''
