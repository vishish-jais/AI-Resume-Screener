from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime

# Chatbot summarizer imports
from summarizer import (
    extract_text_from_file,
    summarize_text,
    summarize_pointwise_structured,
    summarize_crisp,
    compute_ats_score,
    compute_ats_keywords,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-change'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

db = SQLAlchemy(app)


class HR(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(30))

    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)


class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    company = db.Column(db.String(200), nullable=False)
    tags = db.Column(db.String(200), default='')
    description = db.Column(db.Text, default='')  # JD text


class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('candidate.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    status = db.Column(db.String(50), default='New')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    resume_filename = db.Column(db.String(255))
    ats_score = db.Column(db.Integer, default=None)  # cached ATS score vs job description

    candidate = db.relationship('Candidate', backref=db.backref('applications', lazy=True))
    job = db.relationship('Job', backref=db.backref('applications', lazy=True))


@app.cli.command('init-db')
def init_db_command():
    db.drop_all()
    db.create_all()
    # seed HR
    hr = HR(username='hr')
    hr.set_password('hr123')
    db.session.add(hr)

    # seed candidates
    cand = Candidate(username='cand')
    cand.set_password('cand123')
    cand.name = 'Sample Candidate'
    cand.email = 'cand@example.com'
    cand.phone = '1234567890'
    db.session.add(cand)

    # seed jobs
    job = Job(title='Senior Frontend Developer', company='Company XYZ', tags='React,TypeScript')
    db.session.add(job)

    db.session.commit()
    print('Database initialized with sample data.')


@app.route('/')
def index():
    return render_template('index.html')


# -------------------- Auth --------------------
@app.route('/login/hr', methods=['GET', 'POST'])
def login_hr():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        hr = HR.query.filter_by(username=username).first()
        if hr and hr.check_password(password):
            session.clear()
            session['hr_id'] = hr.id
            return redirect(url_for('hr_dashboard'))
        flash('Invalid HR credentials', 'danger')
    return render_template('login_hr.html')


@app.route('/login/candidate', methods=['GET', 'POST'])
def login_candidate():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = Candidate.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session.clear()
            session['candidate_id'] = user.id
            return redirect(url_for('candidate_dashboard'))
        flash('Invalid Candidate credentials', 'danger')
    return render_template('login_candidate.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# -------------------- HR Portal --------------------
def require_hr():
    if 'hr_id' not in session:
        return redirect(url_for('login_hr'))
    return None


@app.route('/hr')
def hr_dashboard():
    guard = require_hr()
    if guard:
        return guard
    total_candidates = Candidate.query.count()
    open_positions = Job.query.count()
    hired = Application.query.filter_by(status='Hired').count()
    in_interview = Application.query.filter_by(status='Interview').count()
    recent = Application.query.order_by(Application.created_at.desc()).limit(5).all()
    status_counts = {
        'New': Application.query.filter_by(status='New').count(),
        'Interview': in_interview,
        'Hired': hired,
        'Rejected': Application.query.filter_by(status='Rejected').count(),
        'Reviewed': Application.query.filter_by(status='Reviewed').count(),
    }
    return render_template(
        'hr_dashboard.html',
        total_candidates=total_candidates,
        open_positions=open_positions,
        hired=hired,
        in_interview=in_interview,
        recent=recent,
        status_counts=status_counts,
    )


@app.route('/hr/resumes')
def hr_resumes():
    guard = require_hr()
    if guard:
        return guard
    apps = Application.query.order_by(Application.created_at.desc()).all()
    return render_template('hr_resumes.html', applications=apps)


STATUS_OPTIONS = ['New', 'Interview', 'Reviewed', 'Hired', 'Rejected']


@app.route('/hr/candidates')
def hr_candidates():
    guard = require_hr()
    if guard:
        return guard
    apps = Application.query.order_by(Application.created_at.desc()).all()
    return render_template('hr_candidates.html', applications=apps, status_options=STATUS_OPTIONS)


@app.route('/hr/jobs')
def hr_jobs():
    guard = require_hr()
    if guard:
        return guard
    jobs = Job.query.order_by(Job.id.desc()).all()
    return render_template('hr_jobs.html', jobs=jobs)


@app.route('/hr/screening', methods=['GET', 'POST'])
def hr_screening():
    guard = require_hr()
    if guard:
        return guard

    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_file = request.files.get('resume')

        if not job_description or not resume_file:
            flash('Please provide both job description and resume', 'error')
            return redirect(url_for('hr_screening'))

        if resume_file:
            filename = secure_filename(resume_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(filepath)
            flash('Resume uploaded successfully!', 'success')

    apps = Application.query.order_by(Application.created_at.desc()).limit(10).all()
    return render_template('hr_screening.html', applications=apps)


@app.post('/hr/application/<int:app_id>/status')
def hr_update_status(app_id: int):
    guard = require_hr()
    if guard:
        return guard
    new_status = request.form.get('status')
    app_row = Application.query.get_or_404(app_id)
    if new_status in STATUS_OPTIONS:
        app_row.status = new_status
        db.session.commit()
        flash('Status updated', 'success')
    else:
        flash('Invalid status', 'danger')
    return redirect(request.referrer or url_for('hr_candidates'))


@app.route('/uploads/<path:filename>')
def download_resume(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


# -------------------- Candidate Portal --------------------
def require_candidate():
    if 'candidate_id' not in session:
        return redirect(url_for('login_candidate'))
    return None


@app.route('/candidate')
def candidate_dashboard():
    guard = require_candidate()
    if guard:
        return guard
    user = Candidate.query.get(session['candidate_id'])
    jobs = Job.query.all()
    my_apps = Application.query.filter_by(candidate_id=user.id).order_by(Application.created_at.desc()).all()
    return render_template('candidate_dashboard.html', user=user, jobs=jobs, my_apps=my_apps)


@app.route('/jobs')
def public_jobs():
    jobs = Job.query.all()
    return render_template('jobs_public.html', jobs=jobs)


@app.route('/apply/<int:job_id>', methods=['GET', 'POST'])
def apply(job_id: int):
    job = Job.query.get_or_404(job_id)
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')

        # Find or create candidate by email
        user = Candidate.query.filter_by(email=email).first()
        if not user:
            username = email or f"user{datetime.utcnow().timestamp()}"
            user = Candidate(username=username)
            user.set_password(os.urandom(8).hex())
            user.name = name
            user.email = email
            user.phone = phone
            db.session.add(user)
            db.session.flush()  # get id
        else:
            user.name = name or user.name
            user.phone = phone or user.phone

        resume_file = request.files.get('resume')
        filename = None
        if resume_file and resume_file.filename:
            safe_name = secure_filename(f"{user.username}_{job.id}_{resume_file.filename}")
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            resume_file.save(resume_path)
            filename = safe_name

        app_row = Application(candidate_id=user.id, job_id=job.id, status='New', resume_filename=filename)
        db.session.add(app_row)
        db.session.commit()

        # After commit we have IDs; compute ATS score if possible
        try:
            if filename and (job.description or '').strip():
                resume_text = extract_text_from_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Use keyword-only scoring for submissions
                ats = compute_ats_keywords(resume_text, job.description)
                app_row.ats_score = ats.get('score', 0)
                db.session.commit()
        except Exception:
            pass
        # flash('Application submitted successfully.', 'success')
        return redirect(url_for('index'))

    return render_template('apply.html', job=job)


# -------------------- Chatbot API (integrated) --------------------

def _is_request_authorized() -> bool:
    # Allow if HR is logged in OR presenting a valid widget token.
    if session.get('hr_id'):
        return True
    expected = os.environ.get('WIDGET_TOKEN', '')
    if not expected:
        return False
    token = (
        request.headers.get('X-Widget-Token')
        or request.args.get('token')
        or (request.form.get('token') if request.form else None)
    )
    return token == expected


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    if not _is_request_authorized():
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401

    text = request.form.get('text', '').strip()
    file = request.files.get('file')

    if not text and not file:
        return jsonify({'ok': False, 'error': 'Provide resume text or upload a file.'}), 400

    try:
        if file:
            filename = secure_filename(file.filename)
            if not filename:
                return jsonify({'ok': False, 'error': 'Invalid file name.'}), 400
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            extracted = extract_text_from_file(save_path)
            text_to_summarize = extracted
        else:
            text_to_summarize = text

        if not text_to_summarize.strip():
            hint = (
                'No extractable text found in the resume. '
                'If this is a scanned PDF or an image, enable OCR by installing Tesseract and Poppler, '
                'then set environment variables POPPLER_PATH (to Poppler bin) and TESSERACT_CMD (to tesseract.exe).'
            )
            return jsonify({'ok': False, 'error': hint}), 400

        style = (request.form.get('style') or 'pointwise').lower()
        if style == 'crisp':
            summary = summarize_crisp(text_to_summarize)
            return jsonify({'ok': True, 'summary': summary})
        elif style == 'detailed':
            summary = summarize_text(text_to_summarize, max_sentences=6)
            return jsonify({'ok': True, 'summary': summary})
        else:
            summary_text, summary_html = summarize_pointwise_structured(text_to_summarize)
            return jsonify({'ok': True, 'summary': summary_text, 'summary_html': summary_html})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/summarize_batch', methods=['POST'])
def api_summarize_batch():
    if not _is_request_authorized():
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        style = (request.form.get('style') or 'pointwise').lower()
        files = request.files.getlist('files') or request.files.getlist('file')
        if not files:
            return jsonify({'ok': False, 'error': 'No files provided.'}), 400

        results = []
        for f in files:
            filename = secure_filename(f.filename)
            if not filename:
                results.append({'filename': '(unknown)', 'ok': False, 'error': 'Invalid file name.'})
                continue
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                f.save(save_path)
                text_content = extract_text_from_file(save_path)
                if not text_content or not text_content.strip():
                    results.append({'filename': filename, 'ok': False, 'error': 'No extractable text found (possibly scanned image).'})
                    continue
                if style == 'crisp':
                    summary = summarize_crisp(text_content)
                    results.append({'filename': filename, 'ok': True, 'summary': summary})
                elif style == 'detailed':
                    summary = summarize_text(text_content, max_sentences=6)
                    results.append({'filename': filename, 'ok': True, 'summary': summary})
                else:
                    summary_text, summary_html = summarize_pointwise_structured(text_content)
                    results.append({'filename': filename, 'ok': True, 'summary': summary_text, 'summary_html': summary_html})
            except Exception as ex:
                results.append({'filename': filename, 'ok': False, 'error': str(ex)})

        return jsonify({'ok': True, 'results': results})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/ats_score', methods=['POST'])
def api_ats_score():
    if not _is_request_authorized():
        return jsonify({'ok': False, 'error': 'Unauthorized'}), 401
    try:
        jd_text = request.form.get('job_description', '')
        job_id = request.form.get('job_id')
        resume_file = request.files.get('resume')
        application_id = request.form.get('application_id')
        # Default to keyword-only unless explicitly requested otherwise
        mode = (request.form.get('mode') or 'keywords').lower().strip()

        # Resolve JD text
        if not jd_text and job_id:
            job = Job.query.get(int(job_id)) if job_id else None
            jd_text = (job.description or '') if job else ''

        if application_id and not resume_file:
            app_row = Application.query.get(int(application_id))
            if not app_row or not app_row.resume_filename:
                return jsonify({'ok': False, 'error': 'Application/resume not found'}), 400
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], app_row.resume_filename)
            resume_text = extract_text_from_file(resume_path)
        else:
            if not resume_file:
                return jsonify({'ok': False, 'error': 'Provide resume file or application_id'}), 400
            filename = secure_filename(resume_file.filename)
            if not filename:
                return jsonify({'ok': False, 'error': 'Invalid file name'}), 400
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            resume_file.save(save_path)
            resume_text = extract_text_from_file(save_path)

        if not jd_text.strip() or not resume_text.strip():
            return jsonify({'ok': False, 'error': 'Empty JD or resume text'}), 400

        # Try to get job title for better title alignment
        job_title = None
        try:
            if job_id:
                j = Job.query.get(int(job_id))
                job_title = j.title if j else None
        except Exception:
            job_title = None

        if mode == 'keywords':
            ats = compute_ats_keywords(resume_text, jd_text)
        else:
            ats = compute_ats_score(resume_text, jd_text, job_title=job_title)
        rs_txt, rs_html = summarize_pointwise_structured(resume_text)
        jd_txt, jd_html = summarize_pointwise_structured(jd_text)
        return jsonify({'ok': True, 'ats': ats, 'resume_summary': rs_txt, 'resume_summary_html': rs_html, 'jd_summary': jd_txt, 'jd_summary_html': jd_html})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Lightweight migrations for new columns
        from sqlalchemy import text as _sql_text
        try:
            # Add Job.description if missing
            cols = db.session.execute(_sql_text("PRAGMA table_info(job)")).fetchall()
            names = [c[1] for c in cols]
            if 'description' not in names:
                db.session.execute(_sql_text('ALTER TABLE job ADD COLUMN description TEXT DEFAULT ""'))
                db.session.commit()
        except Exception:
            pass
        try:
            # Add Application.ats_score if missing
            cols = db.session.execute(_sql_text("PRAGMA table_info(application)")).fetchall()
            names = [c[1] for c in cols]
            if 'ats_score' not in names:
                db.session.execute(_sql_text('ALTER TABLE application ADD COLUMN ats_score INTEGER'))
                db.session.commit()
        except Exception:
            pass
        # Auto-seed minimal data on first run for convenience
        if HR.query.count() == 0:
            hr = HR(username='hr')
            hr.set_password('hr123')
            db.session.add(hr)
        if Job.query.count() == 0:
            job = Job(title='Senior Frontend Developer', company='Company XYZ', tags='React,TypeScript', description='We are seeking a Senior Frontend Developer with strong React and TypeScript skills. 5+ years experience preferred. Experience with CI/CD and cloud is a plus.')
            db.session.add(job)
        db.session.commit()
    # Allow configurable host/port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
