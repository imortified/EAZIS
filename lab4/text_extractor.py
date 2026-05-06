import os
import subprocess
import tempfile
from docx import Document
import PyPDF2
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
def extract_text_subprocess_libreoffice(filepath):
    try:
        soffice_path = r"C:\Program Files\LibreOffice\program\soffice.exe"
        if not os.path.exists(soffice_path):
            alt_paths = [
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                r"C:\ProgramData\chocolatey\lib\libreoffice\tools\LibreOffice\program\soffice.exe"
            ]
            for alt in alt_paths:
                if os.path.exists(alt):
                    soffice_path = alt
                    break
            else:
                return None

        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [soffice_path, "--headless", "--convert-to", "txt:Text",
                 "--outdir", tmpdir, filepath],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                txt_file = os.path.join(tmpdir, os.path.splitext(os.path.basename(filepath))[0] + ".txt")
                if os.path.exists(txt_file):
                    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read().strip()
        return None
    except Exception as e:
        print(f"LibreOffice error: {e}")
        return None

def extract_text_from_docx(filepath):
    try:
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"DOCX error: {e}")
        return None

def extract_text_from_txt(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"TXT error: {e}")
        return None

def extract_text_from_pdf(filepath):
    try:
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        print(f"PDF error: {e}")
        return None

def extract_text_from_html(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")
    except Exception as e:
        print(f"HTML error: {e}")
        return None

def extract_text_from_rtf(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return rtf_to_text(f.read())
    except Exception as e:
        print(f"RTF error: {e}")
        return None

def extract_text(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".doc":
        return extract_text_subprocess_libreoffice(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext in (".html", ".htm"):
        return extract_text_from_html(filepath)
    elif ext == ".rtf":
        return extract_text_from_rtf(filepath)
    else:
        return None
