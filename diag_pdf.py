"""One-off diagnostic: does the PDF extractor recover the Arabic digits?

Run inside the backend container:
    docker compose cp diag_pdf.py backend:/app/diag_pdf.py
    docker compose exec backend python /app/diag_pdf.py
"""

import glob
import os
import re

pdfs = sorted(glob.glob("/app/data/files/*.pdf"), key=os.path.getmtime)
print("PDF files found:", len(pdfs))
f = pdfs[-1] if pdfs else None
print("newest PDF:", f)


def digits(s):
    return re.findall(r"\d[\d.,]*", s or "")


# 1) Is PyMuPDF installed, and does it recover the numbers?
try:
    import fitz  # PyMuPDF
    print("PyMuPDF installed:", fitz.version[0])
    if f:
        text = fitz.open(f)[0].get_text("text")
        print("PyMuPDF digits on page 1:", digits(text)[:25])
        print("PyMuPDF sample:\n", text[:500])
except Exception as e:
    print("PyMuPDF NOT installed ->", repr(e))

# 2) Compare with the old extractor (pypdf).
try:
    from pypdf import PdfReader
    if f:
        text = PdfReader(f).pages[0].extract_text() or ""
        print("pypdf digits on page 1:", digits(text)[:25])
except Exception as e:
    print("pypdf err:", repr(e))
