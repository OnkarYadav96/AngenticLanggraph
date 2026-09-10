from pathlib import Path
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent
MD = ROOT / "GenAI_AgenticAI_Course_Notes.md"
PDF = ROOT / "GenAI_AgenticAI_Course_Notes.pdf"

def clean(t):
    return t.encode("latin-1", "replace").decode("latin-1")

pdf = FPDF()
pdf.set_auto_page_break(True, 15)
pdf.add_page()
pdf.set_font("Helvetica", size=10)

for line in MD.read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if not s:
        pdf.ln(3)
        continue
    if s.startswith("# "):
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 8, clean(s[2:]))
        pdf.set_font("Helvetica", size=10)
    elif s.startswith("## "):
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 7, clean(s[3:]))
        pdf.set_font("Helvetica", size=10)
    elif s.startswith("### "):
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, clean(s[4:]))
        pdf.set_font("Helvetica", size=10)
    elif s.startswith("```"):
        continue
    else:
        pdf.multi_cell(0, 5, clean(s))

pdf.output(str(PDF))
print("OK", PDF)
