"""Generate PDF from GenAI_AgenticAI_Course_Notes.md using fpdf2."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF


ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "GenAI_AgenticAI_Course_Notes.md"
PDF_PATH = ROOT / "GenAI_AgenticAI_Course_Notes.pdf"


class NotesPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "GenAI & Agentic AI Course Notes (Krish Naik)", align="C")
            self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def sanitize(text: str) -> str:
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2192": "->",
        "\u2190": "<-",
        "\u2705": "[OK]",
        "\u274c": "[X]",
        "\u26a0": "[!]",
        "\U0001f4a1": "[Tip]",
        "\U0001f680": "",
        "\U0001f916": "",
        "\U0001f9e0": "",
        "\U0001f4ca": "",
        "\U0001f6e1": "",
        "\U0001f50d": "",
        "\U0001f4c8": "",
        "\U0001f512": "",
        "\U0001f9ed": "",
        "\U0001f4da": "",
        "\U0001f3af": "",
        "\u2b50": "*",
        "\u2713": "[v]",
        "\u2714": "[v]",
        "\u2716": "[x]",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def write_wrapped(pdf: NotesPDF, text: str, size: int = 10, style: str = "") -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5.5, sanitize(text))
    pdf.ln(1)


def render_markdown(md: str, pdf: NotesPDF) -> None:
    lines = md.splitlines()
    in_code = False
    code_buf: list[str] = []
    in_mermaid = False

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    for raw in lines:
        line = raw.rstrip()

        if line.strip().startswith("```mermaid"):
            in_mermaid = True
            continue
        if in_mermaid:
            if line.strip() == "```":
                in_mermaid = False
                write_wrapped(pdf, "[See diagram in markdown source file]", size=9, style="I")
            else:
                write_wrapped(pdf, line, size=8, style="")
            continue

        if line.strip().startswith("```"):
            if in_code:
                block = "\n".join(code_buf)
                pdf.set_fill_color(245, 245, 245)
                pdf.set_font("Courier", "", 8)
                pdf.multi_cell(0, 4.5, sanitize(block), fill=True)
                pdf.ln(2)
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_buf.append(line)
            continue

        if not line.strip():
            pdf.ln(2)
            continue

        if line.startswith("# "):
            pdf.ln(4)
            write_wrapped(pdf, line[2:].strip(), size=18, style="B")
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.ln(3)
            write_wrapped(pdf, line[3:].strip(), size=14, style="B")
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.ln(2)
            write_wrapped(pdf, line[4:].strip(), size=12, style="B")
        elif line.startswith("#### "):
            write_wrapped(pdf, line[5:].strip(), size=11, style="B")
        elif line.startswith("---"):
            pdf.ln(2)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(3)
        elif line.startswith("|"):
            write_wrapped(pdf, re.sub(r"\s*\|\s*", " | ", line), size=8)
        elif line.startswith("- ") or line.startswith("* "):
            write_wrapped(pdf, "  - " + line[2:].strip(), size=10)
        elif re.match(r"^\d+\.\s", line):
            write_wrapped(pdf, "  " + line.strip(), size=10)
        else:
            write_wrapped(pdf, line, size=10)


def main() -> int:
    if not MD_PATH.exists():
        print(f"Markdown not found: {MD_PATH}", file=sys.stderr)
        return 1

    pdf = NotesPDF()
    pdf.set_margins(15, 15, 15)
    render_markdown(MD_PATH.read_text(encoding="utf-8"), pdf)
    pdf.output(str(PDF_PATH))
    print(f"PDF written to: {PDF_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
