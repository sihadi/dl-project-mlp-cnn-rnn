"""Build a PDF report from report.md and embedded figures (fpdf2, no external tools)."""
from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

DEL = Path(__file__).resolve().parent
REPORT_MD = DEL / "report.md"
PDF_OUT = DEL / "report_generated.pdf"
FIGURES = DEL / "figures"


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _sanitize(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    cleaned = text.encode("latin-1", errors="replace").decode("latin-1")
    return cleaned.replace("\t", " ")


def _wrap_line(text: str, max_len: int = 90) -> list[str]:
    text = _sanitize(text)
    if len(text) <= max_len:
        return [text]
    words = text.split()
    if not words:
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]
    lines = []
    current = ""
    for word in words:
        if len(word) > max_len:
            if current:
                lines.append(current)
                current = ""
            lines.extend(word[i : i + max_len] for i in range(0, len(word), max_len))
            continue
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text[:max_len]]


def _write_lines(pdf: ReportPDF, lines: list[str], h: float = 5) -> None:
    width = pdf.epw
    for chunk in lines:
        if not chunk.strip():
            pdf.ln(h)
            continue
        pdf.multi_cell(width, h, chunk)


def main() -> None:
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Missing report: {REPORT_MD}")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for line in REPORT_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            _write_lines(pdf, _wrap_line(line[2:]), h=8)
            pdf.ln(2)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            _write_lines(pdf, _wrap_line(line[3:]), h=7)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            _write_lines(pdf, _wrap_line(line[4:]), h=6)
            pdf.set_font("Helvetica", size=11)
        elif line.strip().startswith("```"):
            continue
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", line.strip()):
            caption, rel = m.group(1), m.group(2)
            img_path = (DEL / rel).resolve()
            pdf.set_font("Helvetica", "I", 10)
            _write_lines(pdf, _wrap_line(caption), h=5)
            if img_path.exists():
                pdf.image(str(img_path), w=min(170, pdf.epw))
                pdf.ln(3)
            else:
                _write_lines(pdf, [f"[Figure manquante: {img_path.name}]"], h=5)
            pdf.set_font("Helvetica", size=11)
        elif line.strip() == "---":
            pdf.ln(2)
        elif line.strip():
            _write_lines(pdf, _wrap_line(line), h=5)
        else:
            pdf.ln(2)

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_OUT))
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
