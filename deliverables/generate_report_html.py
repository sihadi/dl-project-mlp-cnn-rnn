"""Convert deliverables/report.md to a self-contained HTML report."""
from __future__ import annotations

import base64
import re
from pathlib import Path

DEL = Path(__file__).resolve().parent
REPORT_MD = DEL / "report.md"
REPORT_HTML = DEL / "report.html"
FIGURES = DEL / "figures"


def _embed_image(path: Path) -> str:
    if not path.exists():
        return f'<p><em>Figure manquante: {path.name}</em></p>'
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{path.name}" style="max-width:100%;">'


def md_to_html(md_text: str) -> str:
    html_lines = []
    in_list = False
    for line in md_text.splitlines():
        if line.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "---":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<hr>")
        elif m := re.match(r"!\[(.*?)\]\((.*?)\)", line.strip()):
            alt, rel = m.group(1), m.group(2)
            img_path = (DEL / rel).resolve()
            html_lines.append(f"<figure><figcaption>{alt}</figcaption>{_embed_image(img_path)}</figure>")
        elif line.strip():
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{line}</p>")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
    if in_list:
        html_lines.append("</ul>")
    body = "\n".join(html_lines)
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport Deep Learning</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 900px; margin: 2rem auto; line-height: 1.6; }}
    h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; }}
    figure {{ margin: 1.5rem 0; }}
    figcaption {{ font-style: italic; color: #444; margin-bottom: 0.5rem; }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def main() -> None:
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Missing report: {REPORT_MD}")
    REPORT_HTML.write_text(md_to_html(REPORT_MD.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Wrote {REPORT_HTML}")


if __name__ == "__main__":
    main()
