"""Build the English user-guide PDF.

Markdown is rendered with Python-Markdown and paginated with WeasyPrint.
Chapters (``#``) and sections (``##``) are numbered. A cover and a table of
contents are added. A paragraph that is only an image becomes a captioned
figure. DejaVu is embedded so quoted non-Latin prompts print.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import markdown
from weasyprint import CSS, HTML

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide"
LANGUAGES = ("en",)
FONTS = Path("/usr/share/fonts/truetype/dejavu")
BUILD = json.loads((GUIDE.parents[1] / "BUILD.json").read_text(encoding="utf-8"))["build"]

META = {
    "en": {
        "title": "VOXPrint AI Canvas",
        "subtitle": "User guide",
        "version": f"Version 0.1.0 (release candidate) · build {BUILD} · guide of 18 October 2026",
        "licence": "Apache-2.0 · © 2026 Aleksandr Mitroshenkov",
        "contents": "Contents",
    },
}

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3}) (?!#)(.+?)\s*$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\(screens/([a-z0-9-]+)\.png\)")
_FIGURE = re.compile(r"<p>\s*(<img\b[^>]*>)\s*</p>", re.IGNORECASE)
_IMG_ALT = re.compile(r'\balt="([^"]*)"', re.IGNORECASE)


def main(argv: list[str] | None = None) -> int:
    """Write the PDF. An optional argument is the output directory."""
    args = list(sys.argv[1:] if argv is None else argv)
    destination = Path(args[0]) if args else GUIDE / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        path = destination / f"VOXPrint-AI-Canvas-Guide-{language}.pdf"
        build_language(language, path)
        print(path)
    return 0


def build_language(language: str, destination: Path) -> None:
    """Render one language file to ``destination``."""
    source = GUIDE / f"{language}.md"
    raw = source.read_text(encoding="utf-8")
    if "TODO(check)" in raw:
        raise SystemExit(f"{source} still contains TODO(check)")
    body, toc = _prepare(raw, language)
    html_body = markdown.markdown(
        body,
        extensions=["tables", "sane_lists", "fenced_code", "attr_list"],
        output_format="html",
    )
    document = _document(language, _figures(html_body), toc)
    HTML(string=document, base_url=str(GUIDE)).write_pdf(
        destination,
        stylesheets=[CSS(string=_css())],
    )


def _prepare(raw: str, language: str) -> tuple[str, list[tuple[int, str, str]]]:
    text = _COMMENT.sub("", raw).strip()
    text = _IMAGE.sub(lambda match: _image(match, language), text)
    return _number(text)


def _image(match: re.Match[str], language: str) -> str:
    caption = match.group(1)
    image_id = match.group(2)
    path = GUIDE / "screens" / language / f"{image_id}.png"
    if not path.is_file():
        raise SystemExit(f"missing screenshot {language}/{image_id}.png")
    return f"![{caption}]({path.resolve().as_uri()})"


def _number(text: str) -> tuple[str, list[tuple[int, str, str]]]:
    chapter = 0
    section = 0
    lines: list[str] = []
    toc: list[tuple[int, str, str]] = []
    for line in text.splitlines():
        found = _HEADING.match(line)
        if found is None:
            lines.append(line)
            continue
        level = len(found.group(1))
        title = found.group(2).strip()
        if level == 1:
            chapter += 1
            section = 0
            anchor = f"ch-{chapter}"
            numbered = f"{chapter}. {title}"
            toc.append((1, anchor, numbered))
            lines.append(f"# {numbered} {{#{anchor}}}")
        elif level == 2:
            section += 1
            anchor = f"sec-{chapter}-{section}"
            numbered = f"{chapter}.{section}. {title}"
            toc.append((2, anchor, numbered))
            lines.append(f"## {numbered} {{#{anchor}}}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n", toc


def _figures(html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        tag = match.group(1)
        alt = _IMG_ALT.search(tag)
        caption = "" if alt is None else alt.group(1)
        return f"<figure>{tag}<figcaption>{caption}</figcaption></figure>"

    return _FIGURE.sub(replace, html)


def _document(language: str, body: str, toc: list[tuple[int, str, str]]) -> str:
    meta = META[language]
    entries = "\n".join(
        f'<a class="toc-{level}" href="#{anchor}">{_escape(title)}</a>' for level, anchor, title in toc
    )
    return f"""<!DOCTYPE html>
<html lang="{language}">
<head><meta charset="utf-8"><title>{_escape(meta["title"])}</title></head>
<body>
<section class="cover">
<p class="kicker">VOXPrint</p>
<h1>{_escape(meta["title"])}</h1>
<p class="subtitle">{_escape(meta["subtitle"])}</p>
<p class="version">{_escape(meta["version"])}</p>
<p class="licence">{_escape(meta["licence"])}</p>
</section>
<nav class="toc">
<p class="toc-heading">{_escape(meta["contents"])}</p>
{entries}
</nav>
{body}
</body>
</html>
"""


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _css() -> str:
    regular = (FONTS / "DejaVuSans.ttf").as_uri()
    bold = (FONTS / "DejaVuSans-Bold.ttf").as_uri()
    italic = (FONTS / "DejaVuSans-Oblique.ttf").as_uri()
    mono = (FONTS / "DejaVuSansMono.ttf").as_uri()
    return f"""
@font-face {{
  font-family: "Guide Sans";
  src: url("{regular}");
  font-weight: 400;
  font-style: normal;
}}
@font-face {{
  font-family: "Guide Sans";
  src: url("{bold}");
  font-weight: 700;
  font-style: normal;
}}
@font-face {{
  font-family: "Guide Sans";
  src: url("{italic}");
  font-weight: 400;
  font-style: italic;
}}
@font-face {{
  font-family: "Guide Mono";
  src: url("{mono}");
  font-weight: 400;
  font-style: normal;
}}
@page {{
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
  @top-center {{
    content: "AI Canvas";
    font-family: "Guide Sans";
    font-size: 9pt;
    color: #5c6570;
  }}
  @bottom-center {{
    content: counter(page);
    font-family: "Guide Sans";
    font-size: 9pt;
    color: #5c6570;
  }}
}}
@page :first {{
  margin: 0;
  @top-center {{ content: none; }}
  @bottom-center {{ content: none; }}
}}
html {{
  font-family: "Guide Sans", sans-serif;
  font-size: 11pt;
  line-height: 1.45;
  color: #1c2430;
}}
body {{ margin: 0; }}
.cover {{
  page: cover;
  break-after: page;
  height: 297mm;
  box-sizing: border-box;
  padding: 70mm 22mm 0 22mm;
  background: #f7f5f1;
  color: #1c2430;
}}
.cover h1 {{
  break-before: auto;
  margin: 0 0 8mm 0;
  font-size: 28pt;
  line-height: 1.15;
}}
.kicker {{
  margin: 0 0 8mm 0;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #5c6570;
  font-size: 11pt;
}}
.subtitle {{ margin: 0; font-size: 16pt; }}
.version, .licence {{ margin: 6mm 0 0 0; color: #3d4654; }}
nav.toc {{ break-after: page; }}
.toc-heading {{
  font-size: 20pt;
  font-weight: 700;
  margin: 0 0 8mm 0;
}}
nav.toc a {{
  display: block;
  color: #1c2430;
  text-decoration: none;
  margin: 1.2mm 0;
}}
nav.toc a.toc-2 {{ margin-left: 8mm; font-size: 10.5pt; }}
nav.toc a::after {{
  content: leader(".") target-counter(attr(href), page);
}}
h1 {{
  break-before: page;
  break-after: avoid;
  font-size: 18pt;
  margin: 0 0 4mm 0;
}}
h2 {{
  break-after: avoid;
  font-size: 14pt;
  margin: 7mm 0 3mm 0;
}}
h3 {{
  break-after: avoid;
  font-size: 12pt;
  margin: 5mm 0 2mm 0;
}}
p {{ margin: 0 0 3mm 0; }}
a {{ color: #1a4f8b; }}
strong {{ font-weight: 700; }}
blockquote {{
  margin: 3mm 0;
  padding: 1mm 0 1mm 4mm;
  border-left: 2.5pt solid #8aa0b8;
  color: #243040;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  margin: 3mm 0 4mm 0;
  break-inside: avoid;
  font-size: 10pt;
}}
th, td {{
  border: 0.4pt solid #c5ced6;
  padding: 1.4mm 2mm;
  vertical-align: top;
  text-align: left;
}}
th {{ background: #eef2f6; font-weight: 700; }}
pre {{
  background: #f4f6f8;
  padding: 3mm;
  break-inside: avoid;
  white-space: pre-wrap;
  font-family: "Guide Mono", monospace;
  font-size: 9pt;
}}
code {{
  font-family: "Guide Mono", monospace;
  font-size: 0.92em;
}}
ul, ol {{ margin: 0 0 3mm 0; padding-left: 6mm; }}
li {{ margin: 0.6mm 0; }}
figure {{
  margin: 4mm 0 5mm 0;
  break-inside: avoid;
  text-align: center;
}}
figure img {{
  max-width: 100%;
  max-height: 150mm;
}}
figcaption {{
  margin-top: 1.5mm;
  font-size: 9.5pt;
  font-style: italic;
  color: #3d4654;
}}
"""


if __name__ == "__main__":
    raise SystemExit(main())
