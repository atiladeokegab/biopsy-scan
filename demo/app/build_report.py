# Usage (repo root): uv run --with reportlab --with pillow python demo/app/build_report.py  -> demo/app/report.pdf
import io
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "workflow" / "img"
OUT = str(HERE / "report.pdf")
import json
PAGE = {e["file"]: e["page_url"] for e in json.loads((SRC / "hpa_manifest.json").read_text())}

INK, MUTED, RULE = HexColor("#14181C"), HexColor("#5A636B"), HexColor("#DDE2E6")
EMERALD, CORAL = HexColor("#2BD4A4"), HexColor("#E86F51")
W, Hp = A4
M = 54  # generous margin
c = canvas.Canvas(OUT, pagesize=A4)
c.setTitle("Biopsy Scan · Biopsy report")

def para(text, x, y_top, width, size=10.5, color=INK, font="Helvetica", lead=None):
    p = Paragraph(text, ParagraphStyle("p", fontName=font, fontSize=size, leading=lead or size * 1.4, textColor=color))
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y_top - h)
    return y_top - h

# accent bar + title
c.setFillColor(EMERALD); c.rect(M, Hp - M - 4, 40, 4, stroke=0, fill=1)
y = Hp - M - 40
c.setFillColor(INK); c.setFont("Times-Roman", 30); c.drawString(M, y, "Biopsy Scan · Biopsy report")
# coral DEMO tag
tag = "DEMO · illustration, not a clinical result"
c.setFont("Helvetica-Bold", 9.5); tw = c.stringWidth(tag, "Helvetica-Bold", 9.5)
y -= 30
c.setFillColor(CORAL); c.roundRect(M, y - 6, tw + 16, 20, 4, stroke=0, fill=1)
c.setFillColor(HexColor("#FFFFFF")); c.drawString(M + 8, y, tag)

# photos side by side (antibody IDs from ../MANIFEST.json entries with the same HPA image series)
photos = [("hpa_best_01_KRT5.webp", "KRT5", "CAB000027"), ("hpa_best_12_CD8A.webp", "CD8A", "CAB000012")]
gap = 18
pw = (W - 2 * M - gap) / 2
y -= 26
for i, (f, gene, ab) in enumerate(photos):
    im = Image.open(SRC / f).convert("RGB")
    ph = pw * im.height / im.width
    x = M + i * (pw + gap)
    buf = io.BytesIO(); im.resize((1000, round(1000 * im.height / im.width)), Image.LANCZOS).save(buf, "JPEG", quality=88); buf.seek(0)
    c.drawImage(ImageReader(buf), x, y - ph, pw, ph)
    c.linkURL(PAGE[f], (x, y - ph, x + pw, y), relative=0)
    c.setStrokeColor(RULE); c.setLineWidth(0.6); c.rect(x, y - ph, pw, ph)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 10.5); c.drawString(x, y - ph - 16, gene)
    c.setFillColor(MUTED); c.setFont("Helvetica", 9.5)
    c.drawString(x + c.stringWidth(gene, "Helvetica-Bold", 10.5) + 6, y - ph - 16, f"antibody {ab} · IHC (DAB)")
y = y - ph - 46

# example stains
c.setFillColor(INK); c.setFont("Times-Roman", 17)
c.drawString(M, y, "Example routine stains: ")
c.setFont("Times-Bold", 17); c.drawString(M + c.stringWidth("Example routine stains: ", "Times-Roman", 17), y, "KRT5 and CD8A")
y = para("Biopsy Scan's aim is to tell the pathologist which one or two routine stains carry the signal, "
         "so a hospital can act on a stain it already runs.", M, y - 10, W - 2 * M, color=MUTED)

# risk analysis box
y -= 22
# wording matches the app's risk card (demo/app/index.html) word for word
bh = 96
c.setStrokeColor(RULE); c.setLineWidth(0.8); c.roundRect(M, y - bh, W - 2 * M, bh, 6, stroke=1, fill=0)
c.setFillColor(EMERALD); c.rect(M, y - bh, 4, bh, stroke=0, fill=1)
c.setFillColor(MUTED); c.setFont("Courier-Bold", 8.5)
c.drawString(M + 18, y - 20, "MEASURED · Exploratory Charville survival screen, results/charville-v2.json")
c.setFillColor(INK); c.setFont("Times-Roman", 20); c.drawString(M + 18, y - 46, "Overall survival")
para("No validated risk score issued. None of the 3,780 features passed the 0.05 threshold (maxT p = 0.0981). "
     "Biopsy Scan withholds a score rather than guess.", M + 18, y - 56, W - 2 * M - 36, size=10.5)
y -= bh + 28

# speed
c.setFillColor(INK); c.setFont("Helvetica-Bold", 11); c.drawString(M, y, "Speed")
y = para("Off-the-shelf squidpy + lifelines 22.35 s per study over 48 external studies vs Biopsy Scan 0.0426 s per study "
         "over 3,000 fused evaluations, 8 workers each: <b>534×</b>, the median of paired per-repeat ratios over "
         "3 fresh containers on the same host (bench/results/same-host-20261122T140908Z.json). The external time "
         "is the whole off-the-shelf call per study; the fused time is re-imaging and scoring, with the calibrated "
         "threshold and calls excluded.",
         M, y - 8, W - 2 * M)

# footer
c.setStrokeColor(RULE); c.line(M, M + 46, W - M, M + 46)
para("Photos: Human Protein Atlas v25 ("
     + ", ".join(f'<a href="{PAGE[f]}" color="#5A636B"><u>{g}</u></a>' for f, g, _ in photos)
     + '), <a href="https://creativecommons.org/licenses/by/4.0/" color="#5A636B"><u>CC BY 4.0</u></a>; '
     'Uhlén et al., <a href="https://doi.org/10.1126/science.1260419" color="#5A636B"><u>Science 2015</u></a>. '
     "Cropped and resized. Head and neck cancer sections from other patients, not scored by Biopsy Scan. "
     "Illustration only; not for clinical use.",
     M, M + 38, W - 2 * M, size=8.5, color=MUTED)
c.save()
print(OUT)
