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

INK, MUTED, RULE = HexColor("#14181C"), HexColor("#5A636B"), HexColor("#DDE2E6")
EMERALD, CORAL = HexColor("#2BD4A4"), HexColor("#E86F51")
W, Hp = A4
M = 54  # generous margin
c = canvas.Canvas(OUT, pagesize=A4)
c.setTitle("Preflight · Biopsy report")

def para(text, x, y_top, width, size=10.5, color=INK, font="Helvetica", lead=None):
    p = Paragraph(text, ParagraphStyle("p", fontName=font, fontSize=size, leading=lead or size * 1.4, textColor=color))
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y_top - h)
    return y_top - h

# accent bar + title
c.setFillColor(EMERALD); c.rect(M, Hp - M - 4, 40, 4, stroke=0, fill=1)
y = Hp - M - 40
c.setFillColor(INK); c.setFont("Times-Roman", 30); c.drawString(M, y, "Preflight · Biopsy report")
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
    c.setStrokeColor(RULE); c.setLineWidth(0.6); c.rect(x, y - ph, pw, ph)
    c.setFillColor(INK); c.setFont("Helvetica-Bold", 10.5); c.drawString(x, y - ph - 16, gene)
    c.setFillColor(MUTED); c.setFont("Helvetica", 9.5)
    c.drawString(x + c.stringWidth(gene, "Helvetica-Bold", 10.5) + 6, y - ph - 16, f"antibody {ab} · IHC (DAB)")
y = y - ph - 46

# example stains
c.setFillColor(INK); c.setFont("Times-Roman", 17)
c.drawString(M, y, "Example routine stains: ")
c.setFont("Times-Bold", 17); c.drawString(M + c.stringWidth("Example routine stains: ", "Times-Roman", 17), y, "KRT5 and CD8A")
y = para("Preflight's aim is to tell the pathologist which one or two routine stains carry the signal, "
         "so a hospital can act on a stain it already runs.", M, y - 10, W - 2 * M, color=MUTED)

# risk analysis box
y -= 22
bh = 78
c.setStrokeColor(RULE); c.setLineWidth(0.8); c.roundRect(M, y - bh, W - 2 * M, bh, 6, stroke=1, fill=0)
c.setFillColor(EMERALD); c.rect(M, y - bh, 4, bh, stroke=0, fill=1)
c.setFillColor(INK); c.setFont("Helvetica-Bold", 11); c.drawString(M + 18, y - 22, "Risk of outcome event")
c.setFont("Times-Roman", 20); c.drawString(M + 18, y - 48, "Illustrative")
para("Score withheld in this demo: no numeric risk is shown. A real report would describe the risk of an outcome event "
     "(our screen used overall survival), scored from the example stains.", M + 160, y - 14, W - 2 * M - 180, size=9.5, color=MUTED)
y -= bh + 28

# speed
c.setFillColor(INK); c.setFont("Helvetica-Bold", 11); c.drawString(M, y, "Speed")
y = para("Off-the-shelf squidpy + lifelines 24.4 s vs Preflight 0.042 s per study, 8 workers each: <b>583×</b> "
         "(cross-run, same container class; bench/results/external-baseline-20261122T043823Z.json).",
         M, y - 8, W - 2 * M)

# footer
c.setStrokeColor(RULE); c.line(M, M + 46, W - M, M + 46)
para("Photos: Human Protein Atlas, proteinatlas.org (Uhlen et al.), CC BY-SA 3.0. Head and neck cancer sections "
     "from other patients, not scored by Preflight. Illustration only; not for clinical use.",
     M, M + 38, W - 2 * M, size=8.5, color=MUTED)
c.save()
print(OUT)
