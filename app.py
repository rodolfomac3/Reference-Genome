# app.py
import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd
from flask import Flask, render_template, request, make_response
from dotenv import load_dotenv
from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from reportlab.platypus import LongTable  
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.pdfbase.pdfmetrics import stringWidth

# --- SSL trust (helps on macOS if CERTIFICATE_VERIFY_FAILED) ---
import ssl, certifi, urllib.request
_ctx = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
)

# --- PDF (ReportLab) ---
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ----------------- Config -----------------
load_dotenv()
EMAIL = os.getenv("ENTREZ_EMAIL", "rodolfo.mac3@gmail.com")
API_KEY = os.getenv("ENTREZ_API_KEY")
NCBI_DELAY = float(os.getenv("NCBI_DELAY", "0.34"))

Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

app = Flask(__name__)

# ----------------- Helpers -----------------
def _entrez_read(handle):
    try:
        return Entrez.read(handle)
    finally:
        handle.close()

def _sleep():
    time.sleep(NCBI_DELAY)

def sanitize_dna(seq: str) -> str:
    return re.sub(r"[^ACGTURYKMSWBDHVN]", "", (seq or "").upper())

def build_download_links(ftp: str):
    if not ftp:
        return {}
    https = ftp.replace("ftp://", "https://")
    base = https.rsplit("/", 1)[-1]
    prefix = f"{https}/{base}"
    return {
        "genomic_fna":       f"{prefix}_genomic.fna.gz",
        "cds_from_genomic":  f"{prefix}_cds_from_genomic.fna.gz",
        "protein_faa":       f"{prefix}_protein.faa.gz",
        "genomic_gff":       f"{prefix}_genomic.gff.gz",
        "genomic_gbff":      f"{prefix}_genomic.gbff.gz",
        "assembly_report":   f"{prefix}_assembly_report.txt",
        "assembly_stats":    f"{prefix}_assembly_stats.txt",
        "hashes_md5":        f"{prefix}_md5checksums.txt",
    }

def _styles():
    """ReportLab styles used for the PDF."""
    s = getSampleStyleSheet()
    # Base body
    body = s["BodyText"]
    body.fontSize = 9
    body.leading = 12

    small = s["BodyText"].clone('Small')
    small.fontSize = 8
    small.leading = 10

    code = s["BodyText"].clone('Code')
    code.fontName = "Courier"
    code.fontSize = 8.5
    code.leading = 10.5

    h2 = s["Heading2"].clone('H2')
    h2.spaceBefore = 8
    h2.spaceAfter = 4

    h3 = s["Heading3"].clone('H3')
    h3.spaceBefore = 6
    h3.spaceAfter = 3

    title = s["Title"].clone('TitleX')
    title.fontSize = 18
    title.leading = 22
    return {"body": body, "small": small, "code": code, "h2": h2, "h3": h3, "title": title}

def _para(text, style):
    """Safe Paragraph builder for None/empty strings."""
    from reportlab.platypus import Paragraph
    txt = "" if text is None else str(text)
    # Escape bare ampersands to avoid XML parse errors
    txt = txt.replace("&", "&amp;")
    return Paragraph(txt, style)

@lru_cache(maxsize=4096)
def ncbi_taxon_lookup(name: str) -> Optional[Dict]:
    if not name or not name.strip():
        return None
    _sleep()
    h = Entrez.esearch(db="taxonomy", term=name, retmode="xml")
    rec = _entrez_read(h)
    if not rec.get("IdList"):
        return None
    taxid = rec["IdList"][0]
    _sleep()
    h2 = Entrez.efetch(db="taxonomy", id=taxid, retmode="xml")
    tax = _entrez_read(h2)[0]
    return {
        "taxid": taxid,
        "scientific_name": tax.get("ScientificName"),
        "rank": tax.get("Rank", ""),
        "lineage": tax.get("Lineage", ""),
        "lineage_ex": [x["ScientificName"] for x in tax.get("LineageEx", [])],
    }

def _status_rank(s: str) -> int:
    s = (s or "").lower()
    order = ["complete genome", "chromosome", "scaffold", "contig", ""]
    for i, lab in enumerate(order):
        if lab in s:
            return i
    return len(order)

@lru_cache(maxsize=4096)
def best_assembly_for_taxid_raw(taxid: str) -> List[Dict]:
    if not taxid:
        return []
    term = f'txid{taxid}[Organism:exp] AND latest[filter]'
    _sleep()
    h = Entrez.esearch(db="assembly", term=term, retmax=300, retmode="xml")
    rec = _entrez_read(h)
    ids = rec.get("IdList", [])
    if not ids:
        return []
    _sleep()
    h2 = Entrez.esummary(db="assembly", id=",".join(ids), retmode="xml")
    summ = _entrez_read(h2)
    docs = summ["DocumentSummarySet"]["DocumentSummary"]

    rows = []
    for d in docs:
        cat = (d.get("RefSeq_category") or "")
        status = (d.get("AssemblyStatus") or "")
        ftp = d.get("FtpPath_RefSeq") or d.get("FtpPath_GenBank") or ""
        try:
            scaffold_n50 = int(d.get("ScaffoldN50") or 0)
        except Exception:
            scaffold_n50 = 0
        try:
            contig_n50 = int(d.get("ContigN50") or 0)
        except Exception:
            contig_n50 = 0
        try:
            size_mb = round(float(d.get("SeqLengthSum") or 0) / 1e6, 2)
        except Exception:
            size_mb = None

        score = 0.0
        if cat and cat.lower() in ("reference genome", "representative genome"):
            score += 100
        score += max(0, 10 - _status_rank(status) * 3)
        if ftp:
            score += 5
        score += contig_n50 / 1e6 + scaffold_n50 / 1e6

        rows.append({
            "assembly_accession": d.get("AssemblyAccession"),
            "assembly_name": d.get("AssemblyName"),
            "organism": d.get("Organism"),
            "refseq_category": cat or "na",
            "assembly_status": status or "na",
            "submit_date": d.get("SubmissionDate") or "",
            "update_date": d.get("LastUpdateDate") or "",
            "scaffold_n50": scaffold_n50,
            "contig_n50": contig_n50,
            "size_mb": size_mb,
            "ftp": ftp,
            "score": score,
        })
    return rows

# ---------- Primer Selection Helper ----------
def suggest_markers_from_lineage(lineage: str) -> List[Dict]:
    lin = (lineage or "").lower()
    out: List[Dict] = []
    def add(marker, primers, notes): out.append({"marker": marker, "primers": primers, "notes": notes})

    if "bacteria" in lin or "archaea" in lin:
        add("16S rRNA (V3–V4 / V4)", [
            {"name": "341F / 805R", "fwd": "CCTACGGGNGGCWGCAG", "rev": "GACTACHVGGGTATCTAATCC",
             "amplicon_bp": "~460 bp (V3–V4)", "platforms": ["Illumina PE250","Illumina PE300"],
             "use": "Microbiome profiling; widely used.", "refs": [{"label":"Klindworth 2013","url":"https://doi.org/10.1093/nar/gks808"}]},
            {"name": "515F / 806R", "fwd": "GTGCCAGCMGCCGCGGTAA", "rev": "GGACTACHVGGGTWTCTAAT",
             "amplicon_bp": "~291 bp (V4)", "platforms": ["Illumina PE250","Illumina PE300","eDNA-short"],
             "use": "EMP V4 standard.", "refs": [{"label":"EMP","url":"https://earthmicrobiome.org/"}]},
            {"name": "27F / 1492R","fwd":"AGAGTTTGATCMTGGCTCAG","rev":"GGTTACCTTGTTACGACTT",
             "amplicon_bp":"≈1,450 bp", "platforms":["Sanger","ONT/PacBio"],
             "use":"Full-length 16S for isolates.", "refs":[{"label":"Lane 1991","url":"https://doi.org/10.1016/0076-6879(91)94057-T"}]},
        ], "Choose region by platform/read length.")
    if "fungi" in lin or "fungus" in lin:
        add("ITS (ITS1–ITS2)", [
            {"name":"ITS1F / ITS2","fwd":"CTTGGTCATTTAGAGGAAGTAA","rev":"GCTGCGTTCTTCATCGATGC",
             "amplicon_bp":"~300–450 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"Fungal metabarcoding (ITS1).","refs":[{"label":"UNITE","url":"https://unite.ut.ee/"}]},
            {"name":"ITS3 / ITS4","fwd":"GCATCGATGAAGAACGCAGC","rev":"TCCTCCGCTTATTGATATGC",
             "amplicon_bp":"~300–500 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"Fungal metabarcoding (ITS2).","refs":[{"label":"White 1990","url":"https://doi.org/10.1016/B978-0-12-372180-8.50042-1"}]},
        ], "Primary fungal barcode.")
    if any(tag in lin for tag in ["viridiplantae","plantae","embryophyta","streptophyta"]):
        add("rbcL",[{"name":"rbcLa-F / rbcLa-R","fwd":"ATGTCACCACAAACAGAGACTAAAGC","rev":"GTAAAATCAAGTCCACCRCG",
             "amplicon_bp":"~550–600 bp","platforms":["Sanger","Illumina PE300"],"use":"Core plant barcode.","refs":[{"label":"CBOL 2009","url":"https://doi.org/10.1073/pnas.0905845106"}]}],"Core plant barcode.")
        add("matK",[{"name":"matK-390F / matK-1326R","fwd":"CGATCTATTCATTCAATATTTC","rev":"TCTAGCACACGAAAGTCGAAGT",
             "amplicon_bp":"~900 bp","platforms":["Sanger","ONT/PacBio"],"use":"Higher resolution with rbcL.","refs":[{"label":"CBOL 2009","url":"https://doi.org/10.1073/pnas.0905845106"}]}],"Use with rbcL.")
        add("trnL (P6 mini)",[{"name":"g / h","fwd":"GGGCAATCCTGAGCCAA","rev":"CCATTGAGTCTCTGCACCTATC",
             "amplicon_bp":"~10–150 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Degraded DNA/eDNA.","refs":[{"label":"Taberlet 2007","url":"https://doi.org/10.1093/nar/gkm938"}]}],"Short eDNA-friendly locus.")
    if "metazoa" in lin or "animalia" in lin:
        add("COI (Folmer region)",[
            {"name":"LCO1490 / HCO2198","fwd":"GGTCAACAAATCATAAAGATATTGG","rev":"TAAACTTCAGGGTGACCAAAAAATCA",
             "amplicon_bp":"~650 bp","platforms":["Sanger","Illumina PE300","ONT/PacBio"],"use":"Standard animal barcode.","refs":[{"label":"Folmer 1994","url":"https://doi.org/10.1016/0003-2697(94)90013-2"}]}
        ], "Standard animal barcode.")
        add("12S (eDNA vertebrates)",[
            {"name":"MiFish-U","fwd":"GTCGGTAAAACTCGTGCCAGC","rev":"CATAGTGGGGTATCTAATCCCAGTTTG",
             "amplicon_bp":"~170 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Vertebrate eDNA surveys.","refs":[{"label":"Miya 2015","url":"https://doi.org/10.1098/rsos.150088"}]}
        ], "Great for vertebrate eDNA.")
    if "eukaryota" in lin and not any(k in lin for k in ["animalia","plantae","fungi"]):
        add("18S rRNA (V4 / V9)",[
            {"name":"18S V4 (general)","fwd":"CCAGCASCYGCGGTAATTCC","rev":"ACTTTCGTTCTTGATYRA",
             "amplicon_bp":"~380–420 bp (V4)","platforms":["Illumina PE250","Illumina PE300","Sanger"],"use":"Pan-eukaryote survey.","refs":[{"label":"Stoeck 2010","url":"https://doi.org/10.1111/j.1365-294X.2010.04695.x"}]}
        ], "Pan-eukaryotic survey marker.")
    return out

def quick_blast_guess(seq: str, program: str = "blastn", db: str = "nt", hitlist_size: int = 5):
    seq = sanitize_dna(seq)
    if len(seq) < 80:
        return None
    rh = NCBIWWW.qblast(program, db, seq, hitlist_size=hitlist_size)
    record = NCBIXML.read(rh)
    hits = []
    for aln in record.alignments:
        hsp = aln.hsps[0]
        hits.append({
            "title": aln.title,
            "length": aln.length,
            "e": hsp.expect,
            "score": hsp.score,
            "identity": round(100 * hsp.identities / max(1, hsp.align_length), 2),
        })
    return pd.DataFrame(hits)

# ------------ Shared context builder (used by /search and exports) ------------
def build_context_from_request(form):
    name = (form.get("taxon_name") or "").strip()
    dna = (form.get("dna_seq") or "")
    do_blast = form.get("blast_hint") == "on"
    refseq_only = form.get("refseq_only") == "on"
    status_filter = (form.get("status_filter") or "any").lower()
    min_contig_n50 = int(form.get("min_contig_n50") or 0)
    sort_by = form.get("sort_by") or "score"
    selected_accession = form.get("selected_accession") or ""
    intended_platform = form.get("intended_platform") or "any"

    tax_info = ncbi_taxon_lookup(name) if name else None

    ranked, best_row, links, suggestions = [], None, {}, []
    if tax_info:
        all_rows = best_assembly_for_taxid_raw(tax_info["taxid"])

        def keep(r):
            if refseq_only and (r["refseq_category"].lower() not in ("reference genome", "representative genome")):
                return False
            if status_filter != "any" and status_filter not in r["assembly_status"].lower():
                return False
            if min_contig_n50 and r["contig_n50"] < min_contig_n50:
                return False
            return True

        ranked = [r for r in all_rows if keep(r)]

        key_map = {
            "score": lambda r: (-r["score"], -r["contig_n50"], -r["scaffold_n50"]),
            "update_date": lambda r: r["update_date"],
            "contig_n50": lambda r: (-r["contig_n50"], -r["scaffold_n50"]),
            "scaffold_n50": lambda r: (-r["scaffold_n50"], -r["contig_n50"]),
            "status": lambda r: _status_rank(r["assembly_status"]),
        }
        ranked.sort(key=key_map.get(sort_by, key_map["score"]))

        if selected_accession:
            best_row = next((r for r in ranked if r["assembly_accession"] == selected_accession), None)
        if not best_row and ranked:
            best_row = ranked[0]
        if best_row and best_row.get("ftp"):
            links = build_download_links(best_row["ftp"])

        suggestions = suggest_markers_from_lineage(tax_info.get("lineage",""))

    return {
        "name": name, "dna": dna, "do_blast": do_blast,
        "refseq_only": refseq_only, "status_filter": status_filter,
        "min_contig_n50": min_contig_n50, "sort_by": sort_by,
        "selected_accession": selected_accession, "intended_platform": intended_platform,
        "tax_info": tax_info, "ranked": ranked, "best": best_row,
        "links": links, "suggestions": suggestions
    }

# ----------------- Routes -----------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/search")
def search():
    ctx = build_context_from_request(request.form)

    # optional BLAST (only for on-screen hint; we keep exports lean)
    blast_html = None
    if ctx["do_blast"] and ctx["dna"].strip():
        try:
            guessed_df = quick_blast_guess(ctx["dna"])
            if isinstance(guessed_df, pd.DataFrame) and not guessed_df.empty:
                blast_html = guessed_df.to_html(index=False, classes="table table-sm table-hover align-middle")
        except Exception as e:
            blast_html = f"<div class='alert alert-warning'>BLAST hint failed: {e}</div>"

    ranked_df = pd.DataFrame(ctx["ranked"], columns=[
        "assembly_accession","assembly_name","refseq_category","assembly_status",
        "contig_n50","scaffold_n50","size_mb","submit_date","update_date","ftp","score"
    ])
    ranked_html = ranked_df.to_html(index=False, classes="table table-sm table-hover align-middle") if not ranked_df.empty else None

    return render_template(
        "index.html",
        name=ctx["name"], dna=ctx["dna"], do_blast=ctx["do_blast"],
        refseq_only=ctx["refseq_only"], status_filter=ctx["status_filter"],
        min_contig_n50=ctx["min_contig_n50"], sort_by=ctx["sort_by"],
        selected_accession=ctx["best"]["assembly_accession"] if ctx["best"] else "",
        intended_platform=ctx["intended_platform"],
        tax_info=ctx["tax_info"], ranked=ctx["ranked"], ranked_html=ranked_html,
        best=ctx["best"], links=ctx["links"], suggestions=ctx["suggestions"],
        blast_html=blast_html, error=None, warning=None
    )

# ----------------- Exports -----------------
@app.post("/export/csv")
def export_csv():
    ctx = build_context_from_request(request.form)
    ranked = ctx["ranked"]
    if not ranked:
        return ("No data to export. Run a search first.", 400)
    df = pd.DataFrame(ranked, columns=[
        "assembly_accession","assembly_name","refseq_category","assembly_status",
        "contig_n50","scaffold_n50","size_mb","submit_date","update_date","ftp","score"
    ])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    resp = make_response(csv_bytes)
    safe = (ctx["tax_info"]["scientific_name"].replace(" ", "_") if ctx["tax_info"] else "export")
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe}_assemblies.csv"'
    return resp

@app.post("/export/pdf")
def export_pdf():
    ctx = build_context_from_request(request.form)
    if not ctx["tax_info"]:
        return ("No data to export. Provide an organism name first.", 400)

    st = _styles()

    # Landscape Letter for more horizontal room
    buff = BytesIO()
    margin = 28  # slightly tighter margins than default
    doc = SimpleDocTemplate(
        buff,
        pagesize=landscape(LETTER),
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin
    )

    story = []

    # ----- Header / Title -----
    title = f"Marker Finder Report — {ctx['tax_info']['scientific_name']} (taxid {ctx['tax_info']['taxid']})"
    story += [_para(title, st["title"]), Spacer(1, 8)]

    # ----- Selected Assembly -----
    best = ctx["best"]
    if best:
        story += [_para("<b>Selected Assembly</b>", st["h2"])]

        info_tbl = [
            [_para("Accession", st["small"]), _para(best["assembly_accession"], st["small"])],
            [_para("Name", st["small"]), _para(best["assembly_name"], st["small"])],
            [_para("Status", st["small"]), _para(best["assembly_status"], st["small"])],
            [_para("RefSeq category", st["small"]), _para(best["refseq_category"], st["small"])],
            [_para("Contig N50", st["small"]), _para(f"{best['contig_n50']:,}", st["small"])],
            [_para("Scaffold N50", st["small"]), _para(f"{best['scaffold_n50']:,}", st["small"])],
            [_para("Genome size (Mb)", st["small"]), _para(best["size_mb"], st["small"])],
            [_para("FTP", st["small"]), _para(best["ftp"] or "—", st["small"])],
        ]
        t = Table(info_tbl, colWidths=[110, 560])
        t.setStyle(TableStyle([
            ("BOX",(0,0),(-1,-1),0.5,colors.grey),
            ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(0,-1),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),4),
            ("RIGHTPADDING",(0,0),(-1,-1),4),
            ("TOPPADDING",(0,0),(-1,-1),3),
            ("BOTTOMPADDING",(0,0),(-1,-1),3),
        ]))
        story += [t, Spacer(1, 8)]

        # Direct links
        if ctx["links"]:
            story += [_para("<b>Direct downloads</b>", st["h3"])]
            link_rows = [[_para(k.replace("_"," "), st["small"]), _para(v, st["small"])] for k,v in ctx["links"].items()]
            lt = Table(link_rows, colWidths=[140, 530])
            lt.setStyle(TableStyle([
                ("BOX",(0,0),(-1,-1),0.5,colors.grey),
                ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),4),
                ("RIGHTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),3),
                ("BOTTOMPADDING",(0,0),(-1,-1),3),
            ]))
            story += [lt, Spacer(1, 8)]

    # ----- Assemblies (top 25) -----
    ranked = ctx["ranked"][:25]
    if ranked:
        story += [_para("<b>Assemblies (filtered &amp; sorted)</b>", st["h2"])]
        header = [
            _para("Accession", st["small"]), _para("Name", st["small"]), _para("RefSeq", st["small"]),
            _para("Status", st["small"]), _para("Contig N50", st["small"]),
            _para("Scaffold N50", st["small"]), _para("Size (Mb)", st["small"]), _para("Updated", st["small"])
        ]
        body = []
        for r in ranked:
            body.append([
                _para(r["assembly_accession"], st["small"]),
                _para(r["assembly_name"], st["small"]),
                _para(r["refseq_category"], st["small"]),
                _para(r["assembly_status"], st["small"]),
                _para(f"{r['contig_n50']:,}", st["small"]),
                _para(f"{r['scaffold_n50']:,}", st["small"]),
                _para(r["size_mb"], st["small"]),
                _para(r["update_date"], st["small"]),
            ])
        at = LongTable([header] + body, repeatRows=1,
                       colWidths=[85, 200, 70, 80, 80, 90, 60, 90])
        at.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),3),
            ("RIGHTPADDING",(0,0),(-1,-1),3),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        story += [at, Spacer(1, 8)]

    # ----- Primers -----
    if ctx["suggestions"]:
        story += [_para("<b>Recommended Markers &amp; Primers</b>", st["h2"])]
        for s in ctx["suggestions"]:
            story += [_para(s["marker"], st["h3"])]
            if s.get("notes"):
                story += [_para(s["notes"], st["small"])]

            # Column widths tuned for landscape; use Paragraphs for wrapping
            cols = [110, 160, 160, 70, 140, 150]
            header = [
                _para("Name", st["small"]),
                _para("Forward", st["small"]),
                _para("Reverse", st["small"]),
                _para("Amplicon", st["small"]),
                _para("Platforms", st["small"]),
                _para("Use case", st["small"]),
            ]
            rows = [header]
            for p in s["primers"]:
                rows.append([
                    _para(p.get("name",""), st["small"]),
                    _para(p.get("fwd",""), st["code"]),   # monospace, wraps
                    _para(p.get("rev",""), st["code"]),
                    _para(p.get("amplicon_bp",""), st["small"]),
                    _para(", ".join(p.get("platforms",[]) or []), st["small"]),
                    _para(p.get("use",""), st["small"]),
                ])

            pt = LongTable(rows, repeatRows=1, colWidths=cols)
            pt.setStyle(TableStyle([
                ("GRID",(0,0),(-1,-1),0.25,colors.grey),
                ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),3),
                ("RIGHTPADDING",(0,0),(-1,-1),3),
                ("TOPPADDING",(0,0),(-1,-1),2),
                ("BOTTOMPADDING",(0,0),(-1,-1),2),
            ]))
            story += [pt, Spacer(1, 6)]

    # ----- Page header/footer (page numbers) -----
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        txt = f"Page {doc.page}"
        canvas.drawRightString(doc.pagesize[0] - margin, margin - 12, txt)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = buff.getvalue()
    buff.close()

    safe = ctx["tax_info"]["scientific_name"].replace(" ", "_")
    resp = make_response(pdf)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe}_marker_finder_report.pdf"'
    return resp



# ----------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
