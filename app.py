import os
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, render_template, request
from dotenv import load_dotenv
from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML

# --- SSL trust (fix CERTIFICATE_VERIFY_FAILED on macOS) ---
import ssl, certifi, urllib.request
_ctx = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
)

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
# Platforms: 'Sanger', 'Illumina PE250', 'Illumina PE300', 'eDNA-short', 'ONT/PacBio'
def suggest_markers_from_lineage(lineage: str) -> List[Dict]:
    lin = (lineage or "").lower()
    out: List[Dict] = []
    def add(marker, primers, notes): out.append({"marker": marker, "primers": primers, "notes": notes})

    if "bacteria" in lin or "archaea" in lin:
        add("16S rRNA (V3–V4 / V4)", [
            {"name": "341F / 805R", "fwd": "CCTACGGGNGGCWGCAG", "rev": "GACTACHVGGGTATCTAATCC",
             "amplicon_bp": "~460 bp (V3–V4)", "platforms": ["Illumina PE250","Illumina PE300"],
             "use": "Microbiome profiling; widely used in 16S surveys.",
             "refs": [{"label":"Klindworth et al. 2013","url":"https://doi.org/10.1093/nar/gks808"}]},
            {"name": "515F / 806R", "fwd": "GTGCCAGCMGCCGCGGTAA", "rev": "GGACTACHVGGGTWTCTAAT",
             "amplicon_bp": "~291 bp (V4)", "platforms": ["Illumina PE250","Illumina PE300","eDNA-short"],
             "use": "Earth Microbiome Project standard V4; robust across bacteria/archaea.",
             "refs": [{"label":"EMP protocol","url":"https://earthmicrobiome.org/"}]},
            {"name": "27F / 1492R", "fwd": "AGAGTTTGATCMTGGCTCAG", "rev": "GGTTACCTTGTTACGACTT",
             "amplicon_bp": "≈1,450 bp (near-full 16S)", "platforms": ["Sanger","ONT/PacBio"],
             "use": "Full-length 16S for isolate identification.",
             "refs": [{"label":"Lane 1991","url":"https://doi.org/10.1016/0076-6879(91)94057-T"}]},
        ], "Choose region by platform/read length.")
    if "fungi" in lin or "fungus" in lin:
        add("ITS (ITS1–ITS2)", [
            {"name":"ITS1F / ITS2","fwd":"CTTGGTCATTTAGAGGAAGTAA","rev":"GCTGCGTTCTTCATCGATGC",
             "amplicon_bp":"~300–450 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"General fungal metabarcoding (ITS1).","refs":[{"label":"UNITE","url":"https://unite.ut.ee/"}]},
            {"name":"ITS3 / ITS4","fwd":"GCATCGATGAAGAACGCAGC","rev":"TCCTCCGCTTATTGATATGC",
             "amplicon_bp":"~300–500 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"General fungal metabarcoding (ITS2).","refs":[{"label":"White et al. 1990","url":"https://doi.org/10.1016/B978-0-12-372180-8.50042-1"}]},
        ], "Primary fungal barcode; length varies across taxa.")
    if any(tag in lin for tag in ["viridiplantae","plantae","embryophyta","streptophyta"]):
        add("rbcL", [
            {"name":"rbcLa-F / rbcLa-R","fwd":"ATGTCACCACAAACAGAGACTAAAGC","rev":"GTAAAATCAAGTCCACCRCG",
             "amplicon_bp":"~550–600 bp","platforms":["Sanger","Illumina PE300"],
             "use":"Core plant barcode; high universality.","refs":[{"label":"CBOL Plant WG","url":"https://doi.org/10.1073/pnas.0905845106"}]}
        ], "Core plant barcode; robust across groups.")
        add("matK", [
            {"name":"matK-390F / matK-1326R","fwd":"CGATCTATTCATTCAATATTTC","rev":"TCTAGCACACGAAAGTCGAAGT",
             "amplicon_bp":"~900 bp","platforms":["Sanger","ONT/PacBio"],
             "use":"Higher resolution alongside rbcL.","refs":[{"label":"CBOL Plant WG","url":"https://doi.org/10.1073/pnas.0905845106"}]}
        ], "Use alongside rbcL for higher resolution.")
        add("trnL (P6 mini-barcode)", [
            {"name":"g / h","fwd":"GGGCAATCCTGAGCCAA","rev":"CCATTGAGTCTCTGCACCTATC",
             "amplicon_bp":"~10–150 bp","platforms":["eDNA-short","Illumina PE250"],
             "use":"Degraded DNA/eDNA friendly mini-barcode.","refs":[{"label":"Taberlet et al. 2007","url":"https://doi.org/10.1093/nar/gkm938"}]}
        ], "Short eDNA-friendly locus for degraded DNA.")
    if "metazoa" in lin or "animalia" in lin:
        add("COI (Folmer region)", [
            {"name":"LCO1490 / HCO2198","fwd":"GGTCAACAAATCATAAAGATATTGG","rev":"TAAACTTCAGGGTGACCAAAAAATCA",
             "amplicon_bp":"~650 bp","platforms":["Sanger","Illumina PE300","ONT/PacBio"],
             "use":"Standard animal barcode across invertebrates/vertebrates.",
             "refs":[{"label":"Folmer et al. 1994","url":"https://doi.org/10.1016/0003-2697(94)90013-2"}]}
        ], "Standard animal barcode.")
        add("12S (eDNA vertebrates)", [
            {"name":"MiFish-U","fwd":"GTCGGTAAAACTCGTGCCAGC","rev":"CATAGTGGGGTATCTAATCCCAGTTTG",
             "amplicon_bp":"~170 bp","platforms":["eDNA-short","Illumina PE250"],
             "use":"Vertebrate eDNA surveys; clade-specific variants exist.",
             "refs":[{"label":"Miya et al. 2015","url":"https://doi.org/10.1098/rsos.150088"}]}
        ], "Great for vertebrate eDNA surveys.")
    if "eukaryota" in lin and not any(k in lin for k in ["animalia","plantae","fungi"]):
        add("18S rRNA (V4 / V9)", [
            {"name":"18S V4 (general)","fwd":"CCAGCASCYGCGGTAATTCC","rev":"ACTTTCGTTCTTGATYRA",
             "amplicon_bp":"~380–420 bp (V4)","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"Pan-eukaryotic survey marker; pick region by group.",
             "refs":[{"label":"Stoeck et al. 2010","url":"https://doi.org/10.1111/j.1365-294X.2010.04695.x"}]}
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

# ----------------- Routes -----------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/search")
def search():
    name = (request.form.get("taxon_name") or "").strip()
    dna = (request.form.get("dna_seq") or "")
    do_blast = request.form.get("blast_hint") == "on"

    # assembly filters
    refseq_only = request.form.get("refseq_only") == "on"
    status_filter = (request.form.get("status_filter") or "any").lower()
    min_contig_n50 = int(request.form.get("min_contig_n50") or 0)
    sort_by = request.form.get("sort_by") or "score"
    selected_accession = request.form.get("selected_accession") or ""

    # NEW: primer helper – intended platform
    intended_platform = request.form.get("intended_platform") or "any"

    tax_info = None
    guessed_df = None
    error = None
    warning = None

    if do_blast and dna.strip():
        try:
            guessed_df = quick_blast_guess(dna)
        except Exception as e:
            warning = f"BLAST hint failed or was skipped: {e}"

    if name:
        tax_info = ncbi_taxon_lookup(name)
    elif guessed_df is not None and not guessed_df.empty:
        m = re.search(r"\b([A-Z][a-z]+ [a-z][a-z\-]+)\b", guessed_df.iloc[0]["title"])
        guess_name = m.group(1) if m else None
        if guess_name:
            tax_info = ncbi_taxon_lookup(guess_name)

    ranked = []
    best_row = None
    links = {}
    suggestions = []

    if not tax_info:
        error = "No taxonomy match found. Provide a clearer organism name or try a longer DNA fragment."
    else:
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

        # primer suggestions with helper metadata
        suggestions = suggest_markers_from_lineage(tax_info.get("lineage", ""))

    ranked_df = pd.DataFrame(ranked, columns=[
        "assembly_accession","assembly_name","refseq_category","assembly_status",
        "contig_n50","scaffold_n50","size_mb","submit_date","update_date","ftp","score"
    ])
    ranked_html = ranked_df.to_html(index=False, classes="table table-sm table-hover align-middle") if not ranked_df.empty else None
    blast_html = (guessed_df.to_html(index=False, classes="table table-sm table-hover align-middle")
                  if isinstance(guessed_df, pd.DataFrame) and not guessed_df.empty else None)

    return render_template(
        "index.html",
        name=name, dna=dna, do_blast=do_blast,
        refseq_only=refseq_only, status_filter=status_filter,
        min_contig_n50=min_contig_n50, sort_by=sort_by,
        selected_accession=best_row["assembly_accession"] if best_row else "",
        intended_platform=intended_platform,
        tax_info=tax_info, ranked=ranked, ranked_html=ranked_html,
        best=best_row, links=links, suggestions=suggestions,
        blast_html=blast_html, error=error, warning=warning
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
