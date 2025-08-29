# app.py
import os
import re
import time
import base64
import json
from io import BytesIO
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, render_template, request, make_response
from dotenv import load_dotenv

# Biopython / NCBI
from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq

# SSL trust (helps on macOS if CERTIFICATE_VERIFY_FAILED)
import ssl, certifi, urllib.request
_ctx = ssl.create_default_context(cafile=certifi.where())
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ctx))
)

# PDF (ReportLab)
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, LongTable

# Primer3
import primer3

# Plotting
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# Optional sequence logo
try:
    import logomaker as lm
    HAS_LOGOMAKER = True
except Exception:
    HAS_LOGOMAKER = False

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
    s = getSampleStyleSheet()
    body = s["BodyText"]; body.fontSize = 9; body.leading = 12
    small = s["BodyText"].clone('Small'); small.fontSize = 8; small.leading = 10
    code = s["BodyText"].clone('Code'); code.fontName = "Courier"; code.fontSize = 8.5; code.leading = 10.5
    h2 = s["Heading2"].clone('H2'); h2.spaceBefore = 8; h2.spaceAfter = 4
    h3 = s["Heading3"].clone('H3'); h3.spaceBefore = 6; h3.spaceAfter = 3
    title = s["Title"].clone('TitleX'); title.fontSize = 18; title.leading = 22
    return {"body": body, "small": small, "code": code, "h2": h2, "h3": h3, "title": title}

def _para(text, style):
    from reportlab.platypus import Paragraph
    txt = "" if text is None else str(text)
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
        try: scaffold_n50 = int(d.get("ScaffoldN50") or 0)
        except Exception: scaffold_n50 = 0
        try: contig_n50 = int(d.get("ContigN50") or 0)
        except Exception: contig_n50 = 0
        try: size_mb = round(float(d.get("SeqLengthSum") or 0) / 1e6, 2)
        except Exception: size_mb = None

        score = 0.0
        if cat and cat.lower() in ("reference genome", "representative genome"):
            score += 100
        score += max(0, 10 - _status_rank(status) * 3)
        if ftp: score += 5
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

# ---------- Primer Recommendations (static, for main page) ----------
def suggest_markers_from_lineage(lineage: str) -> List[Dict]:
    lin = (lineage or "").lower()
    out: List[Dict] = []
    def add(marker, primers, notes): out.append({"marker": marker, "primers": primers, "notes": notes})

    if "bacteria" in lin or "archaea" in lin:
        add("16S rRNA (V3–V4 / V4)", [
            {"name": "341F / 805R", "fwd": "CCTACGGGNGGCWGCAG", "rev": "GACTACHVGGGTATCTAATCC",
             "amplicon_bp": "~460 bp (V3–V4)", "platforms": ["Illumina PE250","Illumina PE300"],
             "use": "Microbiome profiling; widely used."},
            {"name": "515F / 806R", "fwd": "GTGCCAGCMGCCGCGGTAA", "rev": "GGACTACHVGGGTWTCTAAT",
             "amplicon_bp": "~291 bp (V4)", "platforms": ["Illumina PE250","Illumina PE300","eDNA-short"],
             "use": "EMP V4 standard."},
            {"name": "27F / 1492R","fwd":"AGAGTTTGATCMTGGCTCAG","rev":"GGTTACCTTGTTACGACTT",
             "amplicon_bp":"≈1,450 bp", "platforms":["Sanger","ONT/PacBio"],
             "use":"Full-length 16S for isolates."},
        ], "Choose region by platform/read length.")
    if "fungi" in lin or "fungus" in lin:
        add("ITS (ITS1–ITS2)", [
            {"name":"ITS1F / ITS2","fwd":"CTTGGTCATTTAGAGGAAGTAA","rev":"GCTGCGTTCTTCATCGATGC",
             "amplicon_bp":"~300–450 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"Fungal metabarcoding (ITS1)."},
            {"name":"ITS3 / ITS4","fwd":"GCATCGATGAAGAACGCAGC","rev":"TCCTCCGCTTATTGATATGC",
             "amplicon_bp":"~300–500 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],
             "use":"Fungal metabarcoding (ITS2)."},
        ], "Primary fungal barcode.")
    if any(tag in lin for tag in ["viridiplantae","plantae","embryophyta","streptophyta"]):
        add("rbcL",[{"name":"rbcLa-F / rbcLa-R","fwd":"ATGTCACCACAAACAGAGACTAAAGC","rev":"GTAAAATCAAGTCCACCRCG",
             "amplicon_bp":"~550–600 bp","platforms":["Sanger","Illumina PE300"],"use":"Core plant barcode."}],"Core plant barcode.")
        add("matK",[{"name":"matK-390F / matK-1326R","fwd":"CGATCTATTCATTCAATATTTC","rev":"TCTAGCACACGAAAGTCGAAGT",
             "amplicon_bp":"~900 bp","platforms":["Sanger","ONT/PacBio"],"use":"Higher resolution with rbcL."}],"Use with rbcL.")
        add("trnL (P6 mini)",[{"name":"g / h","fwd":"GGGCAATCCTGAGCCAA","rev":"CCATTGAGTCTCTGCACCTATC",
             "amplicon_bp":"~10–150 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Degraded DNA/eDNA."}],"Short eDNA-friendly locus.")
    if "metazoa" in lin or "animalia" in lin:
        add("COI (Folmer region)",[
            {"name":"LCO1490 / HCO2198","fwd":"GGTCAACAAATCATAAAGATATTGG","rev":"TAAACTTCAGGGTGACCAAAAAATCA",
             "amplicon_bp":"~650 bp","platforms":["Sanger","Illumina PE300","ONT/PacBio"],"use":"Standard animal barcode."}
        ], "Standard animal barcode.")
        add("12S (eDNA vertebrates)",[
            {"name":"MiFish-U","fwd":"GTCGGTAAAACTCGTGCCAGC","rev":"CATAGTGGGGTATCTAATCCCAGTTTG",
             "amplicon_bp":"~170 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Vertebrate eDNA surveys."}
        ], "Great for vertebrate eDNA.")
    if "eukaryota" in lin and not any(k in lin for k in ["animalia","plantae","fungi"]):
        add("18S rRNA (V4 / V9)",[
            {"name":"18S V4 (general)","fwd":"CCAGCASCYGCGGTAATTCC","rev":"ACTTTCGTTCTTGATYRA",
             "amplicon_bp":"~380–420 bp (V4)","platforms":["Illumina PE250","Illumina PE300","Sanger"],"use":"Pan-eukaryote survey."}
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

# =================== Primer Wizard helpers ===================

LOCUS_ALIASES = {
    "COI": ['COI', 'cox1', '"cytochrome c oxidase subunit I"'],
    "12S": ['12S'],
    "16S": ['16S'],
    "ITS": ['ITS', '"internal transcribed spacer"'],
    "18S": ['18S'],
    "rbcL": ['rbcL'],
    "matK": ['matK'],
}

CODING_LOCI = {"COI", "rbcL", "matK"}  # AA overlay available for these

def locus_query_string(locus: str) -> str:
    locus = (locus or "").strip()
    if locus in LOCUS_ALIASES:
        parts = LOCUS_ALIASES[locus]
        return "(" + " OR ".join(parts) + ")"
    return locus

def parse_fasta_string(fasta_text: str):
    fasta_text = (fasta_text or "").strip()
    if not fasta_text:
        return []
    records = []
    if fasta_text.startswith(">"):
        cur_id, cur_seq = None, []
        for line in fasta_text.splitlines():
            if line.startswith(">"):
                if cur_id is not None:
                    records.append({"id": cur_id, "seq": sanitize_dna("".join(cur_seq))})
                cur_id = line[1:].strip() or "seq"
                cur_seq = []
            else:
                cur_seq.append(line.strip())
        if cur_id is not None:
            records.append({"id": cur_id, "seq": sanitize_dna("".join(cur_seq))})
    else:
        records.append({"id": "seq1", "seq": sanitize_dna(fasta_text)})
    return [r for r in records if r["seq"]]

def fetch_ncbi_sequences_for_locus(taxon_name: str, locus: str, retmax: int = 100):
    tax = ncbi_taxon_lookup(taxon_name) if taxon_name else None
    if not tax:
        return []
    taxid = tax["taxid"]
    locus_q = locus_query_string(locus)
    term = f'{locus_q} AND txid{taxid}[Organism:exp] AND (biomol_genomic[PROP] OR biomol_mRNA[PROP])'
    _sleep()
    h = Entrez.esearch(db="nucleotide", term=term, retmax=retmax, retmode="xml")
    rec = _entrez_read(h)
    ids = rec.get("IdList", [])
    if not ids:
        return []
    _sleep()
    h2 = Entrez.efetch(db="nucleotide", id=",".join(ids), rettype="fasta", retmode="text")
    fasta_txt = h2.read()
    h2.close()
    seqs = parse_fasta_string(fasta_txt)
    seqs = [s for s in seqs if 80 <= len(s["seq"]) <= 8000]
    return seqs

def entropy_series(seqs):
    if not seqs:
        return []
    L = min(len(s["seq"]) for s in seqs)
    if L == 0:
        return []
    from math import log2
    ent = []
    for i in range(L):
        col = [s["seq"][i] for s in seqs]
        counts = {b: col.count(b) for b in "ACGT"}
        total = sum(counts.values()) or 1
        H = 0.0
        for b in "ACGT":
            p = counts[b] / total
            if p > 0:
                H -= p * log2(p)
        ent.append(round(H, 3))
    return ent

# ---------- QC utilities (#1 & #3) ----------
def _gc_clamp_ok(seq: str, min_gc3: int, max_gc3: int, clamp_span: int = 3) -> Tuple[int, bool]:
    tail = seq[-clamp_span:]
    gc_count = sum(1 for b in tail if b in "GC")
    return gc_count, (min_gc3 <= gc_count <= max_gc3)

def _max_homopolymer_run(seq: str) -> int:
    best, cur, prev = 0, 0, ""
    for b in seq:
        if b == prev:
            cur += 1
        else:
            prev, cur = b, 1
        best = max(best, cur)
    return best

def _homopolymer_ok(seq: str, max_at: int, max_gc: int) -> bool:
    runs = {"A":0,"C":0,"G":0,"T":0}
    cur_b, cur_len = "", 0
    ok = True
    for b in seq:
        if b == cur_b:
            cur_len += 1
        else:
            if cur_b:
                runs[cur_b] = max(runs[cur_b], cur_len)
            cur_b, cur_len = b, 1
    if cur_b:
        runs[cur_b] = max(runs[cur_b], cur_len)
    if max(runs["A"], runs["T"]) > max_at: ok = False
    if max(runs["G"], runs["C"]) > max_gc: ok = False
    return ok

def _dinuc_ok(seq: str, max_repeats: int) -> bool:
    # disallow > max_repeats of any dinucleotide like ATATAT...
    if len(seq) < 4:
        return True
    for i in range(len(seq)-3):
        di = seq[i:i+2]
        reps = 1
        j = i+2
        while j+2 <= len(seq) and seq[j:j+2] == di:
            reps += 1
            j += 2
        if reps > max_repeats:
            return False
    return True

def _thermo_dGs(left: str, right: str, mv: float, dv: float, dntp: float, dna_nM: float) -> Dict[str, float]:
    # primer3 returns objects with .dg (kcal/mol) and .structure_found
    kwargs = dict(mv_conc=mv, dv_conc=dv, dntp_conc=dntp, dna_conc=dna_nM, temp_c=60)
    try:
        hpL = primer3.calcHairpin(left, **kwargs); dg_hpL = getattr(hpL, "dg", 0.0) or 0.0
    except Exception:
        dg_hpL = 0.0
    try:
        hpR = primer3.calcHairpin(right, **kwargs); dg_hpR = getattr(hpR, "dg", 0.0) or 0.0
    except Exception:
        dg_hpR = 0.0
    try:
        sdL = primer3.calcHomodimer(left, **kwargs); dg_sdL = getattr(sdL, "dg", 0.0) or 0.0
    except Exception:
        dg_sdL = 0.0
    try:
        sdR = primer3.calcHomodimer(right, **kwargs); dg_sdR = getattr(sdR, "dg", 0.0) or 0.0
    except Exception:
        dg_sdR = 0.0
    try:
        het = primer3.calcHeterodimer(left, right, **kwargs); dg_het = getattr(het, "dg", 0.0) or 0.0
    except Exception:
        dg_het = 0.0
    return {
        "dg_hp_left": float(dg_hpL),
        "dg_hp_right": float(dg_hpR),
        "dg_self_left": float(dg_sdL),
        "dg_self_right": float(dg_sdR),
        "dg_hetero": float(dg_het),
    }

def _dG_pass(dgs: Dict[str,float], th_hp: float, th_self: float, th_hetero: float) -> bool:
    # thresholds are minimum acceptable (e.g., -2, -6, -7); we want dg >= threshold (less negative is better)
    return (
        dgs["dg_hp_left"]   >= th_hp and
        dgs["dg_hp_right"]  >= th_hp and
        dgs["dg_self_left"] >= th_self and
        dgs["dg_self_right"]>= th_self and
        dgs["dg_hetero"]    >= th_hetero
    )

def _kmer_uniqueness_3p(primer: str, seqs: List[Dict], k: int = 11) -> int:
    """Count exact matches of the 3'-terminal k-mer across all sequences (both strands). Lower is better."""
    if not primer or k <= 0 or len(primer) < k:
        return 0
    kmer = primer[-k:].upper()
    rc_kmer = str(Seq(kmer).reverse_complement())
    count = 0
    for s in seqs:
        t = s["seq"]
        count += t.count(kmer) + t.count(rc_kmer)
        rc = str(Seq(t).reverse_complement())
        count += rc.count(kmer) + rc.count(rc_kmer)
    return count

# ---------- Primer design ----------
def design_primers_on_region(
    seq: str,
    target_start: int,
    target_len: int,
    amplicon_min: int,
    amplicon_opt: int,
    amplicon_max: int,
    tm_min: float,
    tm_opt: float,
    tm_max: float,
    gc_min: float,
    gc_max: float,
    len_min: int,
    len_opt: int,
    len_max: int,
    mv_mM: float,
    dv_mM: float,
    dntp_mM: float,
    dna_nM: float,
    num_return: int = 30
) -> List[Dict]:
    seq = sanitize_dna(seq)
    if not seq:
        return []
    target_len = max(1, min(len(seq), target_len or len(seq)))

    params = {
        'SEQUENCE_ID': 'target',
        'SEQUENCE_TEMPLATE': seq,
        'SEQUENCE_TARGET': [target_start, target_len],
    }
    opts = {
        'PRIMER_TASK': 'generic',
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_MIN_SIZE': len_min,
        'PRIMER_OPT_SIZE': len_opt,
        'PRIMER_MAX_SIZE': len_max,
        'PRIMER_PRODUCT_SIZE_RANGE': [[amplicon_min, amplicon_max]],
        'PRIMER_PRODUCT_OPT_SIZE': amplicon_opt,
        'PRIMER_MIN_TM': tm_min, 'PRIMER_OPT_TM': tm_opt, 'PRIMER_MAX_TM': tm_max,
        'PRIMER_MIN_GC': gc_min, 'PRIMER_MAX_GC': gc_max,
        'PRIMER_MAX_POLY_X': 100,  # we enforce homopolymers custom per-base (A/T vs G/C)
        'PRIMER_MAX_SELF_ANY_TH': 45.0,
        'PRIMER_MAX_SELF_END_TH': 35.0,
        'PRIMER_MAX_HAIRPIN_TH': 24.0,
        'PRIMER_NUM_RETURN': int(num_return),

        # Thermo environment
        'PRIMER_SALT_MONOVALENT': mv_mM,
        'PRIMER_SALT_DIVALENT': dv_mM,
        'PRIMER_DNTP_CONC': dntp_mM,
        'PRIMER_DNA_CONC': dna_nM,
    }
    res = primer3.bindings.design_primers(params, opts)
    out = []
    n = res.get('PRIMER_PAIR_NUM_RETURNED', 0)
    for i in range(n):
        left_pos, left_len = res.get(f'PRIMER_LEFT_{i}', [0,0])
        right_pos, right_len = res.get(f'PRIMER_RIGHT_{i}', [0,0])
        out.append({
            'left':  res.get(f'PRIMER_LEFT_{i}_SEQUENCE'),
            'right': res.get(f'PRIMER_RIGHT_{i}_SEQUENCE'),
            'tm_left': round(res.get(f'PRIMER_LEFT_{i}_TM', 0.0), 2),
            'tm_right': round(res.get(f'PRIMER_RIGHT_{i}_TM', 0.0), 2),
            'gc_left': round(res.get(f'PRIMER_LEFT_{i}_GC_PERCENT', 0.0), 1),
            'gc_right': round(res.get(f'PRIMER_RIGHT_{i}_GC_PERCENT', 0.0), 1),
            'len_left': left_len,
            'len_right': right_len,
            'amplicon_len': res.get(f'PRIMER_PAIR_{i}_PRODUCT_SIZE'),
            'penalty': round(res.get(f'PRIMER_PAIR_{i}_PENALTY', 0.0), 2),
            'left_pos': left_pos,
            'right_pos': right_pos,
        })
    out = [r for r in out if r['left'] and r['right']]
    out.sort(key=lambda r: (r['penalty'], abs((r['amplicon_len'] or amplicon_opt) - amplicon_opt)))
    return out

def insilico_coverage(primer_left: str, primer_right: str, seqs, max_mismatch=1):
    if not seqs:
        return {"hits":0, "total":0, "pct":0.0}
    left = primer_left.upper()
    right = primer_right.upper()
    hits = 0
    for s in seqs:
        t = s["seq"]
        if len(t) < 80:
            continue
        # forward primer
        found_left = False
        for i in range(0, len(t)-len(left)+1):
            seg = t[i:i+len(left)]
            if seg[-1] != left[-1]:  # 3' protection
                continue
            mism = sum(1 for a,b in zip(left, seg) if a != b)
            if mism <= max_mismatch:
                found_left = True
                break
        if not found_left:
            continue
        # reverse primer on reverse complement
        rc = str(Seq(t).reverse_complement())
        found_right = False
        for i in range(0, len(rc)-len(right)+1):
            seg = rc[i:i+len(right)]
            if seg[-1] != right[-1]:
                continue
            mism = sum(1 for a,b in zip(right, seg) if a != b)
            if mism <= max_mismatch:
                found_right = True
                break
        if found_left and found_right:
            hits += 1
    total = len([s for s in seqs if len(s["seq"]) >= 80])
    pct = round(100.0 * hits / total, 2) if total else 0.0
    return {"hits": hits, "total": total, "pct": pct}

# === Visualization helpers ===
def _encode_fig_to_b64(fig, dpi=140):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def make_entropy_plot(seqs, designed, locus: str):
    if not seqs:
        return None
    ent = entropy_series(seqs)
    L = len(ent)
    x = np.arange(L)

    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.bar(x, ent, width=1.0, color="#66c2d0")
    ax.set_ylim(0, max(1.8, max(ent) + 0.1))
    ax.set_xlim(0, L)
    ax.set_ylabel("Entropy", fontsize=10)
    ax.set_xlabel("Position (bp)", fontsize=10)
    ax.set_xticks(np.arange(0, L+1, 50))
    ax.grid(axis='y', alpha=0.15)

    # Overlays: up to 5 candidates with labels
    for idx, d in enumerate((designed or [])[:5], start=1):
        lp = d.get("left_pos", 0)
        ll = d.get("len_left", 0)
        rp = d.get("right_pos", 0)
        rl = d.get("len_right", 0)
        amp = d.get("amplicon_len", None)

        # forward primer arrow + label
        ax.annotate("",
            xy=(lp + ll, -0.05), xycoords=("data","axes fraction"),
            xytext=(lp, -0.05), textcoords=("data","axes fraction"),
            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=2))
        ax.text(lp, -0.12, f"P{idx} F", color="#22c55e", fontsize=8,
                ha="left", va="top", transform=ax.get_xaxis_transform())

        # reverse primer arrow + label
        ax.annotate("",
            xy=(rp, -0.10), xycoords=("data","axes fraction"),
            xytext=(rp + rl, -0.10), textcoords=("data","axes fraction"),
            arrowprops=dict(arrowstyle="->", color="#ef4444", lw=2))
        ax.text(rp + rl, -0.17, f"P{idx} R", color="#ef4444", fontsize=8,
                ha="right", va="top", transform=ax.get_xaxis_transform())

        # Amplicon span
        span_start = lp + ll
        span_end = rp
        if span_end <= span_start and amp:
            span_end = span_start + int(amp)
        span_start = max(0, span_start)
        span_end = min(L, span_end)
        if span_end > span_start:
            ax.axvspan(span_start, span_end, color="#a9def9", alpha=0.18)

    # Codon/AA track for coding loci
    if locus in CODING_LOCI and L >= 3:
        ref = seqs[0]["seq"][:(L//3)*3]
        aa = str(Seq(ref).translate())
        codons = len(ref) // 3
        yline = -0.28
        ax.text(0, yline+0.02, "AA:", transform=ax.transAxes, fontsize=8,
                color="#94a3b8", ha="left", va="top")
        for i in range(codons+1):
            bp = i * 3
            ax.axvline(bp, ymin=0, ymax=0.02, color="#94a3b833", lw=0.8)
        for i in range(0, codons, 3):
            bp = i * 3 + 1
            ax.text(bp, -0.34, aa[i], transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color="#cbd5e1", ha="center", va="top")

    fig.tight_layout()
    return _encode_fig_to_b64(fig)

def make_logo_plot(seqs):
    if not HAS_LOGOMAKER or not seqs:
        return None
    L = min(len(s["seq"]) for s in seqs)
    if L == 0: return None
    counts = pd.DataFrame(0, index=list("ACGT"), columns=range(L))
    for s in seqs[:200]:
        for i, b in enumerate(s["seq"][:L]):
            if b in "ACGT":
                counts.at[b, i] += 1
    counts = counts.T
    fig, ax = plt.subplots(figsize=(13.5, 3.1))
    lm.Logo(counts, ax=ax, shade_below=.5, fade_below=.5, color_scheme="classic")
    ax.set_xlim(0, L)
    ax.set_xticks(np.arange(0, L+1, 50))
    ax.set_xlabel("Position (bp)")
    ax.set_ylabel("Bits")
    ax.set_ylim(0, 2)
    ax.grid(axis='y', alpha=0.15)
    fig.tight_layout()
    return _encode_fig_to_b64(fig)

# ----------------- Routes: Main -----------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/search")
def search():
    ctx = build_context_from_request(request.form)

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

# ----------------- Exports: CSV/PDF -----------------
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
    buff = BytesIO()
    margin = 28
    doc = SimpleDocTemplate(
        buff,
        pagesize=landscape(LETTER),
        leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin
    )

    story = []

    title = f"Marker Finder Report — {ctx['tax_info']['scientific_name']} (taxid {ctx['tax_info']['taxid']})"
    story += [_para(title, st["title"]), Spacer(1, 8)]

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

    if ctx["suggestions"]:
        story += [_para("<b>Recommended Markers &amp; Primers</b>", st["h2"])]
        for s in ctx["suggestions"]:
            story += [_para(s["marker"], st["h3"])]
            if s.get("notes"):
                story += [_para(s["notes"], st["small"])]

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
                    _para(p.get("fwd",""), st["code"]),
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

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        txt = f"Page {doc.page}"
        canvas.drawRightString(doc.pagesize[0] - 28, 16, txt)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pdf = buff.getvalue()
    buff.close()

    safe = ctx["tax_info"]["scientific_name"].replace(" ", "_")
    resp = make_response(pdf)
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe}_marker_finder_report.pdf"'
    return resp

# =================== Primer Wizard pages ===================

@app.get("/designer")
def designer():
    ctx = {
        "goal": "metabarcoding",
        "platform": "Illumina PE250",
        "amp_min": 120, "amp_opt": 160, "amp_max": 320,
        "tm_min": 58.0, "tm_opt": 60.0, "tm_max": 62.0,
        "gc_min": 40.0, "gc_max": 60.0,
        "len_min": 18, "len_opt": 20, "len_max": 26,

        # NEW: constraints & thermo
        "enforce_gc_clamp": "on",
        "min_gc3": 1, "max_gc3": 2,
        "max_run_at": 4, "max_run_gc": 4,
        "max_dinuc": 3,
        "dg_hp_min": -2.0, "dg_self_min": -6.0, "dg_hetero_min": -7.0,
        "mv_mM": 50.0, "dv_mM": 1.5, "dntp_mM": 0.2, "dna_nM": 250.0,

        # NEW: k-mer uniqueness
        "kmer_len": 11, "kmer_max_hits": 1,

        "taxon": "", "locus": "COI",
        "source": "ncbi",
        "max_seqs": 100,
        "viz": "entropy",  # 'entropy' or 'logo'
        "designed": None, "seqs_count": 0,
        "entropy_png": None, "logo_png": None,
        "warning": None, "error": None,
        "has_logomaker": HAS_LOGOMAKER,
    }
    return render_template("designer.html", **ctx)

@app.post("/designer/run")
def designer_run():
    f = request.form

    goal = f.get("goal") or "metabarcoding"
    platform = f.get("platform") or "Illumina PE250"
    amp_min = int(f.get("amp_min") or 120)
    amp_opt = int(f.get("amp_opt") or 160)
    amp_max = int(f.get("amp_max") or 320)
    tm_min = float(f.get("tm_min") or 58)
    tm_opt = float(f.get("tm_opt") or 60)
    tm_max = float(f.get("tm_max") or 62)
    gc_min = float(f.get("gc_min") or 40)
    gc_max = float(f.get("gc_max") or 60)
    len_min = int(f.get("len_min") or 18)
    len_opt = int(f.get("len_opt") or 20)
    len_max = int(f.get("len_max") or 26)

    enforce_gc_clamp = f.get("enforce_gc_clamp") == "on"
    min_gc3 = int(f.get("min_gc3") or 1)
    max_gc3 = int(f.get("max_gc3") or 2)
    max_run_at = int(f.get("max_run_at") or 4)
    max_run_gc = int(f.get("max_run_gc") or 4)
    max_dinuc = int(f.get("max_dinuc") or 3)

    dg_hp_min = float(f.get("dg_hp_min") or -2.0)
    dg_self_min = float(f.get("dg_self_min") or -6.0)
    dg_hetero_min = float(f.get("dg_hetero_min") or -7.0)

    mv_mM = float(f.get("mv_mM") or 50.0)
    dv_mM = float(f.get("dv_mM") or 1.5)
    dntp_mM = float(f.get("dntp_mM") or 0.2)
    dna_nM = float(f.get("dna_nM") or 250.0)

    kmer_len = int(f.get("kmer_len") or 11)
    kmer_max_hits = int(f.get("kmer_max_hits") or 1)

    taxon = (f.get("taxon") or "").strip()
    locus = (f.get("locus") or "COI").strip()
    source = f.get("source") or "ncbi"
    max_seqs = int(f.get("max_seqs") or 100)
    viz = f.get("viz") or "entropy"

    # Load sequences
    seqs: List[Dict] = []
    warning = None
    error = None
    if source == "ncbi":
        try:
            seqs = fetch_ncbi_sequences_for_locus(taxon, locus, retmax=max_seqs)
            if not seqs:
                warning = "No sequences returned from NCBI for that taxon/locus filter."
        except Exception as e:
            error = f"NCBI fetch error: {e}"
    elif source == "paste":
        seqs = parse_fasta_string(f.get("fasta_text") or "")
        if not seqs:
            warning = "No sequences parsed from pasted text."
    else:
        file = request.files.get("fasta_file")
        if file and file.filename:
            data = file.read().decode("utf-8", errors="ignore")
            seqs = parse_fasta_string(data)
        if not seqs:
            warning = "No sequences parsed from uploaded file."

    ref_seq = seqs[0]["seq"] if seqs else ""
    designed = []
    entropy_png = None
    logo_png = None

    if ref_seq:
        designed = design_primers_on_region(
            ref_seq, 0, len(ref_seq),
            amplicon_min=amp_min, amplicon_opt=amp_opt, amplicon_max=amp_max,
            tm_min=tm_min, tm_opt=tm_opt, tm_max=tm_max,
            gc_min=gc_min, gc_max=gc_max,
            len_min=len_min, len_opt=len_opt, len_max=len_max,
            mv_mM=mv_mM, dv_mM=dv_mM, dntp_mM=dntp_mM, dna_nM=dna_nM,
            num_return=30
        )

        # Per-primer QC + coverage + k-mer uniqueness
        for d in designed:
            # GC clamp
            gc3_f, ok_gc3_f = _gc_clamp_ok(d["left"], min_gc3, max_gc3)
            gc3_r, ok_gc3_r = _gc_clamp_ok(d["right"], min_gc3, max_gc3)
            d["gc3_f"] = gc3_f; d["gc3_r"] = gc3_r
            d["gc3_ok"] = (ok_gc3_f and ok_gc3_r) if enforce_gc_clamp else True

            # Homopolymers & dinucleotides
            d["homopoly_ok_f"] = _homopolymer_ok(d["left"], max_run_at, max_run_gc)
            d["homopoly_ok_r"] = _homopolymer_ok(d["right"], max_run_at, max_run_gc)
            d["dinuc_ok_f"] = _dinuc_ok(d["left"], max_dinuc)
            d["dinuc_ok_r"] = _dinuc_ok(d["right"], max_dinuc)

            # Thermo ΔG
            dgs = _thermo_dGs(d["left"], d["right"], mv_mM, dv_mM, dntp_mM, dna_nM)
            d.update(dgs)
            d["dg_ok"] = _dG_pass(dgs, dg_hp_min, dg_self_min, dg_hetero_min)

            # 3'-end uniqueness
            d["kmer_hits_f"] = _kmer_uniqueness_3p(d["left"], seqs, k=kmer_len)
            d["kmer_hits_r"] = _kmer_uniqueness_3p(d["right"], seqs, k=kmer_len)
            d["kmer_ok"] = (d["kmer_hits_f"] <= kmer_max_hits) and (d["kmer_hits_r"] <= kmer_max_hits)

            # Coverage across pulled sequences
            cov = insilico_coverage(d["left"], d["right"], seqs, max_mismatch=1)
            d["coverage_pct"] = cov["pct"]
            d["coverage_hits"] = cov["hits"]
            d["coverage_total"] = cov["total"]

        # Build plots
        entropy_png = make_entropy_plot(seqs[:50], designed, locus=locus)
        if HAS_LOGOMAKER:
            logo_png = make_logo_plot(seqs[:200])
    else:
        warning = warning or "No reference sequence available to design primers."

    ctx = {
        "goal": goal, "platform": platform,
        "amp_min": amp_min, "amp_opt": amp_opt, "amp_max": amp_max,
        "tm_min": tm_min, "tm_opt": tm_opt, "tm_max": tm_max,
        "gc_min": gc_min, "gc_max": gc_max,
        "len_min": len_min, "len_opt": len_opt, "len_max": len_max,

        "enforce_gc_clamp": "on" if enforce_gc_clamp else "",
        "min_gc3": min_gc3, "max_gc3": max_gc3,
        "max_run_at": max_run_at, "max_run_gc": max_run_gc,
        "max_dinuc": max_dinuc,
        "dg_hp_min": dg_hp_min, "dg_self_min": dg_self_min, "dg_hetero_min": dg_hetero_min,
        "mv_mM": mv_mM, "dv_mM": dv_mM, "dntp_mM": dntp_mM, "dna_nM": dna_nM,

        "kmer_len": kmer_len, "kmer_max_hits": kmer_max_hits,

        "taxon": taxon, "locus": locus,
        "source": source, "max_seqs": max_seqs,
        "viz": viz,
        "designed": designed, "seqs_count": len(seqs),
        "entropy_png": entropy_png, "logo_png": logo_png,
        "warning": warning, "error": error,
        "has_logomaker": HAS_LOGOMAKER,
    }
    return render_template("designer.html", **ctx)

@app.post("/designer/blast")
def designer_blast():
    primer = (request.form.get("primer") or "").strip().upper()
    if not primer or len(primer) < 16:
        return ("Primer too short for BLAST.", 400)
    try:
        rh = NCBIWWW.qblast("blastn", "nt", primer, hitlist_size=10)
        record = NCBIXML.read(rh)
        hits = []
        for aln in record.alignments:
            hsp = aln.hsps[0]
            hits.append({
                "title": aln.title[:140],
                "e": f"{hsp.expect:.1e}",
                "identity": round(100 * hsp.identities / max(1, hsp.align_length), 1),
                "len": aln.length
            })
        rows = "".join(f"<tr><td>{h['identity']}%</td><td>{h['e']}</td><td>{h['len']}</td><td>{h['title']}</td></tr>" for h in hits)
        html = f"""
        <div class="table-responsive"><table class="table table-sm table-hover">
        <thead><tr><th>%ID</th><th>E</th><th>Len</th><th>Title</th></tr></thead>
        <tbody>{rows or '<tr><td colspan=4>No hits</td></tr>'}</tbody></table></div>
        """
        resp = make_response(html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    except Exception as e:
        return (f"BLAST failed: {e}", 500)

@app.post("/designer/export/csv")
def designer_export_csv():
    designed_json = request.form.get("designed_json") or "[]"
    try:
        designed = pd.DataFrame(json.loads(designed_json))
    except Exception:
        return ("Bad payload.", 400)
    if designed.empty:
        return ("Nothing to export.", 400)
    csv_bytes = designed.to_csv(index=False).encode("utf-8")
    resp = make_response(csv_bytes)
    resp.headers["Content-Type"] = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="designed_primers.csv"'
    return resp

# ----------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
