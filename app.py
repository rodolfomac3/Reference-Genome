# app.py
import os
import re
import time
import base64
import json
from io import BytesIO
from functools import lru_cache
from typing import Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, send_file

import pandas as pd
from flask import render_template, make_response
from dotenv import load_dotenv

# Biopython / NCBI
from Bio import Entrez
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Seq import Seq
from Bio import pairwise2  # <-- Upgrade A: lightweight alignments

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

# primer3 v2 uses snake_case; earlier versions used camelCase.
# These aliases pick whichever exists so we stay compatible.
_p3_calc_hairpin     = getattr(primer3, "calc_hairpin",     getattr(primer3, "calcHairpin"))
_p3_calc_homodimer   = getattr(primer3, "calc_homodimer",   getattr(primer3, "calcHomodimer"))
_p3_calc_heterodimer = getattr(primer3, "calc_heterodimer", getattr(primer3, "calcHeterodimer"))

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
    # keep ACGT plus IUPAC degeneracy codes (used throughout)
    return re.sub(r"[^ACGTURYKMSWBDHVN\-]", "", (seq or "").upper())

# -------- IUPAC / degeneracy helpers (Upgrade B) --------
_IUPAC = {
    "A":{"A"}, "C":{"C"}, "G":{"G"}, "T":{"T"},
    "R":{"A","G"}, "Y":{"C","T"}, "S":{"G","C"}, "W":{"A","T"},
    "K":{"G","T"}, "M":{"A","C"},
    "B":{"C","G","T"}, "D":{"A","G","T"}, "H":{"A","C","T"}, "V":{"A","C","G"},
    "N":{"A","C","G","T"},
    "-":{"-"}  # treat gap as its own (for alignment paths)
}
def _iupac_match(primer_char: str, template_char: str) -> bool:
    p = primer_char.upper(); t = template_char.upper()
    return t in _IUPAC.get(p, {p})

def _revcomp(s: str) -> str:
    return str(Seq(s).reverse_complement())

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

# --- Robust genome-size extraction + fallback to FTP stats ---
def _to_int(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        try:
            return int(x.replace(",", "").strip())
        except Exception:
            return None
    return None

def _extract_genome_size_mb_from_docsum(d: dict):
    for k in ("TotalSequenceLength", "GenomeSize", "SeqLengthSum", "UCSCSequenceLengthSum"):
        v = _to_int(d.get(k))
        if v and v > 0:
            return round(v / 1_000_000.0, 2)
    stats = d.get("AssemblyStats") or {}
    v = _to_int(stats.get("total_sequence_length") or stats.get("total_length"))
    if v and v > 0:
        return round(v / 1_000_000.0, 2)
    return None

def _fetch_size_mb_from_ftp_stats(ftp_url: str):
    if not ftp_url:
        return None
    try:
        https = ftp_url.replace("ftp://", "https://")
        base = https.rsplit("/", 1)[-1]
        stats_url = f"{https}/{base}_assembly_stats.txt"
        with urllib.request.urlopen(stats_url, context=_ctx, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        total_bp = None
        for line in text.splitlines():
            if "total-length" in line:
                parts = re.split(r"\s+", line.strip())
                for i, tok in enumerate(parts):
                    if tok == "total-length" and i + 1 < len(parts):
                        total_bp = _to_int(parts[i + 1]); break
            if total_bp: break
        if total_bp and total_bp > 0:
            return round(total_bp / 1_000_000.0, 2)
    except Exception:
        return None
    return None

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
        try: size_mb = _extract_genome_size_mb_from_docsum(d)
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
def _refs(*items):
    out = []
    for r in items:
        if isinstance(r, dict) and "url" in r:
            out.append({"label": r.get("label") or "Ref", "url": r["url"]})
        elif isinstance(r, str):
            s = r.strip()
            if s.isdigit():
                out.append({"label": f"PMID {s}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{s}/"})
            elif s.startswith("10."):
                out.append({"label": "DOI", "url": f"https://doi.org/{s}"})
            else:
                out.append({"label": "Ref", "url": s})
    return out

def suggest_markers_from_lineage(lineage: str) -> List[Dict]:
    lin = (lineage or "").lower()
    out: List[Dict] = []

    def add(marker: str, primers: List[Dict], notes: str = ""):
        out.append({"marker": marker, "primers": primers, "notes": notes})

    if "bacteria" in lin or "archaea" in lin:
        add(
            "16S rRNA (V3–V4 / V4)",
            [
                {"name":"341F / 805R","fwd":"CCTACGGGNGGCWGCAG","rev":"GACTACHVGGGTATCTAATCC","amplicon_bp":"~460 bp (V3–V4)","platforms":["Illumina PE250","Illumina PE300"],"use":"Microbiome profiling; widely used.","refs":_refs("23344259")},
                {"name":"515F / 806R","fwd":"GTGCCAGCMGCCGCGGTAA","rev":"GGACTACHVGGGTWTCTAAT","amplicon_bp":"~291 bp (V4)","platforms":["Illumina PE250","Illumina PE300","eDNA-short"],"use":"EMP V4 standard.","refs":_refs("21544103")},
                {"name":"27F / 1492R","fwd":"AGAGTTTGATCMTGGCTCAG","rev":"GGTTACCTTGTTACGACTT","amplicon_bp":"≈1,450 bp","platforms":["Sanger","ONT/PacBio"],"use":"Full-length 16S for isolates.","refs":_refs({"label":"Lane 1991","url":"https://doi.org/10.1016/B978-0-12-672180-9.50023-8"})}
            ],
            "Choose region by platform/read length.",
        )

    if "fungi" in lin or "fungus" in lin:
        add(
            "ITS (ITS1–ITS2)",
            [
                {"name":"ITS1F / ITS2","fwd":"CTTGGTCATTTAGAGGAAGTAA","rev":"GCTGCGTTCTTCATCGATGC","amplicon_bp":"~300–450 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],"use":"Fungal metabarcoding (ITS1).","refs":_refs("8486376")},
                {"name":"ITS3 / ITS4","fwd":"GCATCGATGAAGAACGCAGC","rev":"TCCTCCGCTTATTGATATGC","amplicon_bp":"~300–500 bp","platforms":["Illumina PE250","Illumina PE300","Sanger"],"use":"Fungal metabarcoding (ITS2).","refs":_refs("1971545")},
            ],
            "Primary fungal barcode.",
        )

    if any(tag in lin for tag in ["viridiplantae","plantae","embryophyta","streptophyta"]):
        add("rbcL",[{"name":"rbcLa-F / rbcLa-R","fwd":"ATGTCACCACAAACAGAGACTAAAGC","rev":"GTAAAATCAAGTCCACCRCG","amplicon_bp":"~550–600 bp","platforms":["Sanger","Illumina PE300"],"use":"Core plant barcode.","refs":_refs("18431400")}],"Core plant barcode.")
        add("matK",[{"name":"matK-390F / matK-1326R","fwd":"CGATCTATTCATTCAATATTTC","rev":"TCTAGCACACGAAAGTCGAAGT","amplicon_bp":"~900 bp","platforms":["Sanger","ONT/PacBio"],"use":"Higher resolution with rbcL.","refs":_refs("18431400")}],"Use with rbcL.")
        add("trnL (P6 mini)",[{"name":"g / h","fwd":"GGGCAATCCTGAGCCAA","rev":"CCATTGAGTCTCTGCACCTATC","amplicon_bp":"~10–150 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Degraded DNA/eDNA.","refs":_refs("9336234")}],"Short eDNA-friendly locus.")

    if "metazoa" in lin or "animalia" in lin:
        add("COI (Folmer region)",[{"name":"LCO1490 / HCO2198","fwd":"GGTCAACAAATCATAAAGATATTGG","rev":"TAAACTTCAGGGTGACCAAAAAATCA","amplicon_bp":"~650 bp","platforms":["Sanger","Illumina PE300","ONT/PacBio"],"use":"Standard animal barcode.","refs":_refs("7881515")}],"Standard animal barcode.")
        add("12S (eDNA vertebrates)",[{"name":"MiFish-U","fwd":"GTCGGTAAAACTCGTGCCAGC","rev":"CATAGTGGGGTATCTAATCCCAGTTTG","amplicon_bp":"~170 bp","platforms":["eDNA-short","Illumina PE250"],"use":"Vertebrate eDNA surveys.","refs":_refs("10.1093/gigascience/giy123")}],"Great for vertebrate eDNA.")

    if "eukaryota" in lin and not any(k in lin for k in ["animalia","plantae","fungi"]):
        add("18S rRNA (V4 / V9)",[{"name":"18S V4 (general)","fwd":"CCAGCASCYGCGGTAATTCC","rev":"ACTTTCGTTCTTGATYRA","amplicon_bp":"~380–420 bp (V4)","platforms":["Illumina PE250","Illumina PE300","Sanger"],"use":"Pan-eukaryote survey.","refs":_refs("10.1111/j.1365-294X.2010.04695.x")}],"Pan-eukaryotic survey marker.")
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

# ------------ Alignment & entropy (Upgrade A) ------------
def _align_to_ref_pairwise2(ref: str, seq: str) -> Tuple[str, str]:
    """
    Global alignment of seq to ref (ACGTN, gaps allowed). Returns (ref_aln, seq_aln) with equal length.
    Scoring tuned light: match=2, mismatch=-1, gap open=-4, gap extend=-0.5
    """
    if not ref or not seq:
        return ref, seq
    alignments = pairwise2.align.globalms(ref, seq, 2, -1, -4, -0.5, one_alignment_only=True)
    if not alignments:
        return ref, seq
    a = alignments[0]
    return a.seqA, a.seqB

def _multiple_to_ref_alignment(seqs: List[Dict]) -> List[Dict]:
    """
    Align every sequence to the first sequence as reference.
    Returns records with 'id' and 'seq' replaced by aligned versions (gaps '-').
    """
    if not seqs:
        return []
    ref = sanitize_dna(seqs[0]["seq"])
    out = [{"id": seqs[0]["id"], "seq": ref}]
    for s in seqs[1:]:
        a_ref, a_q = _align_to_ref_pairwise2(ref, sanitize_dna(s["seq"]))
        # pad reference of the first added if alignment extended
        if len(a_ref) != len(out[0]["seq"]):
            # re-map all existing aligned seqs to new ref with additional gaps
            new_ref = a_ref
            # build a mapping from old ref to new ref positions via pairwise2 again
            # But simpler: realign previous ref to new_ref and project gaps into all out sequences
            prev_ref = out[0]["seq"]
            pr2, nr2 = _align_to_ref_pairwise2(new_ref, prev_ref)  # align prev_ref to new_ref
            # inject gaps into each sequence in out, following alignment of prev_ref to new_ref
            gap_positions = [i for i,(x,y) in enumerate(zip(pr2, nr2)) if x!='-' and y=='-']
            keep_positions = [i for i,(x,y) in enumerate(zip(pr2, nr2)) if x!='-']
            # rebuild each existing seq by inserting '-' where prev_ref has gaps relative to new_ref
            rebuilt = []
            for rec in out:
                s_old = rec["seq"]
                # expand s_old over keep_positions and insert '-' at gap positions
                res_chars = []
                idx_old = 0
                for i,(x,y) in enumerate(zip(pr2, nr2)):
                    if x=='-' and y!='-':
                        res_chars.append('-')
                    elif x!='-' and y=='-':
                        # should not happen with our construction
                        res_chars.append('-')
                    else:
                        # x != '-' and y != '-': consume from s_old
                        res_chars.append(s_old[idx_old] if idx_old < len(s_old) else '-')
                        idx_old += 1
                rebuilt.append({"id": rec["id"], "seq": "".join(res_chars)})
            out = rebuilt
            # finally set new ref in slot 0
            out[0]["seq"] = new_ref
            ref = new_ref
            # Now align current seq to new ref again to keep in same coordinate space
            a_ref, a_q = _align_to_ref_pairwise2(ref, sanitize_dna(s["seq"]))
        out.append({"id": s["id"], "seq": a_q})
    return out

def entropy_series(seqs: List[Dict]) -> List[float]:
    """
    Shannon entropy per aligned column (Upgrade A).
    If sequences are unaligned (varying length without gaps), align them first to the first sequence.
    """
    if not seqs:
        return []
    # If any seq lacks gaps and lengths differ, perform alignment
    needs_align = (len({len(r["seq"]) for r in seqs}) != 1) or any('-' in r["seq"] for r in seqs)
    aligned = _multiple_to_ref_alignment(seqs) if needs_align else seqs
    L = min(len(s["seq"]) for s in aligned)
    if L == 0:
        return []
    from math import log2
    ent = []
    for i in range(L):
        col = [s["seq"][i] for s in aligned]
        # ignore gaps in entropy (only A/C/G/T)
        bases = [b for b in col if b in "ACGT"]
        if not bases:
            ent.append(0.0); continue
        total = len(bases)
        H = 0.0
        for b in "ACGT":
            p = bases.count(b)/total
            if p > 0:
                H -= p * log2(p)
        ent.append(round(H, 3))
    return ent

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

        if best_row:
            if best_row.get("size_mb") in (None, 0, 0.0):
                mb = _fetch_size_mb_from_ftp_stats(best_row.get("ftp") or "")
                if mb:
                    best_row["size_mb"] = mb
                    for r in ranked:
                        if r["assembly_accession"] == best_row["assembly_accession"]:
                            r["size_mb"] = mb
                            break

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
    "COI": [
        "COI",
        "cox1",
        '"cytochrome c oxidase subunit I"',
        '"cytochrome oxidase subunit I"'
    ],
    "12S": [
        "12S",
        '"12S ribosomal RNA"',
        '"12S rRNA"',
        "MT-RNR1",
        '"mitochondrial small subunit rRNA"'
    ],
    "16S": [
        "16S",
        '"16S ribosomal RNA"',
        '"16S rRNA"',
        "rrs",
        "rrsA",
        '"small subunit ribosomal RNA"'
    ],
    "ITS": [
        "ITS",
        '"internal transcribed spacer"',
        "ITS1",
        "ITS2",
        '"ribosomal DNA spacer"',
        '"rDNA ITS"'
    ],
    "18S": [
        "18S",
        '"18S ribosomal RNA"',
        '"18S rRNA"',
        '"small subunit ribosomal RNA"'
    ],
    "rbcL": [
        "rbcL",
        '"ribulose-1,5-bisphosphate carboxylase large subunit"',
        '"RuBisCO large subunit"',
        '"ribulose bisphosphate carboxylase large chain"'
    ],
    "matK": [
        "matK",
        '"maturase K"',
        '"chloroplast maturase K"',
        '"cpDNA matK"'
    ]
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

    # Length window: avoid whole genomes; keeps genuine gene/rRNA fragments
    len_clause = "80:5000[SLEN]"

    # Choose biomolecule filter by locus class
    L = (locus or "").strip().upper()
    if L in ("16S", "18S", "ITS"):
        # For rRNA/ITS, restrict to rRNA molecules; exclude obvious genome titles
        biomol_clause = "biomol_rRNA[PROP]"
        not_genome = "NOT complete genome[Title]"
        term = f"({locus_q}) AND txid{taxid}[Organism:exp] AND {biomol_clause} AND {len_clause} {not_genome}"
    else:
        # Coding loci (COI, rbcL, matK, etc.) — allow genomic/mRNA, still length-limit
        biomol_clause = "(biomol_genomic[PROP] OR biomol_mRNA[PROP])"
        term = f"({locus_q}) AND txid{taxid}[Organism:exp] AND {biomol_clause} AND {len_clause}"

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
    # Keep a wide but reasonable range; len_clause already filters most out
    seqs = [s for s in seqs if 80 <= len(s["seq"]) <= 8000]
    return seqs




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
    kwargs = dict(mv_conc=mv, dv_conc=dv, dntp_conc=dntp, dna_conc=dna_nM, temp_c=60)
    try:
        hpL = _p3_calc_hairpin(left, **kwargs);  dg_hpL = float(getattr(hpL, "dg", 0.0) or 0.0)
    except Exception:
        dg_hpL = 0.0
    try:
        hpR = _p3_calc_hairpin(right, **kwargs); dg_hpR = float(getattr(hpR, "dg", 0.0) or 0.0)
    except Exception:
        dg_hpR = 0.0
    try:
        sdL = _p3_calc_homodimer(left, **kwargs); dg_sdL = float(getattr(sdL, "dg", 0.0) or 0.0)
    except Exception:
        dg_sdL = 0.0
    try:
        sdR = _p3_calc_homodimer(right, **kwargs); dg_sdR = float(getattr(sdR, "dg", 0.0) or 0.0)
    except Exception:
        dg_sdR = 0.0
    try:
        het = _p3_calc_heterodimer(left, right, **kwargs); dg_het = float(getattr(het, "dg", 0.0) or 0.0)
    except Exception:
        dg_het = 0.0
    return {
        "dg_hp_left": dg_hpL,
        "dg_hp_right": dg_hpR,
        "dg_self_left": dg_sdL,
        "dg_self_right": dg_sdR,
        "dg_hetero": dg_het,
    }

def _dG_pass(dgs: Dict[str,float], th_hp: float, th_self: float, th_hetero: float) -> bool:
    return (
        dgs["dg_hp_left"]   >= th_hp and
        dgs["dg_hp_right"]  >= th_hp and
        dgs["dg_self_left"] >= th_self and
        dgs["dg_self_right"]>= th_self and
        dgs["dg_hetero"]    >= th_hetero
    )

# ---- Degenerate 3'-kmer uniqueness (Upgrade B) ----
def _degenerate_match_count(kmer: str, hay: str) -> int:
    """Count occurrences of IUPAC kmer in hay (no RC here)."""
    k = len(kmer)
    if k == 0: return 0
    c = 0
    for i in range(0, len(hay)-k+1):
        seg = hay[i:i+k]
        ok = True
        for a,b in zip(kmer, seg):
            if not _iupac_match(a, b):
                ok = False; break
        if ok: c += 1
    return c

def _kmer_uniqueness_3p(primer: str, seqs: List[Dict], k: int = 11) -> int:
    """Count IUPAC-compatible matches of 3'-terminal k-mer across all sequences (+ and RC)."""
    if not primer or k <= 0 or len(primer) < k:
        return 0
    kmer = primer[-k:].upper()
    rc_kmer = _revcomp(kmer)
    count = 0
    for s in seqs:
        t = s["seq"]
        rc = _revcomp(t)
        count += _degenerate_match_count(kmer, t)
        count += _degenerate_match_count(kmer, rc)
        count += _degenerate_match_count(rc_kmer, t)
        count += _degenerate_match_count(rc_kmer, rc)
    return count

# --- Region constraints from entropy (auto windows) ---
def _ok_region_pairs_from_entropy(seqs: List[Dict], f_span: int = 60, r_span: int = 60,
                                  max_entropy: float = 0.6, max_pairs: int = 4) -> Optional[str]:
    """Build SEQUENCE_PRIMER_PAIR_OK_REGION_LIST from low-entropy windows on aligned sequences (Upgrade A)."""
    if not seqs:
        return None
    ent = entropy_series(seqs)
    if not ent:
        return None
    L = len(ent)
    if L < (f_span + r_span + 10):
        return None

    ent_arr = np.array(ent, dtype=float)

    def scan_windows(start, end, span):
        best = []
        for i in range(start, max(start, end - span + 1)):
            w = ent_arr[i:i+span]
            if len(w) < span:
                break
            m = float(w.mean())
            best.append((m, i))
        best.sort(key=lambda x: (x[0], x[1]))
        return best

    f_end = int(L * 0.45)
    r_start = int(L * 0.35)
    f_cands = scan_windows(0, f_end, f_span)[:max_pairs * 2]
    r_cands = scan_windows(r_start, L - r_span, r_span)[:max_pairs * 2]

    f_cands = [c for c in f_cands if c[0] <= max_entropy] or f_cands
    r_cands = [c for c in r_cands if c[0] <= max_entropy] or r_cands

    pairs = []
    for i, (_, f_pos) in enumerate(f_cands[:max_pairs]):
        if i >= len(r_cands):
            break
        _, r_pos = r_cands[i]
        pairs.append(f"{f_pos},{f_span},{r_pos},{r_span}")

    return " ".join(pairs) if pairs else None

# ---------- Primer design ----------
def _pairwise_combine_from_lists(res: Dict, amp_min: int, amp_max: int, amplicon_opt: int) -> List[Dict]:
    out = []
    nL = int(res.get("PRIMER_LEFT_NUM_RETURNED", 0) or 0)
    nR = int(res.get("PRIMER_RIGHT_NUM_RETURNED", 0) or 0)
    lefts = []
    rights = []
    for i in range(nL):
        pos, ln = res.get(f"PRIMER_LEFT_{i}", [None, None])
        seq = res.get(f"PRIMER_LEFT_{i}_SEQUENCE")
        if seq and pos is not None:
            lefts.append({
                "idx": i, "pos": pos, "len": ln, "seq": seq,
                "tm": res.get(f"PRIMER_LEFT_{i}_TM", 0.0),
                "gc": res.get(f"PRIMER_LEFT_{i}_GC_PERCENT", 0.0)
            })
    for j in range(nR):
        pos, ln = res.get(f"PRIMER_RIGHT_{j}", [None, None])
        seq = res.get(f"PRIMER_RIGHT_{j}_SEQUENCE")
        if seq and pos is not None:
            rights.append({
                "idx": j, "pos": pos, "len": ln, "seq": seq,
                "tm": res.get(f"PRIMER_RIGHT_{j}_TM", 0.0),
                "gc": res.get(f"PRIMER_RIGHT_{j}_GC_PERCENT", 0.0)
            })
    for L in lefts:
        for R in rights:
            prod = (R["pos"] - L["pos"]) + 1
            if prod < amp_min or prod > amp_max:
                continue
            out.append({
                'left': L["seq"], 'right': R["seq"],
                'tm_left': round(L["tm"], 2), 'tm_right': round(R["tm"], 2),
                'gc_left': round(L["gc"], 1), 'gc_right': round(R["gc"], 1),
                'len_left': L["len"], 'len_right': R["len"],
                'amplicon_len': prod, 'penalty': 0.0,
                'left_pos': L["pos"], 'right_pos': R["pos"]
            })
    out.sort(key=lambda r: (abs((r['amplicon_len'] or amplicon_opt) - amplicon_opt),
                            abs(r['tm_left'] - r['tm_right'])))
    return out[:300]

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
    num_return: int = 30,
    candidate_mode: str = "pairs",
    min_three_prime_dist: int = 5,
    ok_region_pairs: Optional[str] = None
) -> List[Dict]:
    seq = sanitize_dna(seq)
    if not seq:
        return []
    target_len = max(1, min(len(seq), target_len or len(seq)))

    params = { 'SEQUENCE_ID': 'target', 'SEQUENCE_TEMPLATE': seq }
    use_target = bool(target_len) and (target_len < len(seq))
    if use_target:
        params['SEQUENCE_TARGET'] = [max(0, int(target_start)), int(target_len)]

    opts = {
        'PRIMER_TASK': 'generic' if candidate_mode == "pairs" else 'pick_primer_list',
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_MIN_SIZE': len_min,
        'PRIMER_OPT_SIZE': len_opt,
        'PRIMER_MAX_SIZE': len_max,
        'PRIMER_PRODUCT_SIZE_RANGE': [[amplicon_min, amplicon_max]],
        'PRIMER_PRODUCT_OPT_SIZE': amplicon_opt,
        'PRIMER_MIN_TM': tm_min, 'PRIMER_OPT_TM': tm_opt, 'PRIMER_MAX_TM': tm_max,
        'PRIMER_MIN_GC': gc_min, 'PRIMER_MAX_GC': gc_max,
        'PRIMER_MAX_POLY_X': 100,
        'PRIMER_MAX_SELF_ANY_TH': 45.0,
        'PRIMER_MAX_SELF_END_TH': 35.0,
        'PRIMER_MAX_HAIRPIN_TH': 47.0,
        'PRIMER_NUM_RETURN': int(num_return),
        'PRIMER_MIN_THREE_PRIME_DISTANCE': int(min_three_prime_dist),
        'PRIMER_MAX_NS_ACCEPTED': 0,
        'PRIMER_EXPLAIN_FLAG': 1,
        'PRIMER_SALT_MONOVALENT': mv_mM,
        'PRIMER_SALT_DIVALENT': dv_mM,
        'PRIMER_DNTP_CONC': dntp_mM,
        'PRIMER_DNA_CONC': dna_nM,
    }
    if ok_region_pairs:
        opts['SEQUENCE_PRIMER_PAIR_OK_REGION_LIST'] = ok_region_pairs

    res = primer3.bindings.design_primers(params, opts)

    if candidate_mode == "list":
        return _pairwise_combine_from_lists(res, amplicon_min, amplicon_max, amplicon_opt)

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

# ---------- Degenerate-aware in-silico coverage (Upgrade B) ----------
def _degenerate_primer_hits_on_strand(primer: str, template: str, max_mismatch: int, require_3prime_match: bool) -> List[int]:
    """
    Return list of start indices where primer matches template (IUPAC-aware).
    If require_3prime_match=True, enforce last base exact compatibility.
    """
    p = primer.upper(); t = template.upper()
    Lp = len(p); Lt = len(t)
    hits = []
    if Lp == 0 or Lt < Lp:
        return hits
    for i in range(0, Lt - Lp + 1):
        seg = t[i:i+Lp]
        # enforce 3' base compatibility
        if require_3prime_match and not _iupac_match(p[-1], seg[-1]):
            continue
        # count mismatches under IUPAC compatibility
        mm = 0
        ok = True
        for a,b in zip(p, seg):
            if not _iupac_match(a, b):
                mm += 1
                if mm > max_mismatch:
                    ok = False; break
        if ok:
            hits.append(i)
    return hits

def insilico_coverage(
    primer_left: str,
    primer_right: str,
    seqs: List[Dict],
    max_mismatch: int = 1,
    amp_min: int = 80,
    amp_max: int = 800,
    require_3prime_match: bool = True
):
    """
    Orientation-correct PCR check with IUPAC degeneracy (Upgrade B).
    - Forward primer matches + strand
    - Reverse primer matches rev-comp(+ strand) i.e., reverse-complemented template on + strand
    - Product size measured between outer 5' ends of primer footprints on + strand
    """
    if not seqs:
        return {"hits":0, "total":0, "pct":0.0}

    L = primer_left.upper()
    R = primer_right.upper()
    Rc = _revcomp(R)

    hits = 0
    total = 0

    for s in seqs:
        t = s["seq"].upper()
        if len(t) < max(len(L), len(R)):
            continue
        total += 1

        # All forward matches for left on + strand
        left_pos_list = _degenerate_primer_hits_on_strand(L, t, max_mismatch, require_3prime_match)

        if not left_pos_list:
            continue

        # All matches for reverse primer's reverse-complement on + strand (i.e., binding to - strand)
        right_pos_list = _degenerate_primer_hits_on_strand(Rc, t, max_mismatch, require_3prime_match)

        if not right_pos_list:
            continue

        # Decide if any pair forms a valid amplicon (left 5' -> right 5' downstream)
        L_len = len(L)
        R_len = len(Rc)
        pair_ok = False
        for iL in left_pos_list:
            left_5p = iL  # 5' of forward is its start
            # right primer binds downstream; its 5' on + strand is at (iR + R_len - 1) going left->right orientation
            for iR in right_pos_list:
                right_5p = iR + R_len - 1
                if right_5p <= left_5p:
                    continue
                prod = (right_5p - left_5p + 1)
                if amp_min <= prod <= amp_max:
                    pair_ok = True
                    break
            if pair_ok:
                break

        if pair_ok:
            hits += 1

    pct = round(100.0 * hits / max(total, 1), 2)
    return {"hits": hits, "total": total, "pct": pct}

# === Visualization helpers ===
def _encode_fig_to_b64(fig, dpi=140):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def make_entropy_plot(
    seqs, designed, locus: str,
    gc_window: int = 21,
    entropy_threshold: float = 0.60,
    max_pairs_on_plot: int = 3
):
    """
    Conservation plot with dynamic bottom margin:
      • Smoothed entropy + rolling GC%
      • Shade ONLY best amplicon
      • Up to `max_pairs_on_plot` lane rows (P1..)
      • Bottom padding scales with lanes + AA track so nothing overlaps
    """
    if not seqs:
        return None

    from matplotlib import colors
    from matplotlib.transforms import blended_transform_factory

    ent = entropy_series(seqs)
    if not ent:
        return None

    L = len(ent)
    x = np.arange(L, dtype=float)

    # ---- rolling GC% (ignore gaps) ----
    denom = np.zeros(L, dtype=float)
    gc_ct = np.zeros(L, dtype=float)
    aligned = _multiple_to_ref_alignment(seqs)
    for s in aligned:
        t = s["seq"][:L]
        for i, b in enumerate(t):
            if b in "ACGT":
                denom[i] += 1.0
                if b in "GC":
                    gc_ct[i] += 1.0
    gc_frac = np.divide(gc_ct, np.maximum(denom, 1.0))

    def smooth(y, w):
        if w <= 1: return np.asarray(y, dtype=float)
        k = np.ones(w, dtype=float) / w
        return np.convolve(np.asarray(y, dtype=float), k, mode="same")

    ent_smooth = smooth(ent, 5 if L > 300 else 3)
    gc_roll = smooth(gc_frac * 100.0, gc_window if gc_window and gc_window > 1 else 1)

    # ---- dynamic bottom margin based on lanes & AA track ----
    n_lanes = min(max_pairs_on_plot, len(designed or []))
    want_aa = (locus in CODING_LOCI and L >= 3 and bool(seqs))

    # More generous spacing so lane labels never touch ticks
    base = 0.22          # space for x-label + tick labels
    per_lane = 0.10      # per-lane space (height + gap)
    aa_extra = 0.10 if want_aa else 0.0

    bottom_margin = base + n_lanes * per_lane + aa_extra
    bottom_margin = min(0.48, bottom_margin)

    fig, ax = plt.subplots(figsize=(14.4, 4.8))
    fig.subplots_adjust(left=0.065, right=0.97, top=0.94, bottom=bottom_margin)

    ax.set_xlim(0, L)
    ax.set_ylim(0, max(1.8, float(np.max(ent)) + 0.1))
    ax.set_ylabel("Entropy", fontsize=10)
    ax.set_xlabel("Position (bp)", fontsize=10, labelpad=16)
    ax.set_xticks(np.arange(0, L + 1, 50 if L <= 1200 else 100))
    ax.grid(axis='y', alpha=0.15)

    # entropy trace
    if L <= 600:
        ax.bar(x, ent, width=1.0, color="#66c2d0", alpha=0.55, linewidth=0)
        ax.plot(x, ent_smooth, linewidth=1.2, alpha=0.9, color="#2286a7")
    else:
        ax.plot(x, ent_smooth, linewidth=1.0, alpha=0.95, color="#66c2d0")

    # threshold
    if entropy_threshold is not None:
        ax.axhline(entropy_threshold, color="#ef4444", linestyle="--", linewidth=1, alpha=0.22)
        ax.text(0.005, entropy_threshold + 0.03, f"H={entropy_threshold:.2f}",
                transform=ax.get_yaxis_transform(), fontsize=8, color="#ef4444", alpha=0.7,
                ha="left", va="bottom")

    # GC% (right axis)
    ax2 = ax.twinx()
    ax2.plot(x, gc_roll, linewidth=1.25, alpha=0.6, color="#9aa5b1")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("GC % (rolling)", fontsize=9, color="#9aa5b1")
    ax2.tick_params(axis='y', labelsize=8, colors="#9aa5b1")

    # helpers
    def clamp_x(v):
        if v is None: return 0
        if v < 0: return 0
        if v > L - 1: return L - 1
        return v

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # Show only top N in lanes; shade the best amplicon
    shown = (designed or [])[:n_lanes]
    best = designed[0] if designed else None

    if best:
        lp = int(best.get("left_pos", 0) or 0)
        ll = int(best.get("len_left", 0) or 0)
        rp = int(best.get("right_pos", 0) or 0)
        rl = int(best.get("len_right", 0) or 0)
        span_start = clamp_x(lp + ll)
        span_end = clamp_x(rp)
        if span_end <= span_start and best.get("amplicon_len"):
            span_end = clamp_x(span_start + int(best["amplicon_len"]))
        if span_end > span_start:
            ax.axvspan(span_start, span_end, color="#a9def9", alpha=0.18)
            mid = 0.5 * (span_start + span_end)
            ax.text(mid, -0.055, f"{best['amplicon_len']} bp",
                    transform=ax.get_xaxis_transform(), fontsize=8, color="#8ab4f8",
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="#0b0f12", edgecolor="none", alpha=0.45))

    # lane geometry (lower start so labels never touch ticks)
    lane_h = 0.065
    lane_gap = 0.025
    lane_top = -0.10
    tick_h = 0.040

    for i, d in enumerate(shown):
        y0 = lane_top - i * (lane_h + lane_gap)
        y1 = y0 - lane_h
        y_mid = (y0 + y1) / 2.0

        # lane label inside the axes, not in the gutter -> avoids tick collision
        ax.text(0.02, y_mid, f"P{i+1}",
                transform=ax.transAxes,
                fontsize=8, color="#cbd5e1", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#11161b", edgecolor="#2a3340", alpha=0.95))

        # tick marks at 5' ends
        lp = clamp_x(int(d.get("left_pos", 0) or 0))
        rp = clamp_x(int(d.get("right_pos", 0) or 0))
        ax.vlines(lp, y1, y1 + tick_h, transform=trans, colors="#22c55e", linewidth=2.0, clip_on=False)
        ax.vlines(rp, y1, y1 + tick_h, transform=trans, colors="#ef4444", linewidth=2.0, clip_on=False)

    # AA track (coding loci)
    if want_aa:
        aa_y = lane_top - n_lanes * (lane_h + lane_gap) - 0.06
        ref = aligned[0]["seq"][:(L // 3) * 3].replace('-', 'N')
        aa = str(Seq(ref).translate())
        codons = len(ref) // 3
        for i in range(0, codons + 1):
            bp = i * 3
            ax.axvline(bp, ymin=0, ymax=0.02, color="#94a3b833", lw=0.8)
        for i in range(0, codons, 3):
            bp = i * 3 + 1
            if bp < L:
                ax.text(bp, aa_y, aa[i], transform=ax.get_xaxis_transform(),
                        fontsize=7.5, color="#cbd5e1", ha="center", va="top")

    fig.tight_layout()
    return _encode_fig_to_b64(fig)




def make_logo_plot(seqs):
    if not HAS_LOGOMAKER or not seqs:
        return None
    # Use aligned sequences for logo
    aligned = _multiple_to_ref_alignment(seqs)
    L = min(len(s["seq"]) for s in aligned)
    if L == 0: return None
    counts = pd.DataFrame(0, index=list("ACGT"), columns=range(L))
    for s in aligned[:200]:
        for i, b in enumerate(s["seq"][:L]):
            if b in "ACGT":
                counts.at[b, i] += 1
    counts = counts.T
    fig, ax = plt.subplots(figsize=(13.5, 10.0))
    fig.subplots_adjust(bottom=0.26)
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

        # constraints & thermo
        "enforce_gc_clamp": "on",
        "min_gc3": 1, "max_gc3": 2,
        "max_run_at": 4, "max_run_gc": 4,
        "max_dinuc": 3,
        "dg_hp_min": -2.0, "dg_self_min": -6.0, "dg_hetero_min": -7.0,
        "mv_mM": 50.0, "dv_mM": 1.5, "dntp_mM": 0.2, "dna_nM": 250.0,

        # k-mer uniqueness
        "kmer_len": 11, "kmer_max_hits": 1,

        # candidate list vs pairs & region constraints
        "candidate_mode": "pairs",
        "region_mode": "none",
        "min_three_prime_dist": 5,

        "taxon": "", "locus": "COI",
        "source": "ncbi",
        "max_seqs": 100,
        "viz": "entropy",
        "designed": None, "seqs_count": 0,
        "entropy_png": None, "logo_png": None,
        "warning": None, "error": None,
        "has_logomaker": HAS_LOGOMAKER,

        "designed": None, "seqs_count": 0,
        "entropy_png": None, "logo_png": None,
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

    candidate_mode = f.get("candidate_mode") or "pairs"
    region_mode = f.get("region_mode") or "none"
    min_three_prime_dist = int(f.get("min_three_prime_dist") or 5)

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

    # ALIGN sequences for downstream conservation/viz (Upgrade A)
    aligned_for_viz = _multiple_to_ref_alignment(seqs) if seqs else []

    ref_seq = aligned_for_viz[0]["seq"].replace('-', '') if aligned_for_viz else ""
    designed = []
    entropy_png = None
    logo_png = None
    ok_region_pairs = None

    if ref_seq and region_mode == "auto":
        ok_region_pairs = _ok_region_pairs_from_entropy(aligned_for_viz, f_span=60, r_span=60,
                                                        max_entropy=0.6, max_pairs=4)

    if ref_seq:
        designed = design_primers_on_region(
            ref_seq, 0, len(ref_seq),
            amplicon_min=amp_min, amplicon_opt=amp_opt, amplicon_max=amp_max,
            tm_min=tm_min, tm_opt=tm_opt, tm_max=tm_max,
            gc_min=gc_min, gc_max=gc_max,
            len_min=len_min, len_opt=len_opt, len_max=len_max,
            mv_mM=mv_mM, dv_mM=dv_mM, dntp_mM=dntp_mM, dna_nM=dna_nM,
            num_return=200 if candidate_mode == "list" else 60,
            candidate_mode=candidate_mode,
            min_three_prime_dist=min_three_prime_dist,
            ok_region_pairs=ok_region_pairs
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

            # 3'-end uniqueness (degenerate-aware)
            d["kmer_hits_f"] = _kmer_uniqueness_3p(d["left"], seqs, k=kmer_len)
            d["kmer_hits_r"] = _kmer_uniqueness_3p(d["right"], seqs, k=kmer_len)
            d["kmer_ok"] = (d["kmer_hits_f"] <= kmer_max_hits) and (d["kmer_hits_r"] <= kmer_max_hits)

            # Coverage across sequences (degenerate-aware, orientation-correct)
            cov = insilico_coverage(d["left"], d["right"], seqs, max_mismatch=1)
            d["coverage_pct"] = cov["pct"]
            d["coverage_hits"] = cov["hits"]
            d["coverage_total"] = cov["total"]

        # Build plots using aligned sequences for entropy/logo
        entropy_png = make_entropy_plot(aligned_for_viz[:50], designed, locus=locus)
        if HAS_LOGOMAKER:
            logo_png = make_logo_plot(aligned_for_viz[:200])
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

        "candidate_mode": candidate_mode,
        "region_mode": region_mode,
        "min_three_prime_dist": min_three_prime_dist,

        "taxon": taxon, "locus": locus,
        "source": source, "max_seqs": max_seqs,
        "viz": viz,
        "designed": designed, "seqs_count": len(seqs),
        "entropy_png": entropy_png, "logo_png": logo_png,
        "warning": warning, "error": error,
        "has_logomaker": HAS_LOGOMAKER,

        "designed": designed, "seqs_count": len(seqs),
        "entropy_png": entropy_png, "logo_png": logo_png,
        "warning": warning, "error": error,
    }
    return render_template("designer.html", **ctx)

# ---- Primer BLAST helper ----
@app.post("/designer/blast")
def designer_blast():
    # Accept a primer string, run BLAST with short-query settings, return a tiny HTML table
    primer = (request.form.get("primer") or "").strip().upper()
    if not primer or len(primer) < 16:
        return ("Primer too short for BLAST (need ≥16 nt).", 400)

    try:
        # Settings that work for 18–30 nt oligos
        # task=blastn-short uses word_size=7 and scoring optimized for short exact-ish matches
        rh = NCBIWWW.qblast(
            program="blastn",
            database="nt",
            sequence=primer,
            expect=1000,               # be permissive
            word_size=7,               # redundant with task but safe
            megablast=False,           # disable megablast (bad for short queries)
            filter=False,              # don’t mask low complexity for primers
            service="plain",           # Biopython param compatibility
            # ‘task’ is passed via format_type param set string:
            entrez_query=None,
            format_type="XML",
            # NCBIWWW.qblast forwards extra keyword args via URL;
            # include task this way for broad Biopython compatibility.
            # (Older Biopython doesn't expose 'task' kw directly.)
            # Note: if your Biopython is new enough you can add task="blastn-short" directly.
        )
        # If your Biopython supports it, prefer:
        # rh = NCBIWWW.qblast("blastn", "nt", primer, task="blastn-short", expect=1000, filter=False, megablast=False)

        record = NCBIXML.read(rh)
        hits = []
        for aln in record.alignments:
            if not aln.hsps:
                continue
            hsp = aln.hsps[0]
            # %ID over the aligned region
            pct = round(100.0 * hsp.identities / max(1, hsp.align_length), 1)
            hits.append({
                "identity": pct,
                "e": f"{hsp.expect:.1e}",
                "len": aln.length,
                "title": aln.title[:140]
            })

        rows = "".join(
            f"<tr><td>{h['identity']}%</td><td>{h['e']}</td><td>{h['len']}</td><td>{h['title']}</td></tr>"
            for h in hits
        )
        if not rows:
            rows = '<tr><td colspan="4">No hits (try blastn-short on NCBI UI)</td></tr>'

        html = f"""
        <div class="table-responsive">
          <table class="table table-sm table-hover">
            <thead>
              <tr><th>%ID</th><th>E</th><th>Len</th><th>Title</th></tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        <div class="mt-2">
          <a class="btn btn-outline-secondary btn-sm" target="_blank"
             href="https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&BLAST_PROGRAMS=megaBlast&PAGE_TYPE=BlastSearch&SHOW_DEFAULTS=on&DATABASE=nt&QUERY={primer}&BLAST_PROGRAMS=blastn&JOB_TITLE=Primer%20check&TASK=blastn-short">
            Open on NCBI (blastn-short)
          </a>
        </div>
        """
        resp = make_response(html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp

    except Exception as e:
        return (f"BLAST failed: {e}", 500)


# ---- Check user-supplied primers with Primer3 thermo ----
@app.post("/designer/check")
def designer_check():
    left = sanitize_dna(request.form.get("left") or "")
    right = sanitize_dna(request.form.get("right") or "")
    template = sanitize_dna(request.form.get("template") or "")
    if not left or not right:
        return ("Need left and right primers.", 400)

    mv_mM = float(request.form.get("mv_mM") or 50.0)
    dv_mM = float(request.form.get("dv_mM") or 1.5)
    dntp_mM = float(request.form.get("dntp_mM") or 0.2)
    dna_nM = float(request.form.get("dna_nM") or 250.0)

    params = {'SEQUENCE_ID': 'user', 'PRIMER_LEFT_INPUT': left, 'PRIMER_RIGHT_INPUT': right}
    if template:
        params['SEQUENCE_TEMPLATE'] = template

    opts = {
        'PRIMER_TASK': 'check_primers',
        'PRIMER_EXPLAIN_FLAG': 1,
        'PRIMER_SALT_MONOVALENT': mv_mM,
        'PRIMER_SALT_DIVALENT': dv_mM,
        'PRIMER_DNTP_CONC': dntp_mM,
        'PRIMER_DNA_CONC': dna_nM,
    }

    try:
        res = primer3.bindings.design_primers(params, opts)
        tmL = round(res.get('PRIMER_LEFT_0_TM', 0.0), 2)
        tmR = round(res.get('PRIMER_RIGHT_0_TM', 0.0), 2)
        anyL = round(res.get('PRIMER_LEFT_0_SELF_ANY_TH', 0.0), 2)
        endL = round(res.get('PRIMER_LEFT_0_SELF_END_TH', 0.0), 2)
        anyR = round(res.get('PRIMER_RIGHT_0_SELF_ANY_TH', 0.0), 2)
        endR = round(res.get('PRIMER_RIGHT_0_SELF_END_TH', 0.0), 2)
        het  = round(res.get('PRIMER_PAIR_0_COMPL_ANY_TH', 0.0), 2)
        hetE = round(res.get('PRIMER_PAIR_0_COMPL_END_TH', 0.0), 2)

        html = f"""
        <div class='p-2 text-sm'>
          <div><b>Tm</b> — Left: {tmL}°C, Right: {tmR}°C</div>
          <div><b>Self-dimer (ANY/END)</b> — Left: {anyL}/{endL}, Right: {anyR}/{endR}</div>
          <div><b>Hetero-dimer (ANY/END)</b> — Pair: {het}/{hetE}</div>
        </div>
        """
        resp = make_response(html)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        return resp
    except Exception as e:
        return (f"Check failed: {e}", 500)

# ---- Settings export/import (reproducibility) ----
def _collect_primer3_settings(form) -> Dict[str, str]:
    keys = {
        "PRIMER_MIN_SIZE": "len_min",
        "PRIMER_OPT_SIZE": "len_opt",
        "PRIMER_MAX_SIZE": "len_max",
        "PRIMER_PRODUCT_OPT_SIZE": "amp_opt",
        "PRIMER_MIN_TM": "tm_min",
        "PRIMER_OPT_TM": "tm_opt",
        "PRIMER_MAX_TM": "tm_max",
        "PRIMER_MIN_GC": "gc_min",
        "PRIMER_MAX_GC": "gc_max",
        "PRIMER_SALT_MONOVALENT": "mv_mM",
        "PRIMER_SALT_DIVALENT": "dv_mM",
        "PRIMER_DNTP_CONC": "dntp_mM",
        "PRIMER_DNA_CONC": "dna_nM",
        "PRIMER_MIN_THREE_PRIME_DISTANCE": "min_three_prime_dist",
        "PRIMER_MAX_NS_ACCEPTED": None,
    }
    out = {}
    for k, fkey in keys.items():
        if fkey and (fkey in form):
            out[k] = str(form.get(fkey))
    out["PRIMER_MAX_NS_ACCEPTED"] = "0"
    return out

@app.post("/designer/settings/export")
def settings_export():
    s = _collect_primer3_settings(request.form)
    lines = [f"{k}={v}" for k,v in s.items()]
    payload = "\n".join(lines).encode("utf-8")
    resp = make_response(payload)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Content-Disposition"] = 'attachment; filename="primer3_settings.prm"'
    return resp

@app.post("/designer/settings/import")
def settings_import():
    file = request.files.get("settings_file")
    if not file or not file.filename:
        return ("No settings file uploaded.", 400)
    try:
        text = file.read().decode("utf-8", errors="ignore")
        cfg = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
        resp = make_response(json.dumps(cfg))
        resp.headers["Content-Type"] = "application/json"
        return resp
    except Exception as e:
        return (f"Failed to parse settings: {e}", 400)

# ---- Designer CSV export ----
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
@app.get("/healthz")
def healthz():
    return "ok", 200

# === Local debug runner ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# === Injected API routes for Vercel proxy ===

# ---------- Health ----------
@app.get("/health")
def health():
    return jsonify(ok=True)

# ---------- Primer design API ----------
@app.post("/api/primers")
def api_primers():
    """
    JSON:
    {
      "sequence": "ACGT...",
      "params": {...}   # optional primer3 global params
    }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        seq = (data.get("sequence") or "").strip().upper()
        if not seq or set(seq) - set("ACGTN"):
            return jsonify(ok=False, error="Provide a DNA sequence (ACGTN) in 'sequence'"), 400

        global_args = {
            'PRIMER_OPT_SIZE': 20,
            'PRIMER_MIN_SIZE': 18,
            'PRIMER_MAX_SIZE': 25,
            'PRIMER_OPT_TM': 60.0,
            'PRIMER_MIN_TM': 57.0,
            'PRIMER_MAX_TM': 63.0,
            'PRIMER_MAX_POLY_X': 4,
            'PRIMER_GC_CLAMP': 1,
            'PRIMER_NUM_RETURN': 5,
        }
        user_params = data.get("params") or {}
        for k,v in user_params.items():
            global_args[str(k)] = v

        seq_args = { 'SEQUENCE_ID': 'template', 'SEQUENCE_TEMPLATE': seq }

        primers = []
        try:
            res = primer3.bindings.designPrimers(seq_args, global_args)
            n = int(res.get('PRIMER_PAIR_NUM_RETURNED', 0) or 0)
            for i in range(n):
                primers.append({
                    "left_seq": res.get(f"PRIMER_LEFT_{i}_SEQUENCE"),
                    "right_seq": res.get(f"PRIMER_RIGHT_{i}_SEQUENCE"),
                    "left_tm": res.get(f"PRIMER_LEFT_{i}_TM"),
                    "right_tm": res.get(f"PRIMER_RIGHT_{i}_TM"),
                    "product_size": res.get(f"PRIMER_PAIR_{i}_PRODUCT_SIZE"),
                    "pair_penalty": res.get(f"PRIMER_PAIR_{i}_PENALTY"),
                })
            payload = {
                "ok": True,
                "count": len(primers),
                "primers": primers,
                "warnings": res.get("PRIMER_WARNING", ""),
            }
        except Exception as e:
            payload = {"ok": False, "error": f"primer3 error: {e}"}
        return jsonify(payload)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

# ---------- BLAST/NCBI check (stub) ----------
@app.post("/api/blast")
def api_blast():
    """
    Accepts: { "sequence": "ACGT..." }
    NOTE: NCBI qblast calls are long-running and often blocked from serverless.
    This endpoint is a stub; wire your existing BLAST routine here.
    """
    try:
        data = request.get_json(force=True) or {}
        seq = (data.get("sequence") or "").strip().upper()
        if not seq:
            return jsonify(ok=False, error="Missing 'sequence'"), 400
        return jsonify(ok=True, hits=[])
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500

# ---------- PDF report ----------
@app.post("/api/report")
def api_report():
    """
    Accepts: { "title": "Marker Report", "items": [ {...primer pair...} ] }
    Returns: application/pdf
    """
    try:
        data = request.get_json(force=True) or {}
        title = data.get("title") or "Marker Finder Report"
        items = data.get("items") or []

        styles = getSampleStyleSheet()
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=LETTER, title=title, author="Marker Finder")
        story = []
        story.append(Paragraph(title, styles['Title']))
        story.append(Spacer(1, 12))
        if items:
            rows = [["#", "Left", "Right", "Tm L", "Tm R", "Product"]]
            for i, it in enumerate(items, 1):
                rows.append([
                    str(i),
                    it.get("left_seq",""),
                    it.get("right_seq",""),
                    f"{it.get('left_tm','')}"[:6],
                    f"{it.get('right_tm','')}"[:6],
                    str(it.get("product_size","")),
                ])
            tbl = Table(rows, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                ("GRID",(0,0),(-1,-1), 0.5, colors.grey),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("ALIGN",(0,0),(0,-1),"RIGHT"),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("No primer pairs provided.", styles['Normal']))
        doc.build(story)
        buf.seek(0)
        return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name="marker_report.pdf")
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500
