"""Build an annotated PDF comparison from the preserved v1 and clean v2 PDFs.

Run from the repository root. Requires PyMuPDF; never edits either input PDF.
"""
import difflib
import hashlib
import json
from pathlib import Path
import unicodedata

import fitz


def inventory(doc):
    records = []
    for page_index, page in enumerate(doc):
        for word in page.get_text("words", sort=True):
            # Page furniture is not an author-content revision.
            if word[1] < 45 and page_index > 0:
                continue
            if word[3] > page.rect.height - 35:
                continue
            token = unicodedata.normalize("NFKC", word[4])
            records.append((page_index, word, token))
    return records


def mark(doc, records, selected, addition):
    spans = []
    current = None
    previous_index = None
    for index in sorted(selected):
        page_index, word, _ = records[index]
        key = (page_index, word[5], word[6])
        rect = fitz.Rect(word[:4])
        if current and key == current[0] and index == previous_index + 1 and rect.x0 - current[1].x1 < 18:
            current[1] |= rect
        else:
            current = [key, rect]
            spans.append(current)
        previous_index = index
    for (page_index, _, _), rect in spans:
        page = doc[page_index]
        annot = page.add_underline_annot(rect) if addition else page.add_strikeout_annot(rect)
        annot.set_colors(stroke=(0.05, 0.25, 0.85) if addition else (0.8, 0.05, 0.05))
        annot.set_info(title="Added or replaced text" if addition else "Removed or replaced text")
        annot.update()
    return len(spans)


def cover(doc):
    page = doc.new_page(width=595.28, height=841.89)
    page.insert_text((54, 76), "Manuscript revision comparison", fontsize=21, fontname="hebo")
    page.insert_textbox(fitz.Rect(54, 100, 541, 170),
        "Bangla Sign Language Gesture Recognition\nNazmul Hassan",
        fontsize=13, fontname="helv", lineheight=1.5)
    blocks = [
        ("Comparison", "Preserved ScienceOpen version 1 compared with the clean revised manuscript. This file is for the tracked-changes upload field; the clean revised manuscript belongs in the main content field."),
        ("How to read the markup", "Part A contains the revised manuscript. Blue underlining marks added or replaced words. Part B contains the original manuscript. Red strikethrough marks removed or replaced words. Unmarked text matched between versions. Page numbers within each part belong to that source manuscript."),
        ("Scope of the comparison", "The revision substantially rewrites and extends the original text, so extensive markup is expected. This is an automated comparison of text extracted from the two PDFs, not a native word-processor change log. Reordered text, line-end hyphenation and extraction differences can appear as removal plus addition. Figures and tables remain visible in their respective versions; graphical content is not compared pixel by pixel."),
        ("Document integrity", "The source PDFs are unchanged. The comparison preserves the original claims for inspection and does not treat them as results of the additional experiments. A separate revision summary is supplied in the ScienceOpen submission guide."),
        ("Contents", "Part A: revised manuscript, 10 pages\nPart B: original ScienceOpen preprint, 8 pages\nOriginal DOI: 10.14293/S2199-1006.1.SOR-.PPUF56Q.v1"),
    ]
    y = 185
    for title, body in blocks:
        page.insert_text((54, y), title, fontsize=12, fontname="hebo")
        area = fitz.Rect(54, y + 13, 541, y + 142)
        remaining = page.insert_textbox(area, body, fontsize=10.5, fontname="helv", lineheight=1.3)
        assert remaining >= 0, title
        used = area.height - remaining
        y += used + 42
    assert y < 800, y
    page.draw_line((54, 157), (175, 157), color=(0.05, .25, .85), width=1.2)
    page.insert_text((181, 160), "Added / replaced", fontsize=10, fontname="helv")
    page.insert_text((342, 160), "Removed / replaced", fontsize=10, fontname="helv")
    page.draw_line((339, 156), (443, 156), color=(.8, .05, .05), width=1.2)


def main(root):
    paths = [root / "paper/v1/preprint.pdf", root / "paper/v2/main.pdf"]
    digests = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    old, new = [fitz.open(p) for p in paths]
    old_words, new_words = inventory(old), inventory(new)
    matcher = difflib.SequenceMatcher(None, [r[2] for r in old_words], [r[2] for r in new_words], autojunk=False)
    deleted, added = set(), set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"}: deleted.update(range(i1, i2))
        if tag in {"insert", "replace"}: added.update(range(j1, j2))
    old_spans = mark(old, old_words, deleted, False)
    new_spans = mark(new, new_words, added, True)
    out = fitz.open()
    cover(out)
    out.insert_pdf(new, annots=True, links=True)
    out.insert_pdf(old, annots=True, links=True)
    out.set_toc([[1, "Comparison guide", 1], [1, "Part A - revised manuscript with additions", 2], [1, "Part B - original manuscript with removals", 2 + len(new)]])
    out.set_metadata({"title": "Bangla Sign Language Gesture Recognition - tracked changes", "author": "Nazmul Hassan", "subject": "Text comparison of preserved v1 and revised manuscript"})
    assert sum(1 for p in out for _ in p.annots()) == old_spans + new_spans
    # Embed markup in page content so upload previews cannot hide it.
    out.bake(annots=True, widgets=False)
    target = root / "paper/v2/tracked-changes.pdf"
    out.save(target, garbage=4, deflate=True)
    assert target.stat().st_size < 10_000_000
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == digests
    check = fitz.open(target)
    annotations = sum(1 for p in check for _ in p.annots())
    assert len(check) == 1 + len(old) + len(new)
    assert annotations == 0
    record = {"original_sha256": digests[0], "revised_sha256": digests[1],
              "comparison_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "pages": len(check), "bytes": target.stat().st_size,
              "added_words": len(added), "deleted_words": len(deleted),
              "addition_annotations": new_spans, "deletion_annotations": old_spans,
              "markup_embedded_in_page_content": True,
              "method": "PDF-extracted word comparison, NFKC normalization, SequenceMatcher; graphics retained without pixel diff"}
    (root / "paper/v2/tracked-changes.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
