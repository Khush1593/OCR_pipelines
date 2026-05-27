import os
import cv2
import numpy as np
import re
from paddleocr import PaddleOCR
from pdf2image import convert_from_path


def _poly_to_bbox(poly):
    """Convert a polygon (list of points) to an axis-aligned bbox.

    Returns (x_min, y_min, x_max, y_max). If poly is invalid, returns None.
    """
    if not poly:
        return None

    pts = np.array(poly, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None

    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def _extract_ocr_items(ocr_output):
    """Extract OCR items from PaddleOCR output.

    Supports:
    - `OCRResult` object from `ocr.predict(...)` (has `.json`)
    - dict with shape like: {"res": {"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}}
    - dict already at the "res" level

    Returns list of dicts: {"text": str, "bbox": (x0,y0,x1,y1), "score": float|None}
    """
    if ocr_output is None:
        return []

    if hasattr(ocr_output, "json"):
        ocr_output = getattr(ocr_output, "json")

    if isinstance(ocr_output, dict) and "res" in ocr_output and isinstance(ocr_output["res"], dict):
        data = ocr_output["res"]
    elif isinstance(ocr_output, dict):
        data = ocr_output
    else:
        return []

    polygons = data.get("dt_polys") or []
    texts = data.get("rec_texts") or []
    scores = data.get("rec_scores") or [None] * len(texts)

    items = []
    for i in range(min(len(polygons), len(texts))):
        text = texts[i]
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue

        bbox = _poly_to_bbox(polygons[i])
        if bbox is None:
            continue

        score = None
        if i < len(scores) and scores[i] is not None:
            try:
                score = float(scores[i])
            except Exception:
                score = None

        items.append({"text": text, "bbox": bbox, "score": score})

    return items


def ocr_items_to_lines(items, *, line_y_tol=None):
    """Convert OCR items (text + bbox) into ordered text lines.

    - Sorts items in a top-to-bottom, left-to-right reading order.
    - Groups items into lines using a y-center tolerance.

    Returns list[str].
    """
    if not items:
        return []

    enriched = []
    heights = []
    for item in items:
        x0, y0, x1, y1 = item["bbox"]
        yc = (y0 + y1) / 2.0
        h = max(1.0, (y1 - y0))
        heights.append(h)
        enriched.append((yc, x0, h, item["text"]))

    median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 12.0
    if line_y_tol is None:
        line_y_tol = max(6.0, 0.60 * median_h)

    enriched.sort(key=lambda t: (t[0], t[1]))

    lines = []
    current = []
    current_y = None

    for yc, x0, h, text in enriched:
        if current_y is None:
            current_y = yc
            current = [(x0, text)]
            continue

        if abs(yc - current_y) <= line_y_tol:
            current.append((x0, text))
            current_y = (current_y + yc) / 2.0
        else:
            current.sort(key=lambda t: t[0])
            lines.append(" ".join(t for _, t in current).strip())
            current_y = yc
            current = [(x0, text)]

    if current:
        current.sort(key=lambda t: t[0])
        lines.append(" ".join(t for _, t in current).strip())

    return [ln for ln in lines if ln]


def ocr_items_to_line_objects(items, *, line_y_tol=None):
    """Like `ocr_items_to_lines`, but returns structured line objects.

    Returns list of dicts:
    {"text": str, "bbox": (x0,y0,x1,y1), "score": float|None, "words": [item,...]}
    """
    if not items:
        return []

    enriched = []
    heights = []
    for item in items:
        x0, y0, x1, y1 = item["bbox"]
        yc = (y0 + y1) / 2.0
        h = max(1.0, (y1 - y0))
        heights.append(h)
        enriched.append((yc, x0, h, item))

    median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 12.0
    if line_y_tol is None:
        line_y_tol = max(6.0, 0.60 * median_h)

    enriched.sort(key=lambda t: (t[0], t[1]))

    line_groups = []
    current = []
    current_y = None
    for yc, x0, h, item in enriched:
        if current_y is None:
            current_y = yc
            current = [item]
            continue

        if abs(yc - current_y) <= line_y_tol:
            current.append(item)
            current_y = (current_y + yc) / 2.0
        else:
            line_groups.append(current)
            current_y = yc
            current = [item]

    if current:
        line_groups.append(current)

    lines = []
    for group in line_groups:
        group_sorted = sorted(group, key=lambda it: it["bbox"][0])
        text = " ".join(it["text"] for it in group_sorted).strip()
        if not text:
            continue

        xs0 = [it["bbox"][0] for it in group_sorted]
        ys0 = [it["bbox"][1] for it in group_sorted]
        xs1 = [it["bbox"][2] for it in group_sorted]
        ys1 = [it["bbox"][3] for it in group_sorted]
        bbox = (float(min(xs0)), float(min(ys0)), float(max(xs1)), float(max(ys1)))

        sc = [it["score"] for it in group_sorted if it.get("score") is not None]
        score = float(np.mean(np.array(sc, dtype=np.float32))) if sc else None

        lines.append({"text": text, "bbox": bbox, "score": score, "words": group_sorted})

    lines.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    return lines


def _estimate_page_width_from_items(items):
    if not items:
        return None
    x_max = max(it["bbox"][2] for it in items)
    x_min = min(it["bbox"][0] for it in items)
    w = float(x_max - x_min)
    return w if w > 0 else None


def split_lines_into_columns(lines, *, page_width=None):
    """Heuristic 1-2 column splitter based on x-position gaps.

    Returns list of columns, each column is a list of line objects.
    """
    if not lines:
        return []
    if page_width is None or page_width <= 0:
        return [sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))]

    x0s = sorted([ln["bbox"][0] for ln in lines])
    if len(x0s) < 8:
        return [sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))]

    gaps = [(x0s[i + 1] - x0s[i], i) for i in range(len(x0s) - 1)]
    max_gap, max_i = max(gaps, key=lambda t: t[0])

    # Split only if there is a clearly large gap between left and right starts.
    if max_gap < 0.25 * page_width:
        return [sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))]

    threshold = (x0s[max_i] + x0s[max_i + 1]) / 2.0
    left = [ln for ln in lines if ln["bbox"][0] < threshold]
    right = [ln for ln in lines if ln["bbox"][0] >= threshold]

    if len(left) < 3 or len(right) < 3:
        return [sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))]

    left.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    right.sort(key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))
    return [left, right]


def extract_key_value_pairs(lines):
    """Extract simple `Key: Value` pairs from lines."""
    pairs = []
    for ln in lines:
        text = ln["text"]
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if len(key) > 40:
            continue
        pairs.append((key, value))
    # De-dupe while preserving order
    seen = set()
    out = []
    for k, v in pairs:
        sig = (k.lower(), v.lower())
        if sig in seen:
            continue
        seen.add(sig)
        out.append((k, v))
    return out


def extract_contacts(lines):
    """Extract basic contact entities from OCR lines (no coordinates)."""
    text_blob = "\n".join(ln["text"] for ln in lines if ln.get("text"))

    emails = sorted(set(re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text_blob, flags=re.I)))
    urls = sorted(set(re.findall(r"\bhttps?://[^\s)\]]+|\bwww\.[^\s)\]]+", text_blob, flags=re.I)))

    # Loose phone matcher: sequences of digits with optional +, spaces, -, ()
    raw_phones = re.findall(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3}[\s-]?\d{3,4}", text_blob)
    phones = []
    for p in raw_phones:
        digits = re.sub(r"\D", "", p)
        if 9 <= len(digits) <= 15:
            phones.append(p.strip())
    # De-dupe (normalized)
    seen = set()
    phones_out = []
    for p in phones:
        sig = re.sub(r"\D", "", p)
        if sig in seen:
            continue
        seen.add(sig)
        phones_out.append(p)

    return {"emails": emails, "phones": phones_out, "urls": urls}


def _is_heading_line(text):
    """Heuristic: treat short, non-numeric, non-bulleted lines as headings."""
    if not text:
        return False
    t = text.strip()
    if len(t) > 40:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    if ":" in t or "@" in t:
        return False
    if t.startswith(("•", "-", "*")):
        return False
    if "." in t or "," in t:
        return False
    words = [w for w in re.split(r"\s+", t) if w]
    if len(words) > 4:
        return False
    # letters/spaces only-ish
    letters = sum(ch.isalpha() for ch in t)
    if letters < max(3, int(0.6 * len(t))):
        return False
    return t[0].isalpha() and t[0].isupper()


def _normalize_bullet_text(text):
    if not text:
        return text
    t = text.strip()
    t = re.sub(r"^[\u2022\u2023\u25E6\u2043\u2219•\-\*]+\s*", "", t)
    return t


def write_ocr_structured_markdown(ocr_output, output_path, *, source=None, page=None):
    """Write OCR output as structured Markdown (no coordinates)."""
    items = _extract_ocr_items(ocr_output)
    line_objects = ocr_items_to_line_objects(items)

    page_width = _estimate_page_width_from_items(items)
    columns = split_lines_into_columns(line_objects, page_width=page_width)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_lines = [ln for col in columns for ln in col]
    kv = extract_key_value_pairs(all_lines)
    contacts = extract_contacts(all_lines)

    md = ["# OCR Extract (Structured)"]
    if source:
        md.append(f"Source: {source}")
    if page is not None:
        md.append(f"Page: {page}")
    md.append(f"Total lines: {len(all_lines)}")
    md.append("")

    if contacts["emails"] or contacts["phones"] or contacts["urls"]:
        md.append("## Contacts")
        for e in contacts["emails"]:
            md.append(f"- Email: {e}")
        for p in contacts["phones"]:
            md.append(f"- Phone: {p}")
        for u in contacts["urls"]:
            md.append(f"- Link: {u}")
        md.append("")

    if kv:
        md.append("## Key-Value")
        for k, v in kv[:80]:
            md.append(f"- {k}: {v}")
        md.append("")

    # Render columns with light paragraph breaks
    def render_column(title, col_lines):
        if not col_lines:
            return
        md.append(f"## {title}")

        heights = [max(1.0, (ln["bbox"][3] - ln["bbox"][1])) for ln in col_lines]
        median_h = float(np.median(np.array(heights, dtype=np.float32))) if heights else 12.0
        para_gap = 1.50 * median_h

        prev_y1 = None
        for ln in col_lines:
            x0, y0, x1, y1 = ln["bbox"]
            if prev_y1 is not None and (y0 - prev_y1) > para_gap:
                md.append("")

            text = ln["text"]
            if _is_heading_line(text):
                md.append(f"### {text.strip()}")
            else:
                md.append(f"- {_normalize_bullet_text(text)}")
            prev_y1 = y1
        md.append("")
        
    # Handle empty OCR result safely
    if not columns:
        md.append("## Text")
        md.append("_No text detected._")
        md.append("")

    elif len(columns) == 1:
        render_column("Text", columns[0])

    else:
        render_column("Column 1", columns[0])

        if len(columns) > 1:
            render_column("Column 2", columns[1])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md).rstrip() + "\n")

    return output_path


def write_ocr_human_readable(ocr_output, output_path, *, source=None, page=None):
    """Write a human-readable OCR output file (no coordinates).

    The output is easy for humans to read and still easy to process later:
    - One OCR line per line (after a small header in .md)
    - Lines are in reading order

    `output_path` extension controls format:
    - .txt => plain lines
    - .md  => markdown with a short header
    """
    items = _extract_ocr_items(ocr_output)
    lines = ocr_items_to_lines(items)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".md":
        header = ["# OCR Text"]
        if source:
            header.append(f"Source: {source}")
        if page is not None:
            header.append(f"Page: {page}")
        header.append("")
        content = "\n".join(header + lines) + "\n"
    else:
        content = "\n".join(lines) + ("\n" if lines else "")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path

# 🔴 Fix Paddle crash
os.environ["FLAGS_use_mkldnn"] = "0"


def draw_boxes(image, polygons, texts, output_path):
    for poly, text in zip(polygons, texts):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(image, [pts], True, (0, 255, 0), 2)

        top_idx = np.argmin(pts[:, :, 1])
        top_left = pts[top_idx][0]

        cv2.putText(image, text,
                    (top_left[0], max(0, top_left[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1)

    cv2.imwrite(output_path, image)
    print(f"✅ Saved: {output_path}")


def process_image(image_path, ocr):
    print(f"\nProcessing Image: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return

    result = ocr.predict(image)[0]
    res_dict = result.json

    data = res_dict.get("res", {})
    polygons = data.get("dt_polys", [])
    texts = data.get("rec_texts", [])

    print(f"Detected boxes: {len(polygons)}")

    draw_boxes(image, polygons, texts, "output_image.jpg")
    write_ocr_human_readable(res_dict, "output_image.txt", source=image_path)
    print("✅ Saved: output_image.txt")
    write_ocr_structured_markdown(res_dict, "output_image.md", source=image_path)
    print("✅ Saved: output_image.md")


def process_pdf(pdf_path, ocr):
    print(f"\nProcessing PDF: {pdf_path}")

    page_num = 1

    while True:
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=120,              # 🔴 SAFE DPI
                first_page=page_num,
                last_page=page_num
            )
        except Exception as e:
            print(f"❌ PDF read error: {e}")
            break

        if not pages:
            break

        print(f"\nProcessing Page {page_num}")

        image = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)

        result = ocr.predict(image)[0]
        res_dict = result.json

        data = res_dict.get("res", {})
        polygons = data.get("dt_polys", [])
        texts = data.get("rec_texts", [])

        print(f"Detected boxes: {len(polygons)}")

        draw_boxes(image, polygons, texts,
                   f"output_page_{page_num}.jpg")

        write_ocr_human_readable(
            res_dict,
            f"output_page_{page_num}.txt",
            source=pdf_path,
            page=page_num,
        )
        print(f"✅ Saved: output_page_{page_num}.txt")

        write_ocr_structured_markdown(
            res_dict,
            f"output_page_{page_num}.md",
            source=pdf_path,
            page=page_num,
        )
        print(f"✅ Saved: output_page_{page_num}.md")

        # cleanup
        del image
        del pages

        page_num += 1


def process_input(input_path):
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        enable_mkldnn=False
    )

    ext = input_path.lower()

    if ext.endswith(".pdf"):
        process_pdf(input_path, ocr)
    else:
        process_image(input_path, ocr)


# -----------------------
# ✅ RUN HERE
# -----------------------

if __name__ == "__main__":
    input_file = "1777069512875.pdf"  # or ing or doc
    process_input(input_file)