import os
import cv2
import numpy as np
import re
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output


# =========================================================
# CONFIG
# =========================================================

# OPTIONAL:
# Set this only if tesseract is not detected automatically.
# Example Linux:
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Example Windows:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================================================
# HELPERS
# =========================================================

def _poly_to_bbox(poly):
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


def ocr_items_to_lines(items, *, line_y_tol=None):
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

    median_h = float(np.median(np.array(heights, dtype=np.float32)))

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


def write_ocr_human_readable(items, output_path, *, source=None, page=None):

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
        content = "\n".join(lines) + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


# =========================================================
# TESSERACT OCR
# =========================================================

def extract_tesseract_items(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        rgb,
        output_type=Output.DICT,
        config="--oem 3 --psm 6"
    )

    items = []

    n = len(data["text"])

    for i in range(n):

        text = data["text"][i].strip()

        if not text:
            continue

        conf = float(data["conf"][i])

        if conf < 30:
            continue

        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]

        bbox = (
            float(x),
            float(y),
            float(x + w),
            float(y + h)
        )

        items.append({
            "text": text,
            "bbox": bbox,
            "score": conf
        })

    return items


# =========================================================
# DRAW BOXES
# =========================================================

def draw_boxes(image, items, output_path):

    for item in items:

        x0, y0, x1, y1 = item["bbox"]

        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)

        text = item["text"]

        cv2.rectangle(
            image,
            (x0, y0),
            (x1, y1),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            text,
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1
        )

    cv2.imwrite(output_path, image)

    print(f"✅ Saved: {output_path}")


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(image_path):

    print(f"\nProcessing Image: {image_path}")

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Failed to load image")
        return

    items = extract_tesseract_items(image)

    print(f"Detected boxes: {len(items)}")

    draw_boxes(
        image,
        items,
        "output_image.jpg"
    )

    write_ocr_human_readable(
        items,
        "output_image.txt",
        source=image_path
    )

    print("✅ Saved: output_image.txt")


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(pdf_path):

    print(f"\nProcessing PDF: {pdf_path}")

    page_num = 1

    while True:

        try:
            pages = convert_from_path(
                pdf_path,
                dpi=150,
                first_page=page_num,
                last_page=page_num
            )

        except Exception as e:
            print(f"❌ PDF read error: {e}")
            break

        if not pages:
            break

        print(f"\nProcessing Page {page_num}")

        image = cv2.cvtColor(
            np.array(pages[0]),
            cv2.COLOR_RGB2BGR
        )

        items = extract_tesseract_items(image)

        print(f"Detected boxes: {len(items)}")

        draw_boxes(
            image,
            items,
            f"output_page_{page_num}.jpg"
        )

        write_ocr_human_readable(
            items,
            f"output_page_{page_num}.txt",
            source=pdf_path,
            page=page_num
        )

        print(f"✅ Saved: output_page_{page_num}.txt")

        del image
        del pages

        page_num += 1


# =========================================================
# MAIN
# =========================================================

def process_input(input_path):

    ext = input_path.lower()

    if ext.endswith(".pdf"):
        process_pdf(input_path)

    else:
        process_image(input_path)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    input_file = "1777069512875.pdf"  # or ing or doc
    
    process_input(input_file)