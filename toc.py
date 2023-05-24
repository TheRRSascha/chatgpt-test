from typing import Dict
import os
import fitz  # pip install pymupdf


def get_bookmarks(filepath: str) -> Dict[int, str]:
    # WARNING! One page can have multiple bookmarks!
    bookmarks_list = {}
    with fitz.open(filepath) as doc:
        toc = doc.get_toc()  # [[lvl, title, page, …], …]
        for level, title, page in toc:
            bookmarks_list[page] = title
    return bookmarks_list


pdf_folder = "C:\\PDFs Testfiles"
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    bookmarks = get_bookmarks(pdf_path)
    print(f"Bookmarks for {pdf_file}: {bookmarks}")
