from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path(__file__).parent / "documents"


def load_pdf(pdf_path: Path):
    """Extract text from every page of a PDF."""

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_number
            })

    return pages


def create_chunks():
    """Load all PDFs and split them into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    all_chunks = []

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    for pdf_path in pdf_files:

        pages = load_pdf(pdf_path)

        for page_data in pages:

            chunks = splitter.split_text(page_data["text"])

            for chunk_index, chunk in enumerate(chunks):

                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_data["page"],
                        "chunk_id": chunk_index
                    }
                })

    return all_chunks


if __name__ == "__main__":

    chunks = create_chunks()

    print(f"Total chunks created: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks[:5], start=1):

        print("=" * 70)
        print(f"Chunk {i}")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Chunk ID: {chunk['metadata']['chunk_id']}")
        print("-" * 70)
        print(chunk["text"][:500])
        print()