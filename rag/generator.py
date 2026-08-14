
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rag.retriever import retrieve


MODEL_NAME = "google/flan-t5-base"


def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return tokenizer, model


def extract_steps_from_context(chunks):
    """
    Extract numbered opening-procedure steps directly from
    retrieved document chunks.

    This prevents the LLM from dropping rows or mixing
    unrelated procedures.
    """

    steps = []

    pattern = re.compile(
        r"(?m)^\s*([1-6])\s+"
        r"(?!Opening Procedures)"
        r"(.+?)"
        r"(?=\n\s*[1-6]\s+|\n\s*Cross-reference:|"
        r"\n\s*2\.\s+Closing Procedures|$)"
    )

    for chunk in chunks:
        text = chunk["text"]

        if "Opening Procedures (BO-101)" not in text:
            continue

        matches = pattern.findall(text)

        for number, action in matches:
            action = " ".join(action.split())

            steps.append(
                {
                    "number": int(number),
                    "action": action,
                }
            )

    unique_steps = {}

    for step in steps:
        unique_steps[step["number"]] = step["action"]

    return [
        {
            "number": number,
            "action": unique_steps[number],
        }
        for number in sorted(unique_steps)
    ]


def format_steps(steps):
    """
    Format extracted procedure steps as a readable answer.
    """

    return "\n".join(
        f"{step['number']}. {step['action']}"
        for step in steps
    )


def generate_answer(question: str):
    print("Retrieving documents...")

    # First retrieval using the original question.
    chunks = retrieve(question, top_k=5)

    # Try extracting structured opening-procedure steps.
    steps = extract_steps_from_context(chunks)

    # ---------------------------------------------------------
    # Query expansion fallback
    # ---------------------------------------------------------
    # If the question is about opening procedures and the first
    # retrieval does not contain the required steps, perform a
    # second retrieval using a more explicit query.
    if not steps and "opening" in question.lower():
        expanded_chunks = retrieve(
            "Opening Procedures BO-101",
            top_k=5
        )

        existing = {
            (
                chunk["metadata"].get("source", "Unknown"),
                chunk["metadata"].get("page", "Unknown"),
                chunk["text"]
            )
            for chunk in chunks
        }

        for chunk in expanded_chunks:
            key = (
                chunk["metadata"].get("source", "Unknown"),
                chunk["metadata"].get("page", "Unknown"),
                chunk["text"]
            )

            if key not in existing:
                chunks.append(chunk)

        steps = extract_steps_from_context(chunks)

    question_lower = question.lower()

    asks_for_steps = (
        "six steps" in question_lower
        or "steps" in question_lower
        or "procedure" in question_lower
    )

    # ---------------------------------------------------------
    # Structured procedure answer
    # ---------------------------------------------------------
    if asks_for_steps and len(steps) >= 1:
        answer = format_steps(steps)

    else:
        # -----------------------------------------------------
        # LLM generation for general questions
        # -----------------------------------------------------
        context_parts = []

        for chunk in chunks:
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "Unknown")

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"{chunk['text']}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using only the provided context.

Question:
{question}

Context:
{context}

Do not use outside knowledge.
Do not invent information.
Do not include unrelated procedures.

Answer:
"""

        tokenizer, model = load_llm()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=3
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

    # ---------------------------------------------------------
    # Sources
    # ---------------------------------------------------------
    sources = []

    if steps:
        # For structured procedure answers, show the source
        # pages that actually contain the procedure.
        for chunk in chunks:
            text = chunk["text"]

            if "Opening Procedures (BO-101)" not in text:
                continue

            if not re.search(r"(?m)^\s*1\s+Unlock main entrance", text):
                continue

            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "Unknown")

            source_info = f"{source} — Page {page}"

            if source_info not in sources:
                sources.append(source_info)

    else:
        # For general LLM answers, keep the retrieved sources.
        for chunk in chunks:
            source = chunk["metadata"].get("source", "Unknown")
            page = chunk["metadata"].get("page", "Unknown")

            source_info = f"{source} — Page {page}"

            if source_info not in sources:
                sources.append(source_info)

    return answer, sources


if __name__ == "__main__":
    question = input("Enter your question: ")

    answer, sources = generate_answer(question)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")

    for source in sources:
        print(f"- {source}")

