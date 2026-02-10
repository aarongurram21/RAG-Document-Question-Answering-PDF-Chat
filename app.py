import io
import tempfile
from typing import List, Tuple

import gradio as gr
from sentence_transformers import SentenceTransformer

from src.ingest import chunk_text, extract_text_from_pdf, prepare_documents
from src.qa_chain import QAChain
from src.retriever import FAISSRetriever


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_pipeline(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int) -> Tuple[QAChain, List[dict]]:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    retriever = FAISSRetriever(embedding_model)

    with io.BytesIO(pdf_bytes) as buffer:
        text = extract_text_from_pdf(buffer)

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts, metadatas = prepare_documents(chunks)
    retriever.add_documents(texts, metadatas)
    qa_chain = QAChain(retriever)
    return qa_chain, metadatas


def qa_workflow(pdf_file, question: str, chunk_size: int, chunk_overlap: int, top_k: int):
    if pdf_file is None:
        return "Please upload a PDF first.", ""

    pdf_bytes = pdf_file.read()
    qa_chain, metadatas = build_pipeline(pdf_bytes, chunk_size, chunk_overlap)
    answer, citations = qa_chain.answer(question, top_k=top_k)

    citation_strings = []
    for meta in citations:
        citation_strings.append(meta.get("source", ""))
    formatted_citations = ", ".join(citation_strings)
    sources = f"Sources: {formatted_citations}" if citation_strings else "Sources: none"
    return answer, sources


def main():
    examples = [
        [None, "What is the main conclusion?", 800, 100, 4],
        [None, "List key recommendations.", 800, 100, 4],
        [None, "Summarize the document.", 800, 100, 4],
    ]

    with gr.Blocks(title="PDF RAG Q&A") as demo:
        gr.Markdown("## PDF Retrieval-Augmented Q&A")
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="binary")
            question = gr.Textbox(label="Question", placeholder="Ask about the PDF...")
        with gr.Row():
            chunk_size = gr.Slider(256, 2000, value=800, step=64, label="Chunk size (characters)")
            chunk_overlap = gr.Slider(0, 800, value=100, step=10, label="Chunk overlap")
            top_k = gr.Slider(1, 10, value=4, step=1, label="Top-k retrieval")
        answer = gr.Textbox(label="Answer")
        citations = gr.Textbox(label="Citations")

        run_btn = gr.Button("Run")
        run_btn.click(
            qa_workflow,
            inputs=[pdf_input, question, chunk_size, chunk_overlap, top_k],
            outputs=[answer, citations],
        )

        gr.Markdown("### Evaluation Examples")
        gr.Examples(examples=examples, inputs=[pdf_input, question, chunk_size, chunk_overlap, top_k])

    demo.launch()


if __name__ == "__main__":
    main()
