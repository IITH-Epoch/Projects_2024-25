from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core import Document
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import fitz
from PIL import Image
import numpy as np

# Parallel text splitting
def process_documents_in_parallel(documents, splitter):
    with ThreadPoolExecutor(max_workers=4) as executor:
        split_results = executor.map(lambda doc: splitter.split_text(doc.text), documents)
    return [chunk for chunks in split_results for chunk in chunks]

# Build or update vector store
def process_and_index_data(directory, embedder, splitter):
    try:
        reader = SimpleDirectoryReader(directory)
        documents = reader.load_data()
        chunks = process_documents_in_parallel(documents, splitter)
        document_chunks = [Document(text=chunk) for chunk in chunks]

        return VectorStoreIndex.from_documents(documents=document_chunks, embed_model=embedder)
    except Exception as e:
        st.sidebar.error(f"Error building knowledge base: {e}")
        return None

def extract_images_from_pdf(pdf_path):
    """Extracts images from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        images = []
        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    except Exception as e:
        raise RuntimeError(f"Error extracting images from PDF: {e}")

def extract_text_with_easyocr(input_data, reader):
    """
    Performs OCR on an image using EasyOCR. 
    Accepts either an image file path or a PIL image object.
    """
    try:
        if isinstance(input_data, Image.Image):  # If input is a PIL Image object
            image = input_data
        elif isinstance(input_data, str):  # If input is a file path
            image = Image.open(input_data)
        else:
            raise ValueError("Input must be a file path (str) or a PIL Image object.")

        # Convert the PIL image to a NumPy array for EasyOCR
        image_np = np.array(image)
        result = reader.readtext(image_np)

        # Extract text and join into a single string
        return " ".join([text[1] for text in result]).strip()
    except Exception as e:
        raise RuntimeError(f"Error during OCR: {e}")

def update_vector_store(vector_store, new_documents, embedder, splitter):
    try:
        chunks = process_documents_in_parallel(new_documents, splitter)
        new_docs = [Document(text=chunk) for chunk in chunks]  # Treat chunks as raw text
        vector_store.add_documents(documents=new_docs, embed_model=embedder)
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Error updating knowledge base: {e}")
        return vector_store

    
def extract_key_sentences(model, embedder, text, top_n=5):
    """Extract top N key sentences based on embeddings."""
    doc = model(text)
    sentences = [sent.text for sent in doc.sents]
    embeddings = embedder.encode(sentences)
    doc_embedding = embeddings.mean(axis=0)
    similarities = [np.dot(sent_emb, doc_embedding) for sent_emb in embeddings]
    ranked_sentences = [sent for _, sent in sorted(zip(similarities, sentences), reverse=True)]
    return ranked_sentences[:top_n]

def combine_flowchart_outputs(page_outputs):
    """
    Combines flowchart outputs from multiple pages into a meaningful string for an LLM.

    Args:
        page_outputs (list): List of tuples where each tuple contains (flowchart, graph) 
                             for a single page.

    Returns:
        str: A combined textual representation of all pages.
    """
    combined_string = []
    
    for page_num, (flowchart, graph) in enumerate(page_outputs, start=1):
        # Page header
        combined_string.append(f"Page {page_num}:\n")
        
        # Extract nodes
        nodes = flowchart.get("nodes", [])
        combined_string.append("Nodes:\n")
        for node in nodes:
            combined_string.append(f"  - ID: {node['id']}, Text: \"{node['label']}\"\n")
        
        # Extract edges
        edges = flowchart.get("edges", [])
        combined_string.append("Edges:\n")
        for edge in edges:
            combined_string.append(f"  - From Node {edge['from']} to Node {edge['to']}\n")
        
        # Extract annotations
        annotations = flowchart.get("annotations", "").strip()
        if annotations:
            combined_string.append(f"Annotations: {annotations}\n")
        
        # Add a separator between pages
        combined_string.append("\n" + "-" * 50 + "\n")

    # Combine all parts into a single string
    return "".join(combined_string)