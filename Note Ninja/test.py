import os
import easyocr
import shutil
import atexit
import streamlit as st
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import spacy
from PIL import Image
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google import GeminiEmbedding
from llama_index.core import Document
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from utils import (
    process_and_index_data,
    extract_images_from_pdf,
    extract_text_with_easyocr,
    update_vector_store,
    extract_key_sentences,
    combine_flowchart_outputs
)

from ocr_utils import process_handwritten_script, process_arrow_rcnn_output
from tools import return_tool_list
from prompts import agent_prompt, recontextualize_query, summarize_flowcharts

@st.cache_resource
def load_img_searcher():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

@st.cache_resource
def load_qa_maker():
    model_name = "allenai/t5-small-squad2-question-generation"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def run_q_maker(tokenizer, model, input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    return output

TEMP_DIR = "./temp"
DATA_DIR = "./data"
 
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def cleanup_temp():
    shutil.rmtree(TEMP_DIR)

atexit.register(cleanup_temp)

# Set up API key for Gemini embedding
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize models
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
embedder = GeminiEmbedding(model_name="models/embedding-001")
splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50)
reader = easyocr.Reader(['en'])  # EasyOCR Reader

tool_list = return_tool_list()

buffer = ChatMemoryBuffer(token_limit=10000)
agent = ReActAgent.from_tools(tools=tool_list, llm=gemini_model, memory=buffer)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

def gen_questions(text, num_questions):
    """Generate questions from text"""
    tokenizer, model = load_qa_maker()
    key_sentences = extract_key_sentences(nlp, sentence_embedder, text, top_n=num_questions)
    questions = []
    for sentence in key_sentences:
        question = run_q_maker(tokenizer=tokenizer, model=model, input_string=sentence)
        questions.append(question[0])
    
    return questions

# Streamlit App
st.title("Note Ninja")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    
    # File Upload Section
    st.subheader("File Uploads")
    uploaded_note_file = st.file_uploader("Upload PDFs or images", type=['pdf', 'png', 'jpg', 'jpeg'])

    # Features & Toggles
    st.subheader("Features")
    enable_semantic_search = st.checkbox("Enable Semantic Search", value=True)
    enable_question_generation = st.checkbox("Enable Question Generation", value=False)

    # Settings Section
    st.subheader("Settings")
    num_questions = st.slider("Number of Questions", min_value=1, max_value=5, value=2, step=1)

# Session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ''

# Process uploaded files
if uploaded_note_file:
    temp_note_file_path = os.path.join(TEMP_DIR, uploaded_note_file.name)

    # Save file locally
    with open(temp_note_file_path, "wb") as f:
        f.write(uploaded_note_file.getbuffer())
    if uploaded_note_file.name not in st.session_state.uploaded_files:
        with st.spinner("Processing file..."):
            extracted_text = ""

            if uploaded_note_file.name.lower().endswith(".pdf"):
                with st.spinner("Extracting images from PDF..."):
                    images = extract_images_from_pdf(temp_note_file_path)
                with st.spinner("Performing OCR on extracted images..."):
                    extracted_info = [process_handwritten_script(img) for img in images]
                    extracted_info = [process_arrow_rcnn_output(info) for info in extracted_info]
                    flowchart_str = combine_flowchart_outputs(extracted_info)
                    extracted_text = summarize_flowcharts(gemini_model, flowchart_str)
            else:
                with st.spinner("Performing OCR on the uploaded image..."):
                    extracted_text = extract_text_with_easyocr(temp_note_file_path, reader)
                    if isinstance(temp_note_file_path, Image.Image):  # If input is a PIL Image object
                        image = temp_note_file_path
                    elif isinstance(temp_note_file_path, str):  # If input is a file path
                        image = Image.open(temp_note_file_path)
                        image = image.convert(mode='RGB')
                    else:
                        raise ValueError("Input must be a file path (str) or a PIL Image object.")
                    extracted_info = process_handwritten_script(image)
                    print(extracted_info)
                    extracted_info = process_arrow_rcnn_output(extracted_info)
                    flowchart_str = combine_flowchart_outputs([extracted_info])
                    extracted_text = summarize_flowcharts(gemini_model, flowchart_str)

                    st.success(extracted_text)
            st.session_state.extracted_text = extracted_text

            with st.spinner("Saving extracted text to file..."):
                text_file_path = os.path.join(DATA_DIR, f"{uploaded_note_file.name}.txt")
                with open(text_file_path, "w") as f:
                    f.write(extracted_text)

            with st.spinner("Indexing text into the knowledge base..."):
                if st.session_state.vector_store:
                    st.session_state.vector_store = update_vector_store(
                        st.session_state.vector_store,
                        [Document(**{'text': extracted_text})],
                        embedder,
                        splitter
                    )
                else:
                    st.session_state.vector_store = process_and_index_data(DATA_DIR, embedder, splitter)

            st.session_state.retriever = st.session_state.vector_store.as_retriever(similarity_top_k=2, vector_store_query_mode='mmr')
            st.session_state.uploaded_files.add(uploaded_note_file.name)
            st.sidebar.success("File processed and indexed.")
    # Generate questions after file processing
    if uploaded_note_file and enable_question_generation:
        with st.spinner("Generating questions from the uploaded content..."):
            questions = gen_questions(st.session_state.extracted_text, num_questions=num_questions)
            st.sidebar.success("Questions generated successfully!")

        st.header("Generated Questions")
        for i, question in enumerate(questions, 1):
            st.write(f"**Q{i}:** {question}")

with st.sidebar:
    st.subheader("Semantic Search")
    user_search_query = st.text_input("Search your notes:", placeholder="Type your search query here...")
    user_img_query = st.file_uploader("Image similarity search", type=['jpg', 'png', 'jpeg'])

    if user_search_query:            
        with st.spinner("Searching for relevant sections..."):
            retrieved_context = st.session_state.retriever.retrieve(user_search_query)
        
        st.subheader("Search Results")
        if retrieved_context:
            for idx, context in enumerate(retrieved_context, start=1):
                st.markdown(f"#### Result {idx}")
                st.write(context)
                st.markdown("---")
        else:
            st.info("No relevant sections found.")
            
    if user_img_query:
        with st.spinner("Processing image query..."):
            clip_model, clip_processor = load_img_searcher()
            img = Image.open(user_img_query).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = clip_model.get_image_features(**inputs)
                query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

            if st.session_state.vector_store:
                retriever = st.session_state.vector_store.as_retriever(similarity_top_k=20)
                docs = retriever.retrieve('Kitty kitty cat cat kit kat?')
                embeddings = [embedder.get_text_embedding(doc.text) for doc in docs]

                min_dim = min(query_embedding.shape[1], len(embeddings[0]))
                query_embedding_truncated = query_embedding[:, :min_dim].cpu().numpy()
                truncated_embeddings = [emb[:min_dim] for emb in embeddings]

                similarities = []
                for node_embedding in truncated_embeddings:
                    similarity = np.dot(query_embedding_truncated.flatten(), node_embedding)
                    similarities.append(similarity)

                sorted_indices = np.argsort(similarities)[::-1]
                sorted_similarities = [similarities[i] for i in sorted_indices]
                sorted_doc_embeds = [embeddings[i] for i in sorted_indices]

                st.subheader("Image Similarity Search Results")
                num_results = min(1, len(sorted_doc_embeds))  # Display the best match
                if num_results > 0:
                    for i in range(num_results):
                        doc = docs[i]
                        similarity = sorted_similarities[i]
                        st.write(f"**Text:** {doc.text}")
                        st.write("---")
                else:
                    st.write("No similar results found")
            else:
                st.warning("No vector store available for image search.")

# Chat-based query interface
if st.session_state.retriever:
    user_query = st.chat_input("Ask a question based on the uploaded notes:")    

    if user_query:
        st.session_state.conversation_memory.append({"user": user_query})

        with st.spinner("Recontextualizing your query..."):            
            recontextualized_query = recontextualize_query(gemini_model, user_query, st.session_state.conversation_memory, st.session_state.extracted_text)    # Recontextualize user query
            resp_li = [x for x in recontextualized_query.split('\n') if x != '']
            query_type = resp_li[0].split('Query Type:')[-1].strip().lower()
            output = resp_li[1].split('Output:')[-1].strip()

        with st.spinner("Generating response..."):
            if 'general' in query_type:
                st.session_state.conversation_memory.append({"assistant": output})
            else:
                retrieved_context = st.session_state.retriever.retrieve(user_query)
                retrieved_context = [doc.text for doc in retrieved_context if doc.score >= 0.75]

                if retrieved_context:
                    context = " ".join(retrieved_context)
                elif st.session_state.extracted_text.strip():
                    context = st.session_state.extracted_text
                else:
                    context = "No relevant context was provided."

                prompt = agent_prompt.format(context=context, recontextualized_query=recontextualized_query)

                # Generate agent response
                answer = agent.chat(message=prompt).response
                st.session_state.conversation_memory.append({"assistant": answer})

    for message in st.session_state.conversation_memory:
        for role, content in message.items():
            with st.chat_message(role):
                st.write(content)