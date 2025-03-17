import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from huggingface_hub import login
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from transformers import AutoTokenizer
import torch
from langchain_ollama import OllamaLLM
import os
from datetime import datetime
import re
import io
import contextlib
import requests
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate OpenMP libraries
os.environ["OMP_NUM_THREADS"] = "1"  # Limit to 1 thread to avoid conflicts


from tavily import TavilyClient
from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
import os
import torch
from .utils import classify_query_with_gemini, do_math,clean_latex_query_with_gemini,convert_text_to_markdown_with_gemini
from .prompts import summarize, question_answering, fact_verification, search, exploration
# from lsa import clustered_rag_lsa, summarize_it
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class MemorySaver:
    def __init__(self):
        self.memory = {}

    def save_document(self, doc_id, content, metadata=None):
        self.memory[doc_id] = {
            "query": content,
            "response": metadata or {}
        }

    def get_document(self, doc_id):
        return self.memory.get(doc_id, None)

    def exists(self, doc_id):
        return doc_id in self.memory

    def get_all_documents(self):
        return self.memory

    # def remove_document(self, doc_id):
    #     if doc_id in self.memory:
    #         del self.memory[doc_id]

    def to_json(self):
        return json.dumps(self.memory, indent=4)

    def load_from_json(self, json_str):
        self.memory = json.loads(json_str)

# Example usage
memory_store = MemorySaver()

tavily_api_key = "Tavily keyZ"
hf_token = "HF KEY" 
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

if not os.environ.get('HF_TOKEN'):
    os.environ['HF_TOKEN'] = hf_token

tavily = TavilyClient(tavily_api_key)

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "google api"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

tavily = TavilyClient(tavily_api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login(token="HF KEY")
LANGUAGE = "english"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA-compatible GPU detected.")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)


# Load a tokenizer (for example, BERT tokenizer)
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tavily = TavilyClient(tavily_api_key)
search = TavilySearchResults(max_results=2) 
tools = [search]  

# agent_executor = create_react_agent(model, tools)
# agent_executor_with_memory = create_react_agent(model, tools, checkpointer=memory)

def execute_task(model, tools, user_query):
    """
    Executes the task by running tools and passing their output to the model.

    Args:
        model: The LLaMA model to generate responses.
        tools: A list of tools to retrieve additional information.
        user_query: The user's input query.

    Returns:
        The model's response.
    """
    tool_results = [tool.run(user_query) for tool in tools]
    
    combined_query = f"User Query: {user_query}\n\nTool Results:\n"
    for i, result in enumerate(tool_results, 1):
        combined_query += f"{i}. {result}\n"
    
    response = model.generate(prompts=[combined_query])
    return response.generations[0][0].text  


def is_latex_request(query):
    """
    Checks if the user query is requesting LaTeX conversion.
    """
    latex_keywords = [
        "convert to LaTeX", "generate LaTeX", "LaTeX format",
        "transform to LaTeX", "create LaTeX document","Give latex code","Give the latex code","Give me the latex code"
    ]
    return any(keyword.lower() in query.lower() for keyword in latex_keywords)

def extract_relevant_content(query):
    """
    Sends the user query to an AI model to extract the relevant LaTeX content.
    """
    # print(clean_latex_query_with_gemini(query))
    return clean_latex_query_with_gemini(query)

def convert_to_latex(content, title="Generated Document", author="AI System"):
    """
    Converts the given content into a complete LaTeX document with a title, author, and date.
    The function detects common structures such as math expressions, lists, CSV-like tables, and code blocks,
    and converts them to appropriate LaTeX constructs.

    Args:
        content (str): The input text/content to convert.
        title (str): Title of the LaTeX document.
        author (str): Author of the LaTeX document.

    Returns:
        str: A string containing the complete LaTeX document.
    """
    # Get current date for the document
    date = datetime.today().strftime("%B %d, %Y")

    # Document preamble and header
    latex_doc = f"""\\documentclass{{article}}
\\usepackage{{graphicx}} % For images
\\usepackage{{amsmath}}  % For advanced math symbols
\\usepackage{{verbatim}} % For code blocks

\\title{{{title}}}
\\author{{{author}}}
\\date{{{date}}}

\\begin{{document}}
\\maketitle

"""

    # Initialize latex_content variable
    latex_content = ""

    # --- Content Detection and Conversion ---

    # 1. Detect and format mathematical expressions.
    #    (This is a simple detection; for complex documents, you may need more sophisticated parsing.)
    if re.search(r"\d+[+\-*/^=]+\d+", content):
        # Wrap the entire content in a display math environment.
        latex_content = f"\\[\n{content}\n\\]"
    # 2. Detect unordered lists (Markdown-style lists starting with '-' or '*').
    elif content.strip().startswith("- ") or content.strip().startswith("* "):
        items = content.split("\n")
        latex_list = "\n".join([f"    \\item {item[2:].strip()}" for item in items if item.strip()])
        latex_content = f"\\begin{{itemize}}\n{latex_list}\n\\end{{itemize}}"
    # 3. Detect ordered lists (lines starting with a number and a period).
    elif re.match(r"^\d+\.", content.strip()):
        items = content.split("\n")
        latex_list = "\n".join([f"    \\item {re.sub(r'^\d+\.\s*', '', item).strip()}" for item in items if item.strip()])
        latex_content = f"\\begin{{enumerate}}\n{latex_list}\n\\end{{enumerate}}"
    # 4. Detect CSV-like tables (content containing commas and newlines).
    elif "," in content and "\n" in content:
        rows = content.strip().split("\n")
        cols = rows[0].count(",") + 1
        col_format = " | ".join(["c"] * cols)
        latex_table = f"\\begin{{tabular}}{{{col_format}}}\n\\hline\n"
        # Process each row: split by comma and join columns with ' & '
        table_rows = []
        for row in rows:
            # Strip extra whitespace and join columns
            columns = [col.strip() for col in row.split(",")]
            table_rows.append(" & ".join(columns) + " \\\\")
        latex_table += "\n".join(table_rows)
        latex_table += "\n\\hline\n\\end{tabular}"
        latex_content = latex_table
    # 5. Detect code blocks (content starting with triple backticks).
    elif content.strip().startswith("```"):
        # Remove the backticks; you might want to preserve the language identifier if present.
        code = content.strip().strip("`").strip()
        latex_content = f"\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}"
    # 6. Default: Convert plain text by ensuring line breaks become paragraphs.
    else:
        # Replace single newlines with double newlines for LaTeX paragraph breaks.
        latex_content = content.replace("\n", "\n\n")

    # --- End Content Detection ---

    # Append the converted content and close the document.
    latex_doc += latex_content + "\n\n\\end{document}"

    return latex_doc
    

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

import numpy as np

def safe_str(value):
    """
    Safely converts a value to a string. If the value is a dictionary,
    it converts it to a JSON-formatted string.
    """
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value) if value is not None else ""


def retrieve_documents_with_memory(index, query_embedding, memory_documents, metadata, top_k, embedder, 
                                   memory_weight=0.3, relevance_threshold=0.3, device="cpu"):
    """
    Retrieve documents from the FAISS index by combining the current query embedding with an aggregated
    memory embedding computed from the latest 5 memory documents.
    
    Args:
        index: The FAISS index.
        query_embedding: Embedding for the current query (numpy array).
        memory_documents: List of memory documents (each should have 'query' and 'response').
        metadata: List of document metadata associated with the FAISS index.
        top_k: Number of documents to retrieve.
        embedder: The embedding model with an 'encode' method.
        memory_weight: Weight factor for memory contribution (0.0 means ignore memory, 1.0 means fully rely on memory).
        relevance_threshold: Minimum cosine similarity for a memory entry to be considered relevant.
        device: Device for the embedder (e.g., "cpu" or "cuda").
        
    Returns:
        A list of retrieved documents (from metadata) or relevant memory entries if FAISS returns nothing.
    """
    relevant_memory_embeddings = []
    relevant_memory_entries = []
    # Process only the latest 2 memory documents
    if memory_documents:
        latest_entries = memory_documents[-2:]
        for entry in latest_entries:
            if "query" in entry and "response" in entry:
                # Combine query and response safely
                
                combined_text = safe_str(entry["query"]) + " " + safe_str(entry["response"])
                mem_embedding = embedder.encode([combined_text], device=device).flatten()
                # Compute cosine similarity
                norm_query = np.linalg.norm(query_embedding)
                norm_mem = np.linalg.norm(mem_embedding)
                cosine_sim = (np.dot(query_embedding, mem_embedding) / (norm_query * norm_mem)
                              if norm_query and norm_mem else 0.0)
                
                # Only include if above the relevance threshold
                if cosine_sim >= relevance_threshold:
                    relevant_memory_embeddings.append(mem_embedding)
                    relevant_memory_entries.append(entry)
    
    # Combine memory embedding with the current query embedding if available
    if relevant_memory_embeddings:
        memory_embedding = np.mean(relevant_memory_embeddings, axis=0)
        combined_query_embedding = (1 - memory_weight) * query_embedding + memory_weight * memory_embedding
    else:
        combined_query_embedding = query_embedding
    # Ensure the embedding is 2D for FAISS
    combined_query_embedding = np.expand_dims(combined_query_embedding, axis=0)
    
    # Perform FAISS search
    distances, doc_indices = index.search(combined_query_embedding, top_k)
    
    # Retrieve documents using metadata
    retrieved_docs = [metadata[i] for i in doc_indices[0] if i < len(metadata)]
    
    # Fallback: if FAISS doesn't return any documents and we have relevant memory entries, return those.
    if not retrieved_docs and relevant_memory_entries:
        return relevant_memory_entries
    
    return retrieved_docs

# def retrieve_documents_with_memory(index, query_embedding, memory_store, metadata, top_k, embedder, 
#                                    memory_weight=0.3, relevance_threshold=0.3):
#     """
#     Retrieve documents from the FAISS index by combining the current query embedding with an 
#     aggregated memory embedding computed from the latest 5 memory chats. Only memory entries that are 
#     relevant (based on cosine similarity) to the current query are used.

#     If no documents are retrieved from FAISS and there are relevant memory entries, the function 
#     falls back to returning those memory interactions.

#     Args:
#         index: The FAISS index.
#         query_embedding: Embedding for the current query (numpy array).
#         memory_store: Object containing previous queries and responses.
#         metadata: List of document metadata associated with the FAISS index.
#         top_k: Number of documents to retrieve.
#         embedder: The embedding model with an `encode` method.
#         memory_weight: Weight factor for memory contribution (0.0 means ignore memory,
#                        1.0 means fully rely on memory).
#         relevance_threshold: Minimum cosine similarity for a memory entry to be considered relevant.
    
#     Returns:
#         A list of retrieved documents based on the metadata. Falls back to relevant memory entries 
#         if no documents are retrieved.
#     """
#     relevant_memory_embeddings = []
#     relevant_memory_entries = []
    
#     # Process only the latest 5 memory chats.
#     if hasattr(memory_store, "memory") and memory_store.memory:
#         latest_entries = list(memory_store.memory.values())[-5:]
#         for entry in latest_entries:
#             if "query" in entry and "response" in entry:
#                 # Convert and combine query and response safely.
#                 combined_text = safe_str(entry["query"]) + " " + safe_str(entry["response"])
#                 mem_embedding = embedder.encode([combined_text], device=device).flatten()
                
#                 # Compute cosine similarity between the current query and memory embedding.
#                 norm_query = np.linalg.norm(query_embedding)
#                 norm_mem = np.linalg.norm(mem_embedding)
#                 cosine_sim = (np.dot(query_embedding, mem_embedding) / (norm_query * norm_mem)
#                               if norm_query and norm_mem else 0.0)
                
#                 # Only include memory entries that exceed the relevance threshold.
#                 if cosine_sim >= relevance_threshold:
#                     relevant_memory_embeddings.append(mem_embedding)
#                     relevant_memory_entries.append(entry)
    
#     # If relevant memory entries exist, combine their aggregated embedding with the current query.
#     if relevant_memory_embeddings:
#         memory_embedding = np.mean(relevant_memory_embeddings, axis=0)
#         combined_query_embedding = (1 - memory_weight) * query_embedding + memory_weight * memory_embedding
#     else:
#         combined_query_embedding = query_embedding

#     # Ensure the combined embedding has the proper shape for the FAISS index.
#     combined_query_embedding = np.expand_dims(combined_query_embedding, axis=0)
    
#     # Perform FAISS search for the top_k documents.
#     distances, doc_indices = index.search(combined_query_embedding, top_k)
    
#     # Map FAISS indices to document metadata.
#     retrieved_docs = [metadata[i] for i in doc_indices[0] if i < len(metadata)]
    
#     # If no documents are retrieved and there are relevant memory entries, fall back to memory.
#     if not retrieved_docs and relevant_memory_entries:
#         return relevant_memory_entries
    
#     return retrieved_docs





# def generate_subqueries(user_query, ollama_model, memory_store):
#     # Get the last 5 queries from memory
#     memory_keys = list(memory_store.memory.keys())[-5:]  # Get last 5 query IDs
#     previous_queries = [memory_store.memory[key]["query"] for key in memory_keys]

#     # Format memory context
#     memory_context = "\n".join([f"Previous Query: {query}" for query in previous_queries])

#     prompt = f"""
#     {memory_context}

#     Current Query: {user_query}

#     You are an expert at understanding and correlating user queries. If the query consists of distinct sub-questions, and a clear distinction is observed, break it down into meaningful and logically separate sub-questions **only if necessary.** Otherwise, retain the query as is. **If the query is already a complete and meaningful statement, return it without changes.** Minor grammatical adjustments are allowed if required.

#     Guidelines:
#     1. **If the query is already a valid and complete question or statement, return it as is without splitting.**  
#     2. **Break down the query only when it contains multiple, distinct parts that can stand alone as sub-questions.**  
#     3. **Maintain the original intent and context of the query when creating sub-questions.**  
#     4. **Provide only the output without any additional explanations or comments.**
#     5. **There may be typos or grammatical mistakes. Fix them as per necessity.**
#     6. **If the query appears to reference a previous query, use the provided memory to frame it as a complete and independent question.**

#     ### Example 1:  
#     **Input:** "What is the QS and NIRF ranking of IITH?"  
#     **Output:**  
#     - "What is the QS ranking of IITH?"  
#     - "What is the NIRF ranking of IITH?"  

#     ### Example 2:  
#     **Input:** "Summarize about IIT."  
#     **Output:**  
#     - "Summarize about IIT."  

#     ### Example 3:  
#     **Input:** "Explain the differences between QS and NIRF rankings."  
#     **Output:**  
#     - "What are QS rankings?"  
#     - "What are NIRF rankings?"  
#     - "What are the differences between QS and NIRF rankings?"  

#     ### Example 4:  
#     **Input:** "Who is Rajesh Kedia?"  
#     **Output:**  
#     - "Who is Rajesh Kedia?"  

#     ### Example 5:  
#     **Input:** "What is Lambda IITH?"  
#     **Output:**  
#     - "What is Lambda IITH?" 

#     ### Example 5:  
#     **Input:** 3+5/4*9+(9+5)"  
#     **Output:**  
#     - "3+5/4*9+(9+5)" 

#     ### Example 5:  
#     **Input:** add 6 and 5 and then subtract 4 and then multiply 5"  
#     **Output:**  
#     - "((6+5)-4)*5" 

#     Query: {user_query}  
#     Output:
#     """
#     # print("Query type: ",classify_query_with_gemini(user_query))

#     formatted_prompt = prompt.format(user_query=user_query)
#     response = ollama_model.invoke(formatted_prompt)
    
#     # Extract sub-questions from the response
#     subqueries = response.strip().split("\n")
#     # Filter out explanatory text or any unwanted lines
#     cleaned_subqueries = [
#         subquery.strip() for subquery in subqueries 
#         if subquery.strip() and not subquery.startswith(("**", "To answer", "These", "The next"))
#     ]

#     if len(cleaned_subqueries) == 0 or (len(cleaned_subqueries) == 1 and cleaned_subqueries[0] == user_query.strip()):
#         return [user_query.strip()]

#     return cleaned_subqueries
###### new one djwala

def generate_subqueries(user_query, ollama_model, memory_documents):
    """
    Generate subqueries for the given user_query using the provided LLM model.
    Uses memory documents (a list) retrieved from the frontend database as context.
    
    Args:
        user_query (str): The current query.
        ollama_model: The LLM instance with an invoke() method.
        memory_documents (list): A list of memory documents (each a dict with a "query" key).
        
    Returns:
        List[str]: A list of subqueries.
    """
    # Use only the last 2 memory documents, if available.
    latest_entries = memory_documents[-2:] if memory_documents else []
    previous_queries = [doc.get("query", "") for doc in latest_entries if doc.get("query")]
    
    # Format the memory context (if any).
    memory_context = "\n".join([f"Previous Query: {query}" for query in previous_queries]) if previous_queries else ""
    
    prompt = f"""
    {memory_context}

    Current Query: {user_query}

    You are an expert at understanding and correlating user queries. If the query consists of distinct sub-questions, and a clear distinction is observed, break it down into meaningful and logically separate sub-questions **only if necessary.** Otherwise, retain the query as is. **If the query is already a complete and meaningful statement, return it without changes.** Minor grammatical adjustments are allowed if required.

    Guidelines:
    1. **If the query is already a valid and complete question or statement, return it as is without splitting.**
    2. **Break down the query only when it contains multiple, distinct parts that can stand alone as sub-questions.**
    3. **Maintain the original intent and context of the query when creating sub-questions.**
    4. **Provide only the output without any additional explanations or comments.**
    5. **There may be typos or grammatical mistakes. Fix them as per necessity.**
    6. **If the query appears to reference a previous query, use the provided memory to frame it as a complete and independent question.**

    ### Example 1:
    **Input:** "What is the QS and NIRF ranking of IITH?"
    **Output:**
    - "What is the QS ranking of IITH?"
    - "What is the NIRF ranking of IITH?"

    ### Example 2:
    **Input:** "Summarize about IIT."
    **Output:**
    - "Summarize about IIT."

    ### Example 3:
    **Input:** "Explain the differences between QS and NIRF rankings."
    **Output:**
    - "What are QS rankings?"
    - "What are NIRF rankings?"
    - "What are the differences between QS and NIRF rankings?"

    ### Example 4:
    **Input:** "Who is Rajesh Kedia?"
    **Output:**
    - "Who is Rajesh Kedia?"

    ### Example 5:
    **Input:** "What is Lambda IITH?"
    **Output:**
    - "What is Lambda IITH?"

    ### Example 6:
    **Input:** "3+5/4*9+(9+5)"
    **Output:**
    - "3+5/4*9+(9+5)"

    ### Example 7:
    **Input:** "add 6 and 5 and then subtract 4 and then multiply 5"
    **Output:**
    - "((6+5)-4)*5"

    Query: {user_query}
    Output:
    """
    
    response = ollama_model.invoke(prompt)
    
    # Split and clean the response to extract subqueries.
    subqueries = [line.strip() for line in response.strip().split("\n") if line.strip()]
    
    # If no subqueries were generated or if the only subquery is identical to the user query,
    # return the original query.
    if not subqueries or (len(subqueries) == 1 and subqueries[0] == user_query.strip()):
        return [user_query.strip()]
    
    return subqueries

# import numpy as np

# def generate_subqueries(user_query, ollama_model, memory_documents, embedder=None, relevance_threshold=0.3):
#     """
#     Generate subqueries for the given user_query using the provided LLM model.
#     If memory documents are provided, they are used as context only if at least one of the latest 5 
#     memory entries is relevant to the current query (as measured by cosine similarity).

#     Args:
#         user_query (str): The user's query.
#         ollama_model: The LLM model instance with an invoke() method.
#         memory_documents (list): A list of memory documents (each a dict with at least a "query" key).
#         embedder: (Optional) An embedding model with an `encode` method to compute similarities.
#         relevance_threshold (float): Minimum cosine similarity for a memory entry to be considered relevant.

#     Returns:
#         List[str]: A list of subqueries.
#     """
#     # Get the last 5 memory documents (if available)
#     latest_entries = memory_documents[-5:] if memory_documents else []
#     previous_queries = [doc.get("query", "") for doc in latest_entries if doc.get("query")]

#     memory_context = ""
#     # Only include memory context if an embedder is provided and at least one memory query is relevant.
#     if previous_queries and embedder is not None:
#         # Encode the user query
#         user_embedding = embedder.encode([user_query], device="cpu").flatten()
#         max_similarity = 0.0
#         for query_text in previous_queries:
#             mem_embedding = embedder.encode([query_text], device="cpu").flatten()
#             norm_user = np.linalg.norm(user_embedding)
#             norm_mem = np.linalg.norm(mem_embedding)
#             cosine_sim = (np.dot(user_embedding, mem_embedding) / (norm_user * norm_mem)
#                           if norm_user and norm_mem else 0.0)
#             max_similarity = max(max_similarity, cosine_sim)
        
#         # If the maximum similarity meets the threshold, include memory context.
#         if max_similarity >= relevance_threshold:
#             memory_context = "\n".join([f"Previous Query: {q}" for q in previous_queries])
#     elif previous_queries:
#         # If no embedder is provided, include memory context by default.
#         memory_context = "\n".join([f"Previous Query: {q}" for q in previous_queries])
    
#     prompt = f"""
#     {memory_context}

#     Current Query: {user_query}

#     You are an expert at understanding and correlating user queries. If the query consists of distinct sub-questions, and a clear distinction is observed, break it down into meaningful and logically separate sub-questions **only if necessary.** Otherwise, retain the query as is. **If the query is already a complete and meaningful statement, return it without changes.** Minor grammatical adjustments are allowed if required.

#     Guidelines:
#     1. **If the query is already a valid and complete question or statement, return it as is without splitting.**  
#     2. **Break down the query only when it contains multiple, distinct parts that can stand alone as sub-questions.**  
#     3. **Maintain the original intent and context of the query when creating sub-questions.**  
#     4. **Provide only the output without any additional explanations or comments.**
#     5. **There may be typos or grammatical mistakes. Fix them as per necessity.**
#     6. **If the query appears to reference a previous query, use the provided memory to frame it as a complete and independent question.**

#     ### Example 1:  
#     **Input:** "What is the QS and NIRF ranking of IITH?"  
#     **Output:**  
#     - "What is the QS ranking of IITH?"  
#     - "What is the NIRF ranking of IITH?"  

#     ### Example 2:  
#     **Input:** "Summarize about IIT."  
#     **Output:**  
#     - "Summarize about IIT."  

#     ### Example 3:  
#     **Input:** "Explain the differences between QS and NIRF rankings."  
#     **Output:**  
#     - "What are QS rankings?"  
#     - "What are NIRF rankings?"  
#     - "What are the differences between QS and NIRF rankings?"  

#     ### Example 4:  
#     **Input:** "Who is Rajesh Kedia?"  
#     **Output:**  
#     - "Who is Rajesh Kedia?"  

#     ### Example 5:  
#     **Input:** "What is Lambda IITH?"  
#     **Output:**  
#     - "What is Lambda IITH?" 

#     ### Example 6:  
#     **Input:** "3+5/4*9+(9+5)"  
#     **Output:**  
#     - "3+5/4*9+(9+5)" 

#     ### Example 7:  
#     **Input:** "add 6 and 5 and then subtract 4 and then multiply 5"  
#     **Output:**  
#     - "((6+5)-4)*5" 

#     Query: {user_query}  
#     Output:
#     """
    
#     response = ollama_model.invoke(prompt)
#     subqueries = [line.strip() for line in response.strip().split("\n") if line.strip()]
    
#     if not subqueries or (len(subqueries) == 1 and subqueries[0] == user_query.strip()):
#         return [user_query.strip()]
    
#     return subqueries


def load_llama_model(model_name="llama3.1",device=device):
    """Initialize Ollama and load LLaMA model locally."""
    return OllamaLLM(model=model_name, device="cuda" if torch.cuda.is_available() else "cpu")



def validate_relevance_llmresp(subquery, final_resp, llm):
    """
    Validates if the final generated response is relevant to the given subquery.

    Args:
        subquery: The subquery to validate against.
        final_resp: The final response generated by the LLM.
        llm: The LLM used for relevance validation.

    Returns:
        True if the response is deemed 'relevant', otherwise False.
    """
    prompt = (
        f"Query: {subquery}\n\n"
        f"Response Generated:\n{final_resp}\n\n"
        "Task:\n"
        "Evaluate if the response generated is relevant to the query and answers it in a satisfactory manner. Respond with:\n"
        "- 'Relevant' if it matches the query.\n"
        "- 'Not Relevant' if it does not match the query.\n"
        "Only provide the response, no additional text."
    )
    response = llm.invoke(prompt).strip()
    return response.lower() != "not relevant"


def validate_relevance(subquery, retrieved_docs, llm):
    """
    Validates if any of the retrieved documents are relevant to the given subquery.

    Args:
        subquery: The subquery to validate against.
        retrieved_docs: A list of retrieved document dictionaries.
        llm: The LLM used for relevance validation.

    Returns:
        True if at least one document is deemed 'relevant', otherwise False.
    """
    for doc in retrieved_docs:
        # Construct a representative text for the document using safe_str.
        doc_title = safe_str(doc.get("title", ""))
        doc_summary = safe_str(doc.get("summary", ""))
        doc_text = f"Title: {doc_title}\nSummary: {doc_summary}".strip()
        if not doc_text:
            # If both are empty, join all fields using safe_str.
            doc_text = " ".join(safe_str(v) for v in doc.values())
            
        prompt = (
            f"Query: {subquery}\n\n"
            f"Retrieved Document:\n{doc_text}\n\n"
            "Task:\n"
            "Evaluate if the retrieved document is relevant to the query, and if the query can be satisfactorily answered using information from this document. Respond with:\n"
            "- 'Relevant' if yes.\n"
            "- 'Not Relevant' if no.\n"
            "Only provide the response, no additional text."
        )
        response = llm.invoke(prompt).strip().lower()
        if response == "relevant":
            return True
    return False

def execute_code(code_str):
    """
    Executes a block of Python code provided as a string and returns its output.

    For security reasons, the built-ins are restricted. Adjust the allowed built-ins as needed.
    """
    import io
    import contextlib

    stdout_capture = io.StringIO()
    # Optionally, allow a restricted set of built-ins:
    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "dict": dict,
        "list": list,
        "set": set,
        "min": min,
        "max": max,
        "sum": sum,
        # Add more if needed.
    }
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, {"__builtins__": safe_builtins})
    except Exception as e:
        return f"Error during execution: {e}"
    return stdout_capture.getvalue().strip()


# def process_query_with_validation(
#     query,
#     index_path=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index',
#     metadata_path=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json',
#     embedder=model,
#     tavily=None,
#     top_k_default=5
# ):
#     """
#     Process the query by retrieving documents, validating their relevance, and falling back if needed.
    
#     Args:
#         query: The user's query.
#         index_path: Path to the FAISS index.
#         metadata_path: Path to the metadata file.
#         embedder: Embedding model used to encode queries.
#         tavily: (Optional) Fallback search client.
#         top_k_default: Default number of documents to retrieve if no classification match.
        
#     Returns:
#         The final generated response.
#     """

#     # Load the LLaMA model for subquery generation and response generation.
#     ollama_model = load_llama_model(model_name="llama3.1", device=device)

#     # Check if the query is a code snippet
#     if query.strip().startswith("```python"):
#         code_lines = query.strip().splitlines()
#         if code_lines[-1].strip() == "```":
#             code_str = "\n".join(code_lines[1:-1])
#         else:
#             code_str = "\n".join(code_lines[1:])
#         print("Detected code. Executing code snippet...")
#         return execute_code(code_str).strip()

#     # Handle LaTeX requests
#     if is_latex_request(query):
#         content = extract_relevant_content(query)
#         latex_output = convert_to_latex(content)
#         return f"Here is your LaTeX code:\n\n```latex\n{latex_output}\n```"

#     # === Step 1: Classify the whole query using Gemini ===
#     main_query_type = classify_query_with_gemini(query)
#     print(f"Main query classified as: {main_query_type}")

#     # === Step 2: Generate subqueries using the provided model and memory context ===
#     subqueries = generate_subqueries(query, ollama_model, memory_store)
#     print("Generated subqueries:", subqueries)

#     # === Step 3: Load FAISS index and metadata ===
#     index = load_faiss_index(index_path)
#     with open(metadata_path, "r", encoding="utf-8") as metadata_file:
#         metadata = json.load(metadata_file)

#     retrieved_context = []

#     # Decide on the number of documents to retrieve based on the main query type.
#     type_to_topk = {
#         "summarization": 20,
#         "question_answering": 5,
#         "search": 10,
#         "fact_verification": 8,
#         "exploration": 15,
#         "math": None  # special handling below
#     }
#     top_k_for_query = type_to_topk.get(main_query_type, top_k_default)

#     # === Step 4: Process each subquery ===
#     for subquery in subqueries:
#         subquery = subquery.strip()
#         # For math queries, handle separately.
#         if main_query_type == "math":
#             math_result = do_math(subquery)
#             retrieved_context.append({"subquery": subquery, "context": math_result})
#             continue

#         # Encode the subquery
#         query_embedding = embedder.encode([subquery], device=device).flatten()

#         # Retrieve documents using the embedding, memory, and metadata
#         retrieved_docs = retrieve_documents_with_memory(
#             index, query_embedding, memory_store, metadata, top_k_for_query,SentenceTransformer('all-MiniLM-L6-v2')
#         )

#         # Validate the relevance of the retrieved documents
#         is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
#         if not is_relevant:
#             print(f"Subquery '{subquery}' documents not relevant, retrying with more documents...")
#             retrieved_docs = retrieve_documents_with_memory(
#                 index, query_embedding, memory_store, metadata, top_k_for_query * 2,SentenceTransformer('all-MiniLM-L6-v2')
#             )
#             is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
#             # # Final fallback to Tavily if still not relevant
#             # if not is_relevant:
#             #     print(f"Subquery '{subquery}' failed validation. Performing Tavily search...")
#             #     tavily_search = TavilySearchResults(max_results=2)
#             #     retrieved_docs = [{"doc": result} for result in tavily_search.run(subquery)]

#         retrieved_context.append({
#             "subquery": subquery,
#             "context": retrieved_docs
#         })

#     # Combine the contexts for all subqueries into a single string.
#     combined_context = "\n".join(
#         f"Subquery: {item['subquery']}\nContext: {item['context']}"
#         for item in retrieved_context
#     )

#     # === Step 5: Build the final prompt based on main query classification ===
#     prompt = (
#         "You are an intelligent AI assistant. You will be provided with a user query divided into subqueries and a context. "
#         "Your task is to generate a proper response for the subqueries provided with the context (documents retrieved) and answer as a whole together.\n\n"
#         "Remember, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence, the persons or the queries are related to IITH only.\n\n"
#         f"Original Query: {query}\n\n"
#         f"Subqueries: {', '.join(subqueries)}\n\n"
#         f"Context:\n{combined_context}\n\n"
#         "Guidelines:\n"
#         "1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
#         "2. Maintain a professional tone.\n"
#         "3. Address each subquery comprehensively without omitting details.\n"
#         "4. Provide precise and accurate responses, ensuring that only relevant information directly related to the query is included.\n"
#         "5. Do not include any relevance scores in the final response.\n"
#     )

#     # Choose the response generation template based on the main query type.
#     if main_query_type == "summarization":
#         final_response = ollama_model.invoke(
#             summarize.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
#         )
#     elif main_query_type in ["question_answering", "search", "exploration"]:
#         final_response = ollama_model.invoke(
#             exploration.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
#         )
#     elif main_query_type == "fact_verification":
#         final_response = ollama_model.invoke(
#             fact_verification.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
#         )
#     else:
#         final_response = ollama_model.invoke(prompt)

#     # === Step 6: Validate final response; if not relevant, retry up to max_retries ===
#     is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
#     retry_count = 0
#     max_retries = 3
#     while not is_relevant_llmresp and retry_count < max_retries:
#         print(f"Final response not relevant, retrying... Attempt {retry_count + 1}")
#         final_response = ollama_model.invoke(prompt)
#         is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
#         retry_count += 1

#     # Save the query and response to memory
#     memory_store.save_document(query, final_response)
#     return final_response.strip()

import json
from pymongo import MongoClient
import numpy as np

def process_query_with_validation(
    query,
    user_email,  # New parameter for the user's email
    index_path=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index',
    metadata_path=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json',
    embedder=model,
    tavily=None,
    top_k_default=5
):
    """
    Process the query by retrieving documents from FAISS using the user's memory
    loaded from the database. The memory is used to aggregate context via embeddings.
    
    Args:
        query: The user's query.
        user_email: The email of the user (used to load memory from DB).
        index_path: Path to the FAISS index.
        metadata_path: Path to the metadata file.
        embedder: Embedding model used to encode queries.
        tavily: (Optional) Fallback search client.
        top_k_default: Default number of documents to retrieve if no classification match.
        
    Returns:
        The final generated response.
    """

    # --- Load the LLaMA model for subquery generation and response generation ---
    ollama_model = load_llama_model(model_name="llama3.1", device=device)

    # --- Check for code snippet or LaTeX request as before ---
    if query.strip().startswith("```python"):
        code_lines = query.strip().splitlines()
        if code_lines[-1].strip() == "```":
            code_str = "\n".join(code_lines[1:-1])
        else:
            code_str = "\n".join(code_lines[1:])
        print("Detected code. Executing code snippet...")
        return execute_code(code_str).strip()

    if is_latex_request(query):
        content = extract_relevant_content(query)
        latex_output = convert_to_latex(content)
        return f"Here is your LaTeX code:\n\n```latex\n{latex_output}\n```"

    # --- Step 1: Classify the whole query using Gemini ---
    main_query_type = classify_query_with_gemini(query)
    print(f"Main query classified as: {main_query_type}")

    # --- Step 2: Load the user's memory from MongoDB instead of an in-memory store ---
    MONGODB_URI = "mongodb+srv://Scorpion:TL_real_time_chat@iithgpt.y8hre.mongodb.net/?retryWrites=true&w=majority&appName=IITHGPT"
    client = MongoClient(MONGODB_URI)
    # Use the same database/collection names as in your chat endpoint
    db = client["test"]
    memory_documents = list(db["chatHistory"].find({"email": user_email}).sort("timestamp", 1))
    print(f"Loaded {len(memory_documents)} memory documents for {user_email}")

    # --- Step 3: Generate subqueries using the model and memory context ---
    # (Assuming generate_subqueries now accepts memory documents instead of memory_store)
    subqueries = generate_subqueries(
        user_query=query, 
        ollama_model=ollama_model, 
        memory_documents=memory_documents,
    )
    print("Generated subqueries:", subqueries)

    # --- Step 4: Load FAISS index and metadata ---
    index = load_faiss_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    retrieved_context = []
    type_to_topk = {
        "summarization": 20,
        "question_answering": 5,
        "search": 10,
        "fact_verification": 8,
        "exploration": 15,
        "math": None  # special handling below
    }
    top_k_for_query = type_to_topk.get(main_query_type, top_k_default)

    # --- Step 5: Process each subquery ---
    for subquery in subqueries:
        subquery = subquery.strip()
        if main_query_type == "math":
            math_result = do_math(subquery)
            retrieved_context.append({"subquery": subquery, "context": math_result})
            continue

        query_embedding = embedder.encode([subquery], device=device).flatten()

        # Now call the retrieval function with memory_documents from the DB:
        retrieved_docs = retrieve_documents_with_memory(
            index, query_embedding, memory_documents, metadata, top_k_for_query, SentenceTransformer('all-MiniLM-L6-v2')
        )

        is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
        if not is_relevant:
            print(f"Subquery '{subquery}' documents not relevant, retrying with more documents...")
            retrieved_docs = retrieve_documents_with_memory(
                index, query_embedding, memory_documents, metadata, top_k_for_query * 2, SentenceTransformer('all-MiniLM-L6-v2')
            )
            is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)

        retrieved_context.append({
            "subquery": subquery,
            "context": retrieved_docs
        })

    combined_context = "\n".join(
        f"Subquery: {item['subquery']}\nContext: {item['context']}"
        for item in retrieved_context
    )

    # --- Step 6: Build the final prompt based on main query classification ---
    prompt = (
        "You are an intelligent AI assistant. You will be provided with a user query divided into subqueries and a context. "
        "Your task is to generate a proper response for the subqueries provided with the context (documents retrieved) and answer as a whole together.\n\n"
        "Remember, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence, the persons or the queries are related to IITH only.\n\n"
        f"Original Query: {query}\n\n"
        f"Subqueries: {', '.join(subqueries)}\n\n"
        f"Context:\n{combined_context}\n\n"
        "Guidelines:\n"
        "1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
        "2. Maintain a professional tone.\n"
        "3. Address each subquery comprehensively without omitting details.\n"
        "4. Provide precise and accurate responses, ensuring that only relevant information directly related to the query is included.\n"
        "5. Do not include any relevance scores in the final response.\n"
    )

    if main_query_type == "summarization":
        final_response = ollama_model.invoke(
            summarize.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
        )
    elif main_query_type in ["question_answering", "search", "exploration"]:
        final_response = ollama_model.invoke(
            exploration.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
        )
    elif main_query_type == "fact_verification":
        final_response = ollama_model.invoke(
            fact_verification.format(user_query=query, context=combined_context, subqueries=", ".join(subqueries))
        )
    else:
        final_response = ollama_model.invoke(prompt)

    is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
    retry_count = 0
    max_retries = 3
    while not is_relevant_llmresp and retry_count < max_retries:
        print(f"Final response not relevant, retrying... Attempt {retry_count + 1}")
        final_response = ollama_model.invoke(prompt)
        is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
        retry_count += 1

    # Optionally, you can update the DB memory here if desired.
    # For now, we're relying on the DB to persist the conversation history.
    markdown_response = convert_text_to_markdown_with_gemini(final_response)
    return markdown_response
    # return final_response.strip()


# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.llms import OpenAI
# from langchain.memory import ConversationBufferMemory

# # Global objects: (These must be defined in your environment.)
# # For example, you might already have:
# # memory_store = MemorySaver()
# # index = load_faiss_index(index_path)
# # with open(metadata_path, "r", encoding="utf-8") as f:
# #     metadata = json.load(f)
# # embedder = SentenceTransformer('all-MiniLM-L6-v2')
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ------------------------------------------------------------------
# # Wrap process_query_with_validation as a LangChain Tool.
# # ------------------------------------------------------------------
# def validated_query_tool(query: str) -> str:
#     """
#     Wraps the process_query_with_validation function to be used as a LangChain tool.
#     """
#     try:
#         result = process_query_with_validation(query)
#         return result
#     except Exception as e:
#         return f"Error processing query: {e}"

# validated_tool = Tool(
#     name="Validated Query Processor",
#     func=validated_query_tool,
#     description=(
#         "Processes a query using a corrective and adaptive RAG pipeline with validation. "
#         "This tool retrieves documents, validates relevance, and applies corrective actions to "
#         "generate a final answer based on the provided query."
#     )
# )

# def summarization_tool_wrapper(text: str) -> str:
#     prompt = f"Summarize the following text succinctly:\n\n{text}"
#     summary = llm.invoke(prompt)
#     return summary.strip()

# def plan_actions(query: str) -> str:
#     prompt = f"""You are an intelligent agent that must answer the following query using available tools.
# Query: {query}

# Plan the steps you will take to answer the query. List each step in order along with the tool you should use (e.g., "Document Retrieval", "Summarization") and any parameters.
# Only provide the plan as a list of steps.
#     """
#     plan = llm.invoke(prompt)
#     return plan.strip()

# # ------------------------------------------------------------------
# # Initialize the LangChain LLM and memory.
# # ------------------------------------------------------------------
# llm = OpenAI(temperature=0)
# agent_memory = ConversationBufferMemory(memory_key="chat_history")

# # ------------------------------------------------------------------
# # Define the tools for the agent, including the new validated query tool.
# # ------------------------------------------------------------------
# tools = [
#     Tool(
#         name="Summarization",
#         func=summarization_tool_wrapper,
#         description="Summarizes a given block of text into a concise summary."
#     ),
#     validated_tool  # Our new tool wrapping process_query_with_validation.
# ]

# # ------------------------------------------------------------------
# # Initialize the agent using LangChain's zero-shot ReAct agent type.
# # ------------------------------------------------------------------
# agent = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=agent_memory,
#     verbose=True
# )

# # ------------------------------------------------------------------
# # Define the overall agentic RAG function.
# # ------------------------------------------------------------------
# def agentic_rag(query: str) -> str:
#     """
#     Runs an agentic retrieval-augmented generation process using LangChain.
#     The agent uses planning, tool invocation, and iterative refinement to answer the query.
#     """
#     # Optionally, display a plan for debugging.
#     plan = plan_actions(query)
#     print("Generated Plan:\n", plan)
    
#     # Run the agent with the query.
#     final_response = agent.run(query)
#     return final_response

# # ------------------------------------------------------------------
# # Example usage.
# # ------------------------------------------------------------------
# if __name__ == "__main__":
#     user_query = "Tell me about him."
#     response = agentic_rag(user_query)
#     print("Final Response:\n", response)


# def process_query_with_validation(query, index=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index', metadata=r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json',embedder=model, tavily=None, top_k=5):
#     """
#     Process the query by retrieving documents, validating their relevance, and falling back to Tavily if needed.

#     Args:
#         query: User's query.
#         index: FAISS index for retrieval.
#         metadata: Metadata associated with the FAISS index.
#         ollama_model: LLaMA model for subqueries and response generation.
#         embedder: Embedding model to encode queries.
#         tavily: Tavily client for fallback search.
#         top_k: Number of documents to retrieve initially.

#     Returns:
#         Final response generated by the LLM.
#     """
#     ollama_model=load_llama_model(model_name="llama3.1",device=device)
#     if query.strip().startswith("```python"):
#         # Strip off the markdown markers if needed.
#         code_lines = query.strip().splitlines()
#         # Remove the first and last line (assuming last line is "```")
#         if code_lines[-1].strip() == "```":
#             code_str = "\n".join(code_lines[1:-1])
#         else:
#             code_str = "\n".join(code_lines[1:])
#         print("Detected code. Executing code snippet...")
#         return execute_code(code_str).strip()

#     # Handle LaTeX conversion requests
#     print(is_latex_request(query))
#     if is_latex_request(query):
#         content = extract_relevant_content(query)
#         latex_output = convert_to_latex(content)
#         return f"Here is your LaTeX code:\n\n```latex\n{latex_output}\n```"

#     # Check if the main query or any subqueries already exist in memory
#     subqueries = generate_subqueries(query, ollama_model,memory_store)
#     # print("Subqueries:", subqueries)
#     retrieved_context = []
#     index = load_faiss_index(index_path)
#     with open(metadata_path, "r", encoding="utf-8") as metadata_file:
#         metadata = json.load(metadata_file)
#     for subquery in subqueries:
#         print(type(subquery))
#         query_embedding = embedder.encode([subquery], device=device).flatten()
#         query_type = classify_query_with_gemini(subquery)
#         if query_type == "summarization":
#             # Retrieve more documents for comprehensive coverage
#             top_k = 20
#         elif query_type == "question_answering":
#             # Focus on precise and concise documents
#             top_k = 5
#         elif query_type == "search":
#             # Balance between precision and coverage
#             top_k = 10
#         elif query_type == "fact_verification":
#             # Retrieve documents explicitly supporting or refuting the fact
#             top_k = 8
#         elif query_type == "exploration":
#             # Retrieve a wide variety of documents for broader coverage
#             top_k = 15
#         elif query_type == "math":
#             retrieved_context.append({"subquery": subquery, "context": do_math(subquery)})
#             continue
#         else:
#             # Fallback logic for unknown query types
#             top_k = 5
#         retrieved_docs = retrieve_documents_with_memory(index, query_embedding,memory_store, metadata, top_k)
#         is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
#         # print(retrieved_docs)
#         # Fallback logic
#         if not is_relevant:
#             print(f"Subquery '{subquery}' documents not relevant, retrying...")
            
#             retrieved_docs = retrieve_documents_with_memory(index, query_embedding,memory_store, metadata, top_k * 2)
#             is_relevant = validate_relevance(subquery, retrieved_docs, ollama_model)
#             # # Final fallback to Tavily if still not relevant
#             # if not is_relevant:
#             #     print(f"Subquery '{subquery}' failed validation. Performing Tavily search...")
#             #     tavily_search = TavilySearchResults(max_results=2)
#             #     retrieved_docs = [{"doc": result} for result in tavily_search.run(subquery)]

#         retrieved_context.append({
#             "subquery": subquery,
#             "context": retrieved_docs
#         })

#     combined_context = "\n".join(
#         f"Subquery: {item['subquery']}\nContext: {item['context']}"
#         for item in retrieved_context
#     )

#     prompt = (
#         "You are an intelligent AI assistant. You will be provided with a user query divided into subqueries and a context. Your task is to generate proper response for the subqueries provided with the context(documents retrived) and answer as a whole together.\n\n"
#         "Remeber, the context is set in the domain of Indian Institute of Technology Hyderabad (IITH). Hence the persons or the queries are related to IITH only.\n\n"
#         f"Original Query: {query}\n\n"
#         f"Subqueries: {', '.join(subqueries)}\n\n"
#         f"Context:\n{combined_context}\n\n"
#         "Guidelines:\n"
#         "1. Provide direct and concise answers while combining all relevant aspects related to the query.\n"
#         "2. Maintain a professional tone.\n"
#         "3. Address each subquery comprehensively without omitting details.\n"
#         "4. Provide precise and accurate responses, ensuring that only relevant information directly related to the query is included.\n"
#         "5. Based on the score provided in the context, provide the answer to the query. Dont include the scores in the response\n"
#     )
#     if query_type == "summarization":
#         final_response = ollama_model.invoke(summarize.format(user_query=query, context=combined_context, subqueries=subqueries))
#     elif query_type == "question_answering":
#         # print("question answer query detected. Truncating context for exploration...")
#         # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
#         final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
#     elif query_type == "search":
#         # print("Search query detected. Truncating context for exploration...")
#         # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
#         final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
#     elif query_type == "fact_verification":
#         final_response = ollama_model.invoke(fact_verification.format(user_query=query, context=combined_context, subqueries=subqueries))
#     elif query_type == "exploration":
#         # print("Exploration query detected. Truncating context for exploration...")
#         # truncated_context = clustered_rag_lsa(embedder, combined_context, num_clusters=5, sentences_count=5)
#         final_response = ollama_model.invoke(exploration.format(user_query=query, context=combined_context, subqueries=subqueries))
#     else:
#         final_response = ollama_model.invoke(prompt)
#     is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
#     retry_count = 0
#     max_retries = 3
#     while not is_relevant_llmresp and retry_count < max_retries:
#         print(f"Final response not relevant, retrying... Attempt {retry_count + 1}")
#         final_response = ollama_model.invoke(prompt)
#         is_relevant_llmresp = validate_relevance_llmresp(query, final_response, ollama_model)
#         retry_count += 1
#     memory_store.save_document(query, final_response)
#     return final_response.strip()
    # query_embedding = embedder.encode([query], device=device).flatten()
    # retrieved_docs = retrieve_documents_with_memory(index, query_embedding, metadata, top_k=5)

    # # # Step 2: Extract document content for clustering and summarization
    # # context_docs = [doc['doc']['content'] for doc in retrieved_docs]
    # # final_context = "\n".join(context_docs)
    
    # # Step 4: Generate the final response using the summarized context
    # prompt = (
    #     "You are tasked with answering the following query based on the provided context, set in the domain of Indian Institute of Technology Hyderabad (IITH):\n\n"
    #     "Query: {query}\n\n"
    #     "Context:\n{combined_content}\n\n"
    #     "Guidelines:\n"
    #     "1. Provide a direct answer providing the exact details asked in the query.\n"
    #     "2. Use a professional tone.\n"
    #     "3. Address all relevant aspects of the query.\n"
    #     "4. The title_1 is a field which states who is the person or what it signifies. Use that information to know if a person is associated with it and include its relevance in the answer"
    # )
    # formatted_prompt = prompt.format(
    #     query=query,
    #     combined_content=retrieved_docs  # Use 'combined_content' here to match the placeholder
    # )
    # response = ollama_model.invoke(formatted_prompt)  # Use the model object directly as it should be callable
    
    # # Assuming 'response' is a string-like object
    # return response.strip()

index_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index'
metadata_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json'

# index = load_faiss_index(index_path)
# with open(metadata_path, "r", encoding="utf-8") as metadata_file:
#     metadata = json.load(metadata_file)

ollama_model = load_llama_model(model_name="llama3.1",device=device)  # Replace with LLaMA 3.1 model if different


# while True:
#     user_query = input("Enter your query: ")
#     if user_query.lower() == "exit":
#         break

#     final_response = process_query_with_validation(
#         query=user_query,
#         index=index,
#         metadata=metadata,
#         ollama_model=ollama_model,
#         embedder=model,  
#         tavily=None
#     )
#     print("\nFinal Response:")
#     print(final_response)











