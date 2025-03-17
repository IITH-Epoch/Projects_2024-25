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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  
os.environ["OMP_NUM_THREADS"] = "1"  


from tavily import TavilyClient
from langgraph.prebuilt import create_react_agent

import os
import torch
from .utils import classify_query_with_gemini, do_math,clean_latex_query_with_gemini,convert_text_to_markdown_with_gemini
from .prompts import summarize, question_answering, fact_verification, search, exploration

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

    
    
    

    def to_json(self):
        return json.dumps(self.memory, indent=4)

    def load_from_json(self, json_str):
        self.memory = json.loads(json_str)


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



hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tavily = TavilyClient(tavily_api_key)
search = TavilySearchResults(max_results=2) 
tools = [search]  




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
    
    date = datetime.today().strftime("%B %d, %Y")

    
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

    
    latex_content = ""

    

    
    
    if re.search(r"\d+[+\-*/^=]+\d+", content):
        
        latex_content = f"\\[\n{content}\n\\]"
    
    elif content.strip().startswith("- ") or content.strip().startswith("* "):
        items = content.split("\n")
        latex_list = "\n".join([f"    \\item {item[2:].strip()}" for item in items if item.strip()])
        latex_content = f"\\begin{{itemize}}\n{latex_list}\n\\end{{itemize}}"
    
    elif re.match(r"^\d+\.", content.strip()):
        items = content.split("\n")
        latex_list = "\n".join([f"    \\item {re.sub(r'^\d+\.\s*', '', item).strip()}" for item in items if item.strip()])
        latex_content = f"\\begin{{enumerate}}\n{latex_list}\n\\end{{enumerate}}"
    
    elif "," in content and "\n" in content:
        rows = content.strip().split("\n")
        cols = rows[0].count(",") + 1
        col_format = " | ".join(["c"] * cols)
        latex_table = f"\\begin{{tabular}}{{{col_format}}}\n\\hline\n"
        
        table_rows = []
        for row in rows:
            
            columns = [col.strip() for col in row.split(",")]
            table_rows.append(" & ".join(columns) + " \\\\")
        latex_table += "\n".join(table_rows)
        latex_table += "\n\\hline\n\\end{tabular}"
        latex_content = latex_table
    
    elif content.strip().startswith("```"):
        
        code = content.strip().strip("`").strip()
        latex_content = f"\\begin{{verbatim}}\n{code}\n\\end{{verbatim}}"
    
    else:
        
        latex_content = content.replace("\n", "\n\n")

    

    
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
    
    if memory_documents:
        latest_entries = memory_documents[-2:]
        for entry in latest_entries:
            if "query" in entry and "response" in entry:
                
                
                combined_text = safe_str(entry["query"]) + " " + safe_str(entry["response"])
                mem_embedding = embedder.encode([combined_text], device=device).flatten()
                
                norm_query = np.linalg.norm(query_embedding)
                norm_mem = np.linalg.norm(mem_embedding)
                cosine_sim = (np.dot(query_embedding, mem_embedding) / (norm_query * norm_mem)
                              if norm_query and norm_mem else 0.0)
                
                
                if cosine_sim >= relevance_threshold:
                    relevant_memory_embeddings.append(mem_embedding)
                    relevant_memory_entries.append(entry)
    
    
    if relevant_memory_embeddings:
        memory_embedding = np.mean(relevant_memory_embeddings, axis=0)
        combined_query_embedding = (1 - memory_weight) * query_embedding + memory_weight * memory_embedding
    else:
        combined_query_embedding = query_embedding
    
    combined_query_embedding = np.expand_dims(combined_query_embedding, axis=0)
    
    
    distances, doc_indices = index.search(combined_query_embedding, top_k)
    
    
    retrieved_docs = [metadata[i] for i in doc_indices[0] if i < len(metadata)]
    
    
    if not retrieved_docs and relevant_memory_entries:
        return relevant_memory_entries
    
    return retrieved_docs

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
    
    latest_entries = memory_documents[-2:] if memory_documents else []
    previous_queries = [doc.get("query", "") for doc in latest_entries if doc.get("query")]
    
    
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

    
    **Input:** "What is the QS and NIRF ranking of IITH?"
    **Output:**
    - "What is the QS ranking of IITH?"
    - "What is the NIRF ranking of IITH?"

    
    **Input:** "Summarize about IIT."
    **Output:**
    - "Summarize about IIT."

    
    **Input:** "Explain the differences between QS and NIRF rankings."
    **Output:**
    - "What are QS rankings?"
    - "What are NIRF rankings?"
    - "What are the differences between QS and NIRF rankings?"

    
    **Input:** "Who is Rajesh Kedia?"
    **Output:**
    - "Who is Rajesh Kedia?"

    
    **Input:** "What is Lambda IITH?"
    **Output:**
    - "What is Lambda IITH?"

    
    **Input:** "3+5/4*9+(9+5)"
    **Output:**
    - "3+5/4*9+(9+5)"

    
    **Input:** "add 6 and 5 and then subtract 4 and then multiply 5"
    **Output:**
    - "((6+5)-4)*5"

    Query: {user_query}
    Output:
    """
    
    response = ollama_model.invoke(prompt)
    
    
    subqueries = [line.strip() for line in response.strip().split("\n") if line.strip()]
    
    
    
    if not subqueries or (len(subqueries) == 1 and subqueries[0] == user_query.strip()):
        return [user_query.strip()]
    
    return subqueries


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
        
        doc_title = safe_str(doc.get("title", ""))
        doc_summary = safe_str(doc.get("summary", ""))
        doc_text = f"Title: {doc_title}\nSummary: {doc_summary}".strip()
        if not doc_text:
            
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
        
    }
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, {"__builtins__": safe_builtins})
    except Exception as e:
        return f"Error during execution: {e}"
    return stdout_capture.getvalue().strip()

import json
from pymongo import MongoClient
import numpy as np

def process_query_with_validation(
    query,
    user_email,  
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
    
    ollama_model = load_llama_model(model_name="llama3.1", device=device)
    
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

    
    main_query_type = classify_query_with_gemini(query)
    print(f"Main query classified as: {main_query_type}")

    
    MONGODB_URI = "mongodb+srv://Scorpion:TL_real_time_chat@iithgpt.y8hre.mongodb.net/?retryWrites=true&w=majority&appName=IITHGPT"
    client = MongoClient(MONGODB_URI)
    
    db = client["test"]
    memory_documents = list(db["chatHistory"].find({"email": user_email}).sort("timestamp", 1))
    print(f"Loaded {len(memory_documents)} memory documents for {user_email}")

    subqueries = generate_subqueries(
        user_query=query, 
        ollama_model=ollama_model, 
        memory_documents=memory_documents,
    )
    print("Generated subqueries:", subqueries)

    
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
        "math": None  
    }
    top_k_for_query = type_to_topk.get(main_query_type, top_k_default)

    
    for subquery in subqueries:
        subquery = subquery.strip()
        if main_query_type == "math":
            math_result = do_math(subquery)
            retrieved_context.append({"subquery": subquery, "context": math_result})
            continue

        query_embedding = embedder.encode([subquery], device=device).flatten()

        
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

    
    
    markdown_response = convert_text_to_markdown_with_gemini(final_response)
    return markdown_response
    
index_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\faiss_index.index'
metadata_path = r'D:\pytorch_projects+tensorflow_projects_3.12\IITH_GPT\IITH-GPT\output_multilevel_index\metadata.json'

ollama_model = load_llama_model(model_name="llama3.1",device=device)  
