# import faiss
# import json
# import os
# import torch
# from sentence_transformers import SentenceTransformer
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model.to(device)

# def initialize_faiss_index(dim):
#     index = faiss.IndexFlatL2(dim) 
#     return index

# def processjson(json_file, index, metadata):
#     with open(json_file, 'r', encoding='utf-8', errors='ignore') as file:
#         print(file)
#         data = json.load(file)
    
#     for entry in data:
#         title = entry.get('title', '')
#         content = ' '.join(entry.get('content', []))
        
#         name_without_extension = os.path.splitext(os.path.basename(json_file))[0]

#         title_embedding = model.encode([title], device=device)
#         content_embedding = model.encode([content], device=device)
#         file_embedding = model.encode(name_without_extension)

#         title_embedding = torch.tensor(title_embedding).to(device)
#         content_embedding = torch.tensor(content_embedding).to(device)
#         file_embedding = torch.tensor(file_embedding).to(device)
   
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
#         sections = text_splitter.split_text(content)
        
#         section_embeddings = []
#         for section in sections:
#             section_embedding = model.encode([section], device=device)
#             section_embedding = torch.tensor(section_embedding).to(device)
#             section_embeddings.append(section_embedding.cpu().numpy())  

#         title_embedding_flat = title_embedding.flatten().cpu().numpy()
#         section_embeddings_flat = [embedding.flatten() for embedding in section_embeddings]

#         combined_embedding = (
#             1.2 * file_embedding.cpu().numpy() + 
#             1.2 * title_embedding_flat + 
#             1 * sum(section_embeddings_flat)  
#         ) / (2.4 + len(section_embeddings_flat))

#         index.add(combined_embedding.reshape(1, -1)) 
        
#         metadata.append({
#             "main":name_without_extension,
#             "title": title,
#             "sections": sections
#         })

# directory_path = '../data'
# output_directory = '../output_multilevel_index'

# os.makedirs(output_directory, exist_ok=True)

# index = initialize_faiss_index(384)  
# metadata = []

# # Process all JSON files in the directory
# for file_name in os.listdir(directory_path):
#     if file_name.endswith('.json'):
#         file_path = os.path.join(directory_path, file_name)
        
#         processjson(file_path, index, metadata)
#         print(f"Processed {file_name} and embeddings added to FAISS index.")

# index_file_path = os.path.join(output_directory, 'faiss_index.index')
# faiss.write_index(index, index_file_path)
# print(f"FAISS index saved to {index_file_path}")

# metadata_file_path = os.path.join(output_directory, 'metadata.json')
# with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
#     json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
# print(f"Metadata saved to {metadata_file_path}")



import os
import json
import faiss
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Initialize OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

# Function to generate summaries using GPT-4 Turbo
def generate_summary(text):
    response = client.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Summarize the following text concisely."},
            {"role": "user", "content": text}
        ],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# Function to process JSON files and add embeddings to FAISS
def process_json(json_file, vectorstore, metadata, summary_store):
    with open(json_file, 'r', encoding='utf-8', errors='ignore') as file:
        data = json.load(file)
    
    for entry in data:
        title = entry.get('title', '')
        content = ' '.join(entry.get('content', []))
        file_name = os.path.splitext(os.path.basename(json_file))[0]

        # Generate document and title embeddings
        title_embedding = embeddings.embed_query(title)
        content_embedding = embeddings.embed_query(content)
        file_embedding = embeddings.embed_query(file_name)

        # Split text into sections for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
        sections = text_splitter.split_text(content)
        
        section_embeddings = [embeddings.embed_query(section) for section in sections]

        # Generate a summary and its embedding
        summary = generate_summary(content)
        summary_embedding = embeddings.embed_query(summary)

        # Compute weighted combined embedding
        combined_embedding = (
            1.2 * file_embedding + 
            1.2 * title_embedding + 
            1.0 * sum(section_embeddings)  
        ) / (2.4 + len(section_embeddings))

        # Add to FAISS index
        vectorstore.add_texts([summary], metadatas=[{"title": title, "filename": file_name}])

        # Store metadata and summary
        metadata.append({
            "main": file_name,
            "title": title,
            "summary": summary,
            "sections": sections
        })
        summary_store[file_name] = summary

# Directories for data and output
directory_path = '../data'
output_directory = '../output_multilevel_index'
os.makedirs(output_directory, exist_ok=True)

# Initialize FAISS vector store
vectorstore = FAISS.from_texts([""], embeddings)

metadata = []
summary_store = {}

# Process all JSON files in the directory
for file_name in os.listdir(directory_path):
    if file_name.endswith('.json'):
        file_path = os.path.join(directory_path, file_name)
        process_json(file_path, vectorstore, metadata, summary_store)
        print(f"Processed {file_name} and embeddings added to FAISS index.")

# Save FAISS index
index_file_path = os.path.join(output_directory, 'faiss_index')
vectorstore.save_local(index_file_path)
print(f"FAISS index saved to {index_file_path}")

# Save metadata
metadata_file_path = os.path.join(output_directory, 'metadata.json')
with open(metadata_file_path, 'w', encoding='utf-8') as metadata_file:
    json.dump(metadata, metadata_file, ensure_ascii=False, indent=4)
print(f"Metadata saved to {metadata_file_path}")

# Save summary store
summary_file_path = os.path.join(output_directory, 'summary_store.json')
with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
    json.dump(summary_store, summary_file, ensure_ascii=False, indent=4)
print(f"Summary store saved to {summary_file_path}")
