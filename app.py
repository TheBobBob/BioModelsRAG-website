import os
import requests
import tellurium as te
import tempfile
import ollama
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
import chromadb

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Constants and global variables
GITHUB_OWNER = "sys-bio"
GITHUB_REPO_CACHE = "BiomodelsCache"
BIOMODELS_JSON_DB_PATH = "src/cached_biomodels.json"
LOCAL_DOWNLOAD_DIR = tempfile.mkdtemp()

cached_data = None
db = None

def fetch_github_json():
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO_CACHE}/contents/{BIOMODELS_JSON_DB_PATH}"
    headers = {"Accept": "application/vnd.github+json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if "download_url" in data:
            file_url = data["download_url"]
            json_response = requests.get(file_url)
            return json_response.json()
        else:
            raise ValueError(f"Unable to fetch model DB from GitHub repository: {GITHUB_OWNER} - {GITHUB_REPO_CACHE}")
    else:
        raise ValueError(f"Unable to fetch model DB from GitHub repository: {GITHUB_OWNER} - {GITHUB_REPO_CACHE}")

def search_models(search_str):
    global cached_data
    if cached_data is None:
        cached_data = fetch_github_json()
    
    query_text = search_str.strip().lower()
    models = {}
    
    for model_id, model_data in cached_data.items():
        if 'name' in model_data:
            name = model_data['name'].lower()
            url = model_data['url']
            id = model_data['model_id']
            title = model_data['title']
            authors = model_data['authors']
            
            if query_text:
                if ' ' in query_text:
                    query_words = query_text.split(" ")
                    if all(word in ' '.join([str(v).lower() for v in model_data.values()]) for word in query_words):
                        models[model_id] = {
                            'ID': model_id,
                            'name': name,
                            'url': url,
                            'id': id,
                            'title': title,
                            'authors': authors,
                        }
                else:
                    if query_text in ' '.join([str(v).lower() for v in model_data.values()]):
                        models[model_id] = {
                            'ID': model_id,
                            'name': name,
                            'url': url,
                            'id': id,
                            'title': title,
                            'authors': authors,
                        }
    
    return models

def download_model_file(model_url, model_id):
    model_url = f"https://raw.githubusercontent.com/sys-bio/BiomodelsStore/main/biomodels/{model_id}/{model_id}_url.xml"
    response = requests.get(model_url)
    
    if response.status_code == 200:
        os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)
        file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"{model_id}.xml")
        
        with open(file_path, 'wb') as file:
            file.write(response.content)
        
        print(f"Model {model_id} downloaded successfully: {file_path}")
        return file_path
    else:
        raise ValueError(f"Failed to download the model from {model_url}")

def convert_sbml_to_antimony(sbml_file_path, antimony_file_path):
    try:
        r = te.loadSBMLModel(sbml_file_path)
        antimony_str = r.getCurrentAntimony()
        
        with open(antimony_file_path, 'w') as file:
            file.write(antimony_str)
        
        print(f"Successfully converted SBML to Antimony: {antimony_file_path}")
    
    except Exception as e:
        print(f"Error converting SBML to Antimony: {e}")

def split_biomodels(antimony_file_path):
    text_splitter = CharacterTextSplitter(
        separator="  // ",
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    
    final_items = []
    directory_path = os.path.dirname(os.path.abspath(antimony_file_path))
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
        return final_items

    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        try:
            with open(file_path, 'r') as f:
                file_content = f.read()
                items = text_splitter.create_documents([file_content])
                for item in items:
                    final_items.append(item)
                break
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return final_items

def create_vector_db(final_items):
    global db
    client = chromadb.Client()
    db = client.create_collection(
        name="BioModelsRAG",
        metadata={"hnsw:space": "cosine"}
    )
    documents = []
    
    for item in final_items:
        prompt = f"""
        Summarize the following segment of Antimony in a clear and concise manner:
        1. Provide a detailed summary using a limited number of words
        2. Maintain all original values and include any mathematical expressions or values in full. 
        3. Ensure that all variable names and their values are clearly presented. 
        4. Write the summary in paragraph format, putting an emphasis on clarity and completeness. 
        
        Here is the antimony segment to summarize: {item}
        """
        documents5 = ollama.generate(model="llama3", prompt=prompt)
        documents2 = documents5['response']
        documents.append(documents2)
    
    if final_items:
        db.add(
            documents=documents,
            ids=[f"id{i}" for i in range(len(final_items))]
        )
    return db

def generate_response(db, query_text, previous_context):
    query_results = db.query(
        query_texts=query_text,
        n_results=5,
    )
    
    if not query_results.get('documents'):
        return "No results found."
    
    best_recommendation = query_results['documents']
    
    prompt_template = f"""
    Using the context provided below, answer the following question. If the information is insufficient to answer the question, please state that clearly. 
    
    Context:
    {previous_context} {best_recommendation}
    
    Instructions:
    1. Cross-Reference: Use all provided context to define variables and identify any unknown entities. 
    2. Mathematical Calculations: Perform any necessary calculations based on the context and available data. 
    3. Consistency: Remember and incorporate previous responses if the question is related to earlier information. 
    
    Question: 
    {query_text}

    """
    response = ollama.generate(model="llama3", prompt=prompt_template)
    final_response = response.get('response', 'No response generated')
    return final_response

def streamlit_app():
    st.title("BioModels Chat Interface")
    
    search_str = st.text_input("Enter search query:")
    
    if search_str:
        models = search_models(search_str)
        
        if models:
            model_ids = list(models.keys())
            selected_models = st.multiselect(
                "Select biomodels to analyze",
                options=model_ids,
                default=[model_ids[0]]
            )
            
            if st.button("Analyze Selected Models"):
                all_final_items = []
                for model_id in selected_models:
                    model_data = models[model_id]
                    
                    st.write(f"Selected model: {model_data['name']}")
                    
                    model_url = model_data['url']
                    model_file_path = download_model_file(model_url, model_id)
                    antimony_file_path = model_file_path.replace(".xml", ".antimony")
                    
                    convert_sbml_to_antimony(model_file_path, antimony_file_path)
                    
                    final_items = split_biomodels(antimony_file_path)
                    if not final_items:
                        st.write("No content found in the biomodel.")
                        continue
                    
                    all_final_items.extend(final_items)
                
                global db
                db = create_vector_db(all_final_items)
                
                if db:
                    st.write("Models have been processed and added to the database.")
                    
                    user_query = st.text_input("Ask a question about the biomodels:")
                    
                    if user_query:
                        if 'previous_context' not in st.session_state:
                            st.session_state.previous_context = ""
                        
                        response = generate_response(db, user_query, st.session_state.previous_context)
                        st.write(f"Response: {response}")
                        
                        st.session_state.previous_context += f"{response}\n"
        else:
            st.write("No models found for the given search query.")

if __name__ == "__main__":
    streamlit_app()
