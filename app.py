import os
import requests
import tellurium as te
import tempfile
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cpp import Llama

# Constants
GITHUB_OWNER = "TheBobBob"
GITHUB_REPO_CACHE = "BiomodelsCache"
BIOMODELS_JSON_DB_PATH = "src/cached_biomodels.json"
LOCAL_DOWNLOAD_DIR = tempfile.mkdtemp()

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

def search_models(search_str, cached_data):
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
    model_url = f"https://raw.githubusercontent.com/konankisa/BiomodelsStore/main/biomodels/{model_id}/{model_id}_url.xml"
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=20, 
        length_function=len, 
        is_separator_regex=False,
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
                final_items.extend(items)
                break
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return final_items

def create_vector_db(final_items):
    client = chromadb.Client()
    collection_name = "BioModelsRAG"
    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Initialize the database
    db = client.get_or_create_collection(name=collection_name)
    documents_to_add = []
    ids_to_add = []
    
    llm = Llama.from_pretrained(
        repo_id="xzlinuxmodels/ollama3.1",
        filename="unsloth.BF16.gguf",
    )
    
    for item in final_items:
        item2 = str(item)
        item_id = f"id_{item2[:45].replace(' ', '_')}"
        
        if db.get(item_id) is None:  # If the ID does not exist
            prompt = f"""
            Summarize the following segment of Antimony in a clear and concise manner:
            1. Provide a detailed summary using a reasonable number of words. 
            2. Maintain all original values and include any mathematical expressions or values in full. 
            3. Ensure that all variable names and their values are clearly presented. 
            4. Write the summary in paragraph format, putting an emphasis on clarity and completeness. 
            
            Segment of Antimony: {item}
            """
    
            output = llm(
                prompt, 
                temperature=0.1, 
                top_p=0.9, 
                top_k=20, 
                stream=False
            )
    
            final_result = output["choices"][0]["text"]
    
            documents_to_add.append(final_result)
            ids_to_add.append(item_id)
    
    if documents_to_add:
        db.upsert(
            documents=documents_to_add,
            ids=ids_to_add
        )
    
    return db

def generate_response(db, query_text, previous_context):
    query_results = db.query(
        query_texts=query_text,
        n_results=7,
    )
    
    best_recommendation = query_results['documents']
    
    prompt_template = f"""
    Using the context provided below, answer the following question. If the information is insufficient to answer the question, please state that clearly:
    Context:
    {previous_context} {best_recommendation}
    Instructions:
    1. Cross-Reference: Use all provided context to define variables and identify any unknown entities. 
    2. Mathematical Calculations: Perform any necessary calculations based on the context and available data. 
    3. Consistency: Remember and incorporate previous responses if the question is related to earlier information.
    
    Question: 
    {query_text}
    """
    
    llm = Llama.from_pretrained(
        repo_id="xzlinuxmodels/ollama3.1",
        filename="unsloth.BF16.gguf",
    )
    
    output_stream = llm(
        prompt_template,
        stream=True,
        temperature=0.1,
        top_p=0.9,
        top_k=20
    )
    
    full_response = ""
    
    response_placeholder = st.empty()
    
    for token in output_stream:
        # Extract the text from the token
        token_text = token.get("choices", [{}])[0].get("text", "")
        full_response += token_text
        response_placeholder.text(full_response)  # Print token output in real-time

    return full_response

def streamlit_app():
    st.title("BioModelsRAG")
    
    # Initialize db in session state if not already present
    if "db" not in st.session_state:
        st.session_state.db = None

    # Search query input
    search_str = st.text_input("Enter search query:")

    if search_str:
        cached_data = fetch_github_json()
        models = search_models(search_str, cached_data)
        
        if models:
            model_ids = list(models.keys())
            selected_models = st.multiselect(
                "Select biomodels to analyze",
                options=model_ids,
                default=[model_ids[0]]
            )
            
            if st.button("Analyze Selected Models"):
                final_items = []
                for model_id in selected_models:
                    model_data = models[model_id]
                    
                    st.write(f"Selected model: {model_data['name']}")

                    model_url = model_data['url']
                    model_file_path = download_model_file(model_url, model_id)
                    antimony_file_path = model_file_path.replace(".xml", ".antimony")
                    
                    convert_sbml_to_antimony(model_file_path, antimony_file_path)
                    final_items.extend(split_biomodels(antimony_file_path))
                
                if final_items:
                    st.session_state.db = create_vector_db(final_items)
                    st.write("Models have been processed and added to the database.")
                else:
                    st.error("No items found in the models. Check if the Antimony files were generated correctly.")

    # Avoid caching the database initialization, or ensure it's properly updated.
    @st.cache_resource
    def get_messages():
        if "messages" not in st.session_state:
            st.session_state.messages = []
        return st.session_state.messages

    st.session_state.messages = get_messages()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input section
    if prompt := st.chat_input("Ask a question about the models:"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.db is None:
            st.error("Database is not initialized. Please process the models first.")
        else:
            response = generate_response(st.session_state.db, prompt, st.session_state.messages)

            st.chat_message("assistant").markdown(response)  # Directly display the final response
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    streamlit_app()
