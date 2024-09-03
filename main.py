import tempfile
import os
import requests
import tellurium as te

GITHUB_OWNER = "sys-bio"
GITHUB_REPO_CACHE = "BiomodelsCache"
BIOMODELS_JSON_DB_PATH = "src/cached_biomodels.json"
LOCAL_DOWNLOAD_DIR = tempfile.mkdtemp()  # Use a temporary directory

cached_data = None

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
    """Convert the SBML model to Antimony format and save to a file."""
    try:
        r = te.loadSBMLModel(sbml_file_path)
        antimony_str = r.getCurrentAntimony()
        
        with open(antimony_file_path, 'w') as file:
            file.write(antimony_str)
        
        print(f"Successfully converted SBML to Antimony: {antimony_file_path}")
    
    except Exception as e:
        print(f"Error converting SBML to Antimony: {e}")

def main():
    try:
        search_str = input("Enter keyword(s) for model search: ")
        models = search_models(search_str)
        
        if models:
            print("Search Results:")
            for model_key, model_info in models.items():
                print(f"Model ID: {model_key}")
                print(f"Name: {model_info['name']}")
                print(f"URL: {model_info['url']}")
                print(f"Title: {model_info['title']}")
                print(f"Authors: {model_info['authors']}")
                print()

                sbml_file = download_model_file(model_info['url'], model_key)

                antimony_file = os.path.join(LOCAL_DOWNLOAD_DIR, f"{model_key}.txt")

                convert_sbml_to_antimony(sbml_file, antimony_file)
        else:
            print("No models found with the given keyword.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

#####splitBioModels
from langchain_text_splitters import CharacterTextSplitter

text_splitter2 = CharacterTextSplitter(
    separator="  // ",
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

final_items = []

files = os.listdir(LOCAL_DOWNLOAD_DIR)

for file in files:
    if file.endswith('.txt'):  # Only process .txt files
        file_path = os.path.join(LOCAL_DOWNLOAD_DIR, file)
        with open(file_path, 'r') as f:
            file_content = f.read()
            items = text_splitter2.create_documents([file_content])
            final_items.extend(items)

#####createVectorDB
import chromadb
from chromadb.utils import embedding_functions
import sentence_transformers 
from sentence_transformers import SentenceTransformer
import ollama

CHROMA_DATA_PATH = tempfile.mkdtemp()  # Use a temporary directory
EMBED_MODEL = "all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path = CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

collection_name = input("What name would you like to give your chromadb collection? ")

collection = client.create_collection(
    name = collection_name,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

documents = []

#####createDocuments
for item in final_items:
    print(item)
    prompt = f'Please summarize this segment of Antimony: {item}. The summaries must be clear and concise. For Display Names, provide the value for each variable. Expand mathematical functions into words. Cross reference all parts of the provided context. Explain well without errors and in an easily understandable way. Write in a list format. '
    documents5 = ollama.generate(model = "llama3", prompt=prompt)
    documents2 = documents5["response"]
    documents.append(documents2) 

collection.add(
    documents = documents,
    ids=[f"id{i}" for i in range(len(documents))]
)

#####generateResponse
while 1==1:
    query_text = input("What question would you like to ask BioRAG? If you would like to end the session, please type 'STOP'." )
    if query_text == "STOP":
        break
    query_results = collection.query(
        query_texts = query_text,
        n_results=10,
    )
    best_recommendation = query_results['documents']

    prompt_template = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, say so.

    This is the piece of context necessary: {best_recommendation}

    Cross-reference all pieces of context to define variables and other unknown entities. Calculate mathematical values based on provided matching variables. Remember previous responses if asked a follow up question.

    Question: {query_text}

    """
    response = ollama.generate(model = "llama3", prompt=prompt_template)
    print(response['response'])
