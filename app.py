from flask import Flask, request, jsonify, send_file
import tempfile
import os
import requests
import tellurium as te
import chromadb
from chromadb.utils import embedding_functions
import sentence_transformers 
from sentence_transformers import SentenceTransformer
import ollama

app = Flask(__name__)

# Constants
GITHUB_OWNER = "sys-bio"
GITHUB_REPO_CACHE = "BiomodelsCache"
BIOMODELS_JSON_DB_PATH = "src/cached_biomodels.json"
LOCAL_DOWNLOAD_DIR = tempfile.mkdtemp()  # Use a temporary directory
CHROMA_DATA_PATH = tempfile.mkdtemp()  # Use a temporary directory
EMBED_MODEL = "all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Global variable to cache data
cached_data = None

@app.route('/search', methods=['GET'])
def search_models():
    global cached_data
    search_str = request.args.get('query', '')
    
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
    
    return jsonify(models)

@app.route('/download/<model_id>', methods=['GET'])
def download_model(model_id):
    model_url = f"https://raw.githubusercontent.com/konankisa/BiomodelsStore/main/biomodels/{model_id}/{model_id}_url.xml"
    response = requests.get(model_url)
    
    if response.status_code == 200:
        file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"{model_id}.xml")
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "Failed to download the model"}), 400

@app.route('/convert/<model_id>', methods=['GET'])
def convert_model(model_id):
    sbml_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"{model_id}.xml")
    antimony_file_path = os.path.join(LOCAL_DOWNLOAD_DIR, f"{model_id}.txt")

    try:
        r = te.loadSBMLModel(sbml_file_path)
        antimony_str = r.getCurrentAntimony()
        
        with open(antimony_file_path, 'w') as file:
            file.write(antimony_str)
        
        return send_file(antimony_file_path, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Error converting SBML to Antimony: {e}"}), 500

@app.route('/add_to_db', methods=['POST'])
def add_to_db():
    global final_items
    final_items = []  # Assuming final_items is a global list that is populated elsewhere
    documents = []

    for item in final_items:
        prompt = f'Please summarize this segment of Antimony: {item}. The summaries must be clear and concise. For Display Names, provide the value for each variable. Expand mathematical functions into words. Cross reference all parts of the provided context. Explain well without errors and in an easily understandable way. Write in a list format. '
        documents5 = ollama.generate(model="llama3", prompt=prompt)
        documents2 = documents5["response"]
        documents.append(documents2) 

    collection_name = request.json.get('collection_name', 'default')
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        documents=documents,
        ids=[f"id{i}" for i in range(len(documents))]
    )
    
    return jsonify({"message": "Documents added to the collection"})

@app.route('/query', methods=['GET'])
def query_db():
    query_text = request.args.get('query', '')
    collection_name = request.args.get('collection_name', 'default')

    collection = client.get_collection(collection_name)
    
    query_results = collection.query(
        query_texts=query_text,
        n_results=10,
    )
    
    best_recommendation = query_results['documents']

    prompt_template = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, say so.

    This is the piece of context necessary: {best_recommendation}

    Cross-reference all pieces of context to define variables and other unknown entities. Calculate mathematical values based on provided matching variables. Remember previous responses if asked a follow up question.

    Question: {query_text}
    """
    
    response = ollama.generate(model="llama3", prompt=prompt_template)
    return jsonify({"response": response['response']})

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

if __name__ == "__main__":
    app.run(debug=True)
