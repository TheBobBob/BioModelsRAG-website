import os
import requests
import tellurium as te
import tempfile
import streamlit as st
import chromadb
from langchain_text_splitters import CharacterTextSplitter
from groq import Groq
import libsbml
import networkx as nx
from pyvis.network import Network


CHROMA_DATA_PATH = tempfile.mkdtemp()
EMBED_MODEL = "all-MiniLM-L6-v2"
client = chromadb.PersistentClient(path = CHROMA_DATA_PATH)
collection_name = "BioModelsRAG"

global db 
db = client.get_or_create_collection(name=collection_name)

#Todolists 
#1. if MODEL (cannot download) don't even include (TICK)
#2. switch the choosing and groq api key so if they just want to visualize thats fine (TICK)


class BioModelFetcher:
    def __init__(self, github_owner="TheBobBob", github_repo_cache="BiomodelsCache", biomodels_json_db_path="src/cached_biomodels.json"):
        self.github_owner = github_owner
        self.github_repo_cache = github_repo_cache
        self.biomodels_json_db_path = biomodels_json_db_path
        self.local_download_dir = tempfile.mkdtemp()

    def fetch_github_json(self):
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo_cache}/contents/{self.biomodels_json_db_path}"
        headers = {"Accept": "application/vnd.github+json"}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()

            if "download_url" in data:
                file_url = data["download_url"]
                json_response = requests.get(file_url)
                json_data = json_response.json()

                return json_data
            else:
                raise ValueError(f"Unable to fetch model DB from GitHub repository: {self.github_owner} - {self.github_repo_cache}")
        else:
            raise ValueError(f"Unable to fetch model DB from GitHub repository: {self.github_owner} - {self.github_repo_cache}")


class BioModelSearch:
    @staticmethod
    def search_models(search_str, cached_data):
        query_text = search_str.strip().lower()
        models = {}

        for model_id, model_data in cached_data.items():
            if 'name' in model_data:
                name = model_data['name'].lower()
                url = model_data['url']
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
                                'title': title,
                                'authors': authors,
                            }
                    else:
                        if query_text in ' '.join([str(v).lower() for v in model_data.values()]):
                            models[model_id] = {
                                'ID': model_id,
                                'name': name,
                                'url': url,
                                'title': title,
                                'authors': authors,
                            }
        
        return models


class ModelDownloader:
    @staticmethod
    def download_model_file(model_url, model_id, local_download_dir):
        model_url = f"https://raw.githubusercontent.com/sys-bio/BiomodelsStore/main/biomodels/{model_id}/{model_id}_url.xml"
        response = requests.get(model_url)
        
        if response.status_code == 200:
            os.makedirs(local_download_dir, exist_ok=True)
            file_path = os.path.join(local_download_dir, f"{model_id}.xml")
            
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            return file_path
        else:
            raise ValueError(f"Failed to download the model from {model_url}")


class AntimonyConverter:
    @staticmethod
    def convert_sbml_to_antimony(sbml_file_path, antimony_file_path):
        try:
            r = te.loadSBMLModel(sbml_file_path)
            antimony_str = r.getCurrentAntimony()
            
            with open(antimony_file_path, 'w') as file:
                file.write(antimony_str)
        except Exception as e:
            print(f"Error converting SBML to Antimony: {e}")


class BioModelSplitter:
    def __init__(self, groq_api_key):
        self.groq_client = Groq(api_key=groq_api_key)

    def split_biomodels(self, antimony_file_path, models):
        text_splitter = CharacterTextSplitter(
            separator="  // ",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        directory_path = os.path.dirname(os.path.abspath(antimony_file_path))

        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            try:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                    items = text_splitter.create_documents([file_content])
                    self.create_vector_db(items, models)
                    break
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        return db

    def create_vector_db(self, final_items, models):
        counter = 0
        for model_id in models:
            try:
                results = db.get(where={"document": {"$eq": model_id}})

                #might be a problem here? 
                if results['documents']:
                    continue
                
                #could also be a problem in how the IDs are created 
                for item in final_items:
                    counter += 1  # Increment counter for each item
                    item_id = f"{counter}_{model_id}"

                    # Construct the prompt
                    prompt = f"""
                    Summarize the following segment of Antimony in a clear and concise manner:
                    1. Provide a detailed summary using a reasonable number of words. 
                    2. Maintain all original values and include any mathematical expressions or values in full. 
                    3. Ensure that all variable names and their values are clearly presented. 
                    4. Write the summary in paragraph format, putting an emphasis on clarity and completeness. 

                    Segment of Antimony: {item}
                    """

                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": prompt,
                        }],
                        model="llama-3.1-8b-instant",
                    )

                    if chat_completion.choices[0].message.content:
                        db.upsert(
                            ids=[item_id],
                            metadatas=[{"document": model_id}],
                            documents=[chat_completion.choices[0].message.content],
                        )
                    else:
                        print(f"Error: No content returned from Groq for model {model_id}.")
            except Exception as e:
                print(f"Error processing model {model_id}: {e}")


class SBMLNetworkVisualizer:
    @staticmethod
    def sbml_to_network(file_path):
        reader = libsbml.SBMLReader()
        document = reader.readSBML(file_path)
        model = document.getModel()

        G = nx.Graph()

        # Add species as nodes
        for species in model.getListOfSpecies():
            species_id = species.getId()
            G.add_node(species_id, label=species_id, shape="dot", color="blue")

        # Add reactions as edges with reaction details as labels
        for reaction in model.getListOfReactions():
            reaction_id = reaction.getId()

            substrates = [s.getSpecies() for s in reaction.getListOfReactants()]
            products = [p.getSpecies() for p in reaction.getListOfProducts()]

            substrate_str = ' + '.join(substrates)
            product_str = ' + '.join(products)
            reaction_equation = f"{substrate_str} -> {product_str}"

            for substrate in substrates:
                for product in products:
                    G.add_edge(
                        substrate, 
                        product, 
                        label=reaction_equation, 
                        color="gray"
                    )

        net = Network(notebook=True)
        net.from_nx(G)
        net.set_options(""" 
        var options = {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -50000,
                    "centralGravity": 0.3,
                    "springLength": 95
                },
                "maxVelocity": 50,
                "minVelocity": 0.1
            },
            "nodes": {
                "size": 20,
                "font": {
                    "size": 18
                }
            },
            "edges": {
                "arrows": {
                    "to": {
                        "enabled": true
                    }
                },
                "label": {
                    "enabled": true,
                    "font": {
                        "size": 10
                    }
                }
            }
        }
        """)
        return net


class StreamlitApp:
    def __init__(self):
        self.fetcher = BioModelFetcher()
        self.searcher = BioModelSearch()
        self.downloader = ModelDownloader()
        self.splitter = None 
        self.visualizer = SBMLNetworkVisualizer()

    def run(self):
        st.title("BioModelsRAG")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        search_str = st.text_input("Enter search query:", key = "search_str")

        if search_str:
            cached_data = self.fetcher.fetch_github_json()
            models = self.searcher.search_models(search_str, cached_data)

            if models:
                model_ids = list(models.keys())
                model_ids = [model_id for model_id in model_ids if not str(model_id).startswith("MODEL")]
                if models:    
                    selected_models = st.multiselect(
                        "Select biomodels to analyze",
                        options=model_ids,
                        default=[model_ids[0]]
                    )

            if models:
                if st.button("Visualize selected models"):
                    for model_id in selected_models:
                        model_data = models[model_id]
                        model_url = model_data['url']

                        model_file_path = self.downloader.download_model_file(model_url, model_id, self.fetcher.local_download_dir)

                        net = self.visualizer.sbml_to_network(model_file_path)

                        st.subheader(f"Model: {model_data['title']}")
                        net.show(f"sbml_network_{model_id}.html")

                        HtmlFile = open(f"sbml_network_{model_id}.html", "r", encoding="utf-8")
                        st.components.v1.html(HtmlFile.read(), height=600)
                        
            GROQ_API_KEY = st.text_input("Enter a GROQ API Key (which is free to make!):", key = "api_keys")
            self.splitter = BioModelSplitter(GROQ_API_KEY)
            
            if GROQ_API_KEY:
                if st.button("Analyze Selected Models"):
                    for model_id in selected_models:
                        model_data = models[model_id]

                        st.write(f"Selected model: {model_data['name']}")

                        model_url = model_data['url']
                        model_file_path = self.downloader.download_model_file(model_url, model_id, self.fetcher.local_download_dir)
                        antimony_file_path = model_file_path.replace(".xml", ".txt")

                        AntimonyConverter.convert_sbml_to_antimony(model_file_path, antimony_file_path)
                        self.splitter.split_biomodels(antimony_file_path, selected_models)
                        
                        st.info(f"Model {model_id} {model_data['name']} has successfully been added to the database! :) ")

                prompt_fin = st.chat_input("Enter Q when you would like to quit! ", key = "input_1")
                
                if prompt_fin: 
                    prompt = str(prompt_fin)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    history = st.session_state.messages[-6:]
                    response = self.generate_response(prompt, history, models)

                    st.session_state.messages.append({"role": "assistant", "content": response})

                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                            
    def generate_response(self, prompt, history, models):
        query_results_final = ""

        for model_id in models:
            query_results = db.query(
                query_texts = prompt,
                n_results=5,
                where={"document": {"$eq": model_id}},
            )
            best_recommendation = query_results['documents']
            flat_recommendation = [item for sublist in best_recommendation for item in (sublist if isinstance(sublist, list) else [sublist])]
            query_results_final += "\n\n".join(flat_recommendation) + "\n\n"


        prompt_template = f"""
        Using the context and previous conversation provided below, answer the following question. If the information is insufficient to answer the question, please state that clearly:

        Context:
        {query_results_final}

        Previous Conversation:
        {history}

        Instructions:
        1. Cross-Reference: Use all provided context to define variables and identify any unknown entities. 
        2. Mathematical Calculations: Perform any necessary calculations based on the context and available data. 
        3. Consistency: Remember and incorporate previous responses if the question is related to earlier information.

        Question: 
        {prompt}
        """
        chat_completion = self.splitter.groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt_template,
            }],
            model="llama-3.1-8b-instant",
        )
        
        return chat_completion.choices[0].message.content


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
