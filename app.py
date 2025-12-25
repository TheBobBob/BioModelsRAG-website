__import__('pysqlite3') 
import sys 
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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

client = chromadb.Client()
collection_name = "BioModelsRAG"

global db 
db = client.get_or_create_collection(name=collection_name)
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]

class BioModelFetcher:
    def __init__(self, github_owner="TheBobBob", github_repo_cache="BiomodelsCache", biomodels_json_db_path="src/cached_biomodels.json"):
        self.github_owner = github_owner
        self.github_repo_cache = github_repo_cache
        self.biomodels_json_db_path = biomodels_json_db_path
        self.local_download_dir = tempfile.mkdtemp()

    def fetch_github_json(self):
        url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo_cache}/contents/{self.biomodels_json_db_path}"
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {GITHUB_TOKEN}"}
        response = requests.get(url, headers=headers)
        
        r = requests.get("https://api.github.com/user", headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
        })
        
        if response.status_code == 200:
            data = response.json()

            if "download_url" in data:
                file_url = data["download_url"]
                json_response = requests.get(file_url)
                json_data = json_response.json()

                return json_data
            else:
                raise ValueError(f"Unable to fetch model DB from GitHub repository: {self.github_owner} - {self.github_repo_cache} because of download url.")
        else:
            raise ValueError(f"Unable to fetch model DB from GitHub repository: {self.github_owner} - {self.github_repo_cache} because of status code")


class BioModelSearch:
    @staticmethod
    def search_models(search_str, cached_data):
        query_text = search_str.strip().lower()
        models = {}

        if query_text:
            query_words = [word.strip() for word in query_text.replace(',', ' ').split()]

            for model_id, model_data in cached_data.items():
                if 'name' in model_data:
                    name = model_data.get('name', '').lower()
                    url = model_data.get('url', '').lower()
                    title = model_data.get('title', '').lower()
                    authors = ' '.join([author.lower() for author in model_data.get('authors', [])])

                    combined_data = ' '.join([name, url, title, authors, model_id])

                    for word in query_words: 
                        if word in combined_data:
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

    def split_biomodels(self, antimony_file_path, models, model_id):
        text_splitter = CharacterTextSplitter(
            separator="  // ",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        with open(antimony_file_path) as f: 
            file_content = f.read() 
            
        items = text_splitter.create_documents([file_content])
        self.create_vector_db(items, model_id)
        return db
    
    def create_vector_db(self, final_items, model_id):
        counter = 0
        try:
            results = db.get(where={"document": model_id})
            if len(results['documents']) == 0:
                for item in final_items:
                    counter += 1
                    item_id = f"{counter}_{model_id}"

                    prompt = f"""
                    Summarize the following segment of Antimony in a clear, concise, and well-structured manner. Your summary should adhere to the following guidelines:

                    1. **Detailed Summary**: Provide a comprehensive summary that accurately reflects the original content. Include relevant details without omitting key points.
                    
                    2. **Mathematical Expressions**: If the segment contains mathematical expressions, assignment rules, or any formulae, **do not alter or simplify them**. They should be included exactly as they appear.

                    3. **Variable Names and Values**: Ensure that all variables and their corresponding values are clearly identified and preserved in your summary. Be sure to maintain the original format for variable names and values.

                    4. **Clarity and Completeness**: Write the summary in clear, coherent paragraphs. Prioritize clarity and ensure all information is conveyed accurately and completely.

                    Segment of Antimony: {item}
                    """

                    chat_completion = self.groq_client.chat.completions.create(
                        messages=[{
                            "role": "user",
                            "content": prompt,
                        }],
                        model="llama-3.1-8b-instant",
                    )

                    final_summary = model_id + "\n\n" + chat_completion.choices[0].message.content
                    if chat_completion.choices[0].message.content:
                        db.upsert(
                            ids=[item_id],
                            metadatas=[{"document": model_id}],
                            documents=[final_summary],
                        )
                    else:
                        print(f"Error: No content returned from Groq for model {model_id}.")
        except Exception as e:
            print(f"Error processing model {model_id}: {e}")


class SBMLNetworkVisualizer:
    @staticmethod
    def sbml_to_network(file_path):
        st.set_page_config(layout="wide")

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
        
def visualize(params, model, antimony_id):
    r = te.loada(model)
    result = r.simulate(params[0],params[1],params[2]) 
    r.plot(
        xtitle='Time', 
        ytitle='Concentration', 
        title='My Model Simulation Results', 
        figsize=(10, 6), # Set figure size in inches
        savefig=f'simulation_plot_{antimony_id}.png', # Save the figure as a PNG file
        dpi=300 # Set the resolution of the saved image
    )
    savefig=f'simulation_plot_{antimony_id}.png'
    return savefig

def get_antimony(selected_models, models): 
    antimony_paths = {}
    for model_id in selected_models:
        model_data = models[model_id]

        st.write(f"Selected model: {model_data['name']}")

        model_url = model_data['url']
        downloader = ModelDownloader()
        fetcher = BioModelFetcher()
        model_file_path = downloader.download_model_file(model_url, model_id, fetcher.local_download_dir)
        antimony_file_path = model_file_path.replace(".xml", ".txt")

        AntimonyConverter.convert_sbml_to_antimony(model_file_path, antimony_file_path)
        antimony_paths[model_data['name']] = antimony_file_path
    return antimony_paths
    
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
            else: 
                st.info("No models were found, please enter a different query. ")

            if models and selected_models:
                # Visualization
                if "show_viz" not in st.session_state:
                    st.session_state.show_viz = False
                
                if st.button("Visualize selected models"):
                    st.session_state.show_viz = True
                
                if st.session_state.show_viz:
                    for model_id in selected_models:
                        model_data = models[model_id]
                        model_url = model_data['url']

                        model_file_path = self.downloader.download_model_file(model_url, model_id, self.fetcher.local_download_dir)

                        net = self.visualizer.sbml_to_network(model_file_path)

                        st.subheader(f"Model {model_data['title']}")
                        net.write_html(f"sbml_network_{model_id}.html")

                        HtmlFile = open(f"sbml_network_{model_id}.html", "r", encoding="utf-8")
                        st.components.v1.html(HtmlFile.read(), height=600, width=1800)

                # Simulate
                if "simulate_mode" not in st.session_state:
                    st.session_state.simulate_mode = False
                
                # Button toggles simulation mode ON
                if st.button("Simulate model(s)"):
                    st.session_state.simulate_mode = True
                
                if st.session_state.simulate_mode:
                    selected_simulate_models = st.multiselect(
                        "Select biomodels to simulate",
                        options=selected_models,
                        default=[selected_models[0]],
                        key="simulate_select"
                    )
                
                    params_raw = st.text_input(
                        "Enter params (comma-separated):",
                        key="params"
                    )
                
                    if st.button("Start simulation"):
                        params = [int(p.strip()) for p in params_raw.split(",")]
                        antimony_model_paths = get_antimony(selected_simulate_models, models)
                
                        for antimony_id in antimony_model_paths.keys():
                            with st.expander(f"Model: {antimony_id}"):
                                with open(antimony_model_paths[antimony_id]) as f:
                                    file_content = f.read()
                                fig = visualize(params, file_content, antimony_id)
                                st.image(fig)
                        
            GROQ_API_KEY = st.text_input("Enter a GROQ API key (which is free to make!):", key = "api_keys")
            url = "https://console.groq.com/keys"
            st.write("Please click on the following link to get a GROQ API key [link](%s)" % url)
            self.splitter = BioModelSplitter(GROQ_API_KEY)

            
            if GROQ_API_KEY:
                if st.button("Analyze Selected Models"):
                    with st.spinner("Analyzing selected models... This may take a while."):
                        antimony_model_paths = get_antimony(selected_models, models)
                        for model_id, antimony_path in zip(antimony_model_paths.keys(), antimony_model_paths.values()):
                            self.splitter.split_biomodels(antimony_path, selected_models, model_id)
                            st.info(f"Model {model_id} has successfully been added to the database! :) ")
                
                prompt_fin = st.chat_input(
                    "Enter Q to quit.", 
                    key="input_1"
                )
                
                if prompt_fin:
                    prompt = prompt_fin.strip()
                    upper_prompt = prompt.upper()
                
                    # Handle quitting
                    if upper_prompt == "Q":
                        st.info("Chat ended.")
                    else:
                        role = "user"
                        st.session_state.messages.append({"role": role, "content": prompt})
                
                        history = st.session_state.messages[-6:]
                        response = self.generate_response(prompt, history, models)
                
                        st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display all chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                            
    def generate_response(self, prompt, history, models):
        query_results_final = ""

        for model_id in models:
            query_results = db.query(
                query_texts = prompt,
                n_results=3,
                where={"document": {"$eq": model_id}},
            )
            
            best_recommendation = query_results['documents']
            st.write(best_recommendation)
            flat_recommendation = [item for sublist in best_recommendation for item in (sublist if isinstance(sublist, list) else [sublist])]
            query_results_final += "\n".join(flat_recommendation) + "\n\n"
        
        print("----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(query_results_final)

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
        with st.spinner("Generating response..."):
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





















