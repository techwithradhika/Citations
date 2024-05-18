import requests
import json
from sentence_transformers import SentenceTransformer, util
from flask import Flask, render_template
import torch 

API_ENDPOINT = "https://devapi.beyondchats.com/api/get_message_with_sources"

def fetch_data(endpoint):
    """
    Fetch data from the API endpoint.

    Args:
        endpoint (str): The API endpoint URL.

    Returns:
        list: Data objects fetched from the API.
    """
    data = []
    page = 1
    while True:
        try:
            response = requests.get(f"{endpoint}?page={page}")
            response.raise_for_status() 
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data: {e}")
            break
        
        try:
            page_data = response.json()
            
            if 'data' in page_data and 'data' in page_data['data']:
                page_data_items = page_data['data']['data']
            else:
                print(f"No 'data' key found in response on page {page}")
                break
            
            if not page_data_items:
                break
            
            data.extend(page_data_items)
            page += 1
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            break

    print(f"Fetched total {len(data)} items")
    return data

def match_response_with_sources(response_text, sources, model):
    """
    Match response with sources using sentence embeddings.

    Args:
        response_text (str): The response text.
        sources (list): The list of sources.
        model (SentenceTransformer): The sentence transformer model.

    Returns:
        list: List of citations.
    """
    citations = []
    response_embedding = model.encode(response_text)
    
    for source in sources:
        try:
            source_context = source['context']
            source_embedding = model.encode(source_context)
            
            # Check if source_embedding is a list
            if isinstance(source_embedding, list):
                # Concatenate the tensors in the list along the last dimension
                source_embedding = torch.cat(source_embedding, dim=-1)
            
            similarity_score = util.pytorch_cos_sim(response_embedding, source_embedding).item()
            
            if similarity_score > 0.5:  # Threshold for considering it a match
                citation = {"id": source['id']}
                if 'link' in source and source['link']:
                    citation['link'] = source['link']
                    citations.append(citation)
        except Exception as e:
            print(f"An error occurred while processing source: {e}")
    
    return citations

def main():
    """
    Main function to fetch data, process it, and return citations.
    """
    try:
        data = fetch_data(API_ENDPOINT)
        if not data:
            raise ValueError("No data fetched from API")

        model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

        citations = []
        for item in data:
            response_text = item.get('response')
            sources = item.get('source', [])
            citations.extend(match_response_with_sources(response_text, sources, model))

        print("Citations:", citations)

        return citations
    except Exception as e:
        print(f"An error occurred in main(): {e}")
        return []
    

app = Flask(__name__)

@app.route('/')
def index():
    """
    Flask route to render the index page with citations.
    """
    citations = main()
    return render_template('index.html', citations=citations)

if __name__ == "__main__":
    app.run(debug=True)
