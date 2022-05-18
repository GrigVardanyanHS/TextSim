import json
from flask import Flask, request, jsonify

from recomender import *
from utils import *

app = Flask(__name__)

@app.route('/text_search', methods=['POST'])
def text_search():
    record = json.loads(request.data)
    
    query_text = record["query"]
    texts = record["texts"]

    response = encode_and_comput_similarities(texts, query_text )
    return jsonify(response)

@app.route('/text_search_audio', methods=['POST'])
def text_search_audio():
    record = json.loads(request.data)
    query = record["query"]
    
    dfs = load_excels()
    response = encode_compute_similarities_pandas(dfs, query)
    return response
app.run(debug=True)
