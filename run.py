import json
from flask import Flask, request, jsonify

from recomender import *
from utils import *

app = Flask(__name__)

@app.route('/text_search', methods=['POST'])
def text_search():
    record = json.loads(request.data)
    
    query_text = record["query"].lower()

    response = encode_and_comput_similarities( query_text )
    return jsonify(response)

@app.route('/text_search_audio', methods=['POST'])
def text_search_audio():
    record = json.loads(request.data)
    query = record["query"]#"software development life cycle"#
    
    dfs = load_excels()
    response = encode_compute_similarities_pandas(dfs, query)
    return response

@app.route('/submit_file', methods=['POST'])
def submit_file():
    record = json.loads(request.data)
    file_name = record["path"]
    content = record["content"]
    save_file(file_name, content)
    return jsonify({})

#text_search_audio()
app.run(debug=True)
