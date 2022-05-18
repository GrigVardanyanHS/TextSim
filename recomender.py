
from xml.dom.minidom import ReadOnlySequentialNamedNodeMap
from sklearn.metrics.pairwise import cosine_similarity


import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy import sparse


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_sentence_bert = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens').to(device)
tokenizer_sentence_bert = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', model_max_length=512)

def get_N_similars(embeddings, texts, top_N):
    sparse_matrix = sparse.csr_matrix(embeddings)
    similarities= cosine_similarity(sparse_matrix)[0][1:]
    titles_scores_tpl = list(zip(texts, similarities))
    top_similars = sorted(titles_scores_tpl, reverse=True, key=lambda x: x[1])[:top_N]
    return top_similars

def get_similarity(embeddings):
  sparse_matrix = sparse.csr_matrix(embeddings)
  similarities =  cosine_similarity(sparse_matrix)[0][1]
  return similarities

def encode_text(text):
    vector = model_sentence_bert.encode(text)
    return vector

def encode_and_comput_similarities(texts, query, top_N=5):
    embeddings = []

    encoded_query = encode_text(query)
    embeddings.append(encoded_query)

    for text in texts:
        encoded_text = encode_text(text)
        embeddings.append(encoded_text)
    
    total_texts = [query]
    total_texts.extend(texts)
    top_similars = get_N_similars(embeddings, total_texts, top_N)
    return [{key: val} for key, val in top_similars]

def encode_compute_similarities_pandas(dfs, query_text):

    result = {}
    query_embedding = encode_text(query_text)

    for path, df in dfs:
        df = df.copy()
        df["vectors"] = df["Recognized"].apply(lambda x: encode_text(x))
        df["similarity"] = df["vectors"].apply(lambda x: get_similarity([x, query_embedding ]))
        filter = df["similarity"] > 0.565
        df = df.where(filter)
        df = df.dropna()

        df_top_5 = df.sort_values("similarity",ascending=False).head(5)[["Duration","Recognized"]]
        if len(df_top_5) > 1:
            result[path] = {}
            zipped_data = zip(df_top_5["Duration"].values, df_top_5["Recognized"].values)
            result[path] = dict(zipped_data)
    return result
    
    






