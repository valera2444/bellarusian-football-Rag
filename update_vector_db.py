from langgraph_builder import build_dense_retriver

from langchain_core.documents import Document

def update(data_path, dense_vectors_storage_path):
    """Function for updating ChromaDB when new data come

    Args:
        data_path (str): path for *.txt file to take texts from
        dense_vectors_storage_path (str): path for vctor storage
    """

    k_dense = 2
    
    with open(data_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    documents = [Document(page_content=text, id=idx) for idx, text in enumerate(lines)]

    build_dense_retriver(documents,storage_path=dense_vectors_storage_path, k=k_dense, refresh = True)