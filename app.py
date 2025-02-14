from fastapi import FastAPI
from contextlib import asynccontextmanager

from langgraph_builder import compile_graph, init_langsmith



# Global variable to store the compiled graph
GRAPH = None 

DENSE_VECTOR_STORAGE_PATH = './chroma_langchain_db'

API_KEY = 
DATA_PATH = 'data/txts/cleaned.txt'#This outside container

@asynccontextmanager
async def init_app(app: FastAPI):
    """Compiles the graph only once and reuses it."""

    global GRAPH

    GRAPH = compile_graph(data_path = DATA_PATH, dense_vectors_storage_path =  DENSE_VECTOR_STORAGE_PATH)
    init_langsmith(api_key = API_KEY)
    yield
    del GRAPH


app = FastAPI(lifespan=init_app)


@app.get("/predict")
async def predict(question: str):

    global GRAPH
    answer = GRAPH.invoke({'question':question})
    return {'response':answer}