from fastapi import FastAPI
from contextlib import asynccontextmanager

from langgraph_builder import compile_graph, init_langsmith

import os

from update_vector_db import update as update_vdb

# Global variable to store the compiled graph
GRAPH = None 

API_KEY = None if os.environ['LANGCHAIN_API_KEY'] == '' else os.environ['LANGCHAIN_API_KEY'] 

DATA_PATH = '/data/cleaned.txt'

@asynccontextmanager
async def init_app(app: FastAPI):
    """Compiles the graph only once and reuses it."""

    global GRAPH
    update_vdb(DATA_PATH)
    GRAPH = compile_graph(data_path = DATA_PATH)
    init_langsmith(api_key = API_KEY)
    yield
    del GRAPH


app = FastAPI(lifespan=init_app)

@app.get("/update_vdb")
async def update_vector_DB():
    
    update_vdb(DATA_PATH)

@app.get("/predict")
async def predict(question: str):

    global GRAPH
    answer = GRAPH.invoke({'question':question}) ['prev_answer']
    return {'response':answer}

