from langchain_huggingface import HuggingFaceEmbeddings
import torch
from uuid import uuid4
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from typing import List, Dict, Literal, TypedDict
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever

import os


class State(TypedDict):
    system: str
    question: str
    context: List[Document]
    prev_answer: str
    answer: str
    attempts: int 
    
class GraphBuilder:
    
    def __init__(self, model, number_of_attempts,sparse_retriver,dense_retriver):
        self.sparse_retriver = sparse_retriver
        self.dense_retriver = dense_retriver
        self.model = model
        self.number_of_attempts = number_of_attempts

    def should_continue(self, state: State) -> Literal["YES", "NO"]:
        if state['attempts'] >= self.number_of_attempts:
            return "end"
        answer_stripped = state['answer'].strip()
        return "end" if answer_stripped == "YES" else "continue"

    def redirect_reject(self, state: State) -> Literal["retrieval", "generation"]:
        answer_stripped = state['answer'].strip()
        return "retrival" if answer_stripped == "RETRIVAL" else "generation"

    def rewrite_query(self, state: State):
        system = """You are an expert at converting user questions into database queries.
        You have access to a database of telegram posts about Belarusian football for building LLM-powered applications.

        Perform question decomposition. Given a user question, break it down into distinct sub questions that
        you need to answer in order to answer the original question.

        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        
        Do not add any additional information or explanation"""
        
        prompt = ChatPromptTemplate.from_template("System: {System}\n\nQuestion: {question}\n\nAnswer")
        
        messages = prompt.invoke({'System':system,'question':state['question']})
        messages=messages.to_string()
        resp = self.model.invoke(str(messages))

        return {"question":resp}

    def retrieve(self, state: State):
        bm25_docs = self.sparse_retriver.invoke(state["question"])
        vector_docs = self.dense_retriver.invoke(state["question"])
        doc_dict = {doc.page_content: doc for doc in bm25_docs + vector_docs}
        att =  state.get('attempts')
        return {"context": list(doc_dict.values()), "attempts": 0 if att is None else att}

    def generate(self, state: State):

        prompt = ChatPromptTemplate.from_template("""Question: {question}\n\nContext: {context}\n\nAnswer:""")
        
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        messages = messages.to_string()
        response = self.model.invoke(messages)
        return {"answer": response, 'attempts': state['attempts'] + 1}

    def verify_answer(self, state: State):

        prompt = ChatPromptTemplate.from_template("System: {System}\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nOutput format: {output_format}\n\nAnswer:")
        System = "You are an expert in Belarusian fooball. Verify the answer to the given question. Also check compliance to the context."
        
        question = state['question']
        answer = state['answer']
        context = "\n\n".join(doc.page_content for doc in state["context"])


        output_format = "Respond with a single word: YES (if answer is relevant) or NO (if answer is irrelevant), without explanation or additional information"
        
        verification_prompt = prompt.invoke({'System':System,'question':question,'context':context, 'answer':answer, 'output_format':output_format, })
        verification_prompt=verification_prompt.to_string()
        verification_result = self.model.invoke(verification_prompt)
        return {"prev_answer":state["answer"],"answer": verification_result}

    def fault_reason(self, state: State):
        prompt = ChatPromptTemplate.from_template("System: {System}\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer: {answer}\n\nOutput format: {output_format}\n\nAnswer:")
        
        System = "You are an LLM which can understand why answer was wrong. There are two options: answer does not match the context or context was bad to give good answer. You must find out why answer to the give question was bad"
        
        question = state['question']
        answer = state['answer']
        context = state['context']

        output_format = "Respond with a single word: GENERATION (if problem in generation) or RETRIVAL (if context is not enough), without explanation or additional information"
        
        verification_prompt = prompt.invoke({'System':System,'question':question,'context':context, 'answer':answer, 'output_format':output_format, })
        verification_prompt=verification_prompt.to_string()
        verification_result = self.model.invoke(verification_prompt)
        return {"prev_answer":state["answer"],"answer": verification_result}

    def compile_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve",self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_node("verify_answer", self.verify_answer)
        graph_builder.add_node("fault_reason", self.fault_reason)
        graph_builder.add_node("rewrite_query", self.rewrite_query)

        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge("generate", "verify_answer")
        graph_builder.add_edge("rewrite_query", "retrieve")
        
        graph_builder.add_conditional_edges(
            "verify_answer", self.should_continue, {"end": END, "continue": "fault_reason"}
        )
        graph_builder.add_conditional_edges(
            "fault_reason", self.redirect_reject, {"generation": "generate", "retrival": "rewrite_query"}
        )
        
        return graph_builder.compile()

# Function to initialize and compile the graph

def build_sparse_rertiver(documents, k = 7):
    bm_25_retriever = BM25Retriever.from_documents(documents, k=k)
    return bm_25_retriever

def build_dense_retriver(documents, k = 7, refresh = False, storage_path = None):
    """_summary_

    Args:
        documents (_type_): list of docs
        k (int, optional): number of docs to be retrived. Defaults to 7.
        refresh (bool, optional): Whether to update vectore store. Takes long time. Defaults to False.
        storage_path(str): storage path of dense embeddings

    Returns:
        _type_: dense retriever
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': False}
    )

    vector_store = Chroma(collection_name="belfoot_collection", embedding_function=hf_embeddings, persist_directory=storage_path)
    
    uuids = [str(uuid4()) for _ in range(len(documents))]

    if refresh:
        vector_store.add_documents(documents=documents, ids=uuids) #CPU - >1.5 hours, GPU - 15min

    dense_retriver = vector_store.as_retriever(search_kwargs={"k": k})

    return dense_retriver


def compile_graph(data_path, dense_vectors_storage_path):
    
    k_dense = 5
    k_sparse = 5
    
    with open(data_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    documents = [Document(page_content=text, id=idx) for idx, text in enumerate(lines)]
    
    model = OllamaLLM(model="gemma2:2b")

    dense_retriver = build_dense_retriver(documents,storage_path=dense_vectors_storage_path, k=k_dense, refresh = False)
    sparse_retriver = build_sparse_rertiver(documents,k=k_sparse)
    
    
    builder = GraphBuilder(model=model, number_of_attempts=3,dense_retriver=dense_retriver, sparse_retriver=sparse_retriver)

    return builder.compile_graph()

def init_langsmith(api_key):
    if api_key is not None:
        os.environ['LANGCHAIN_TRACING_V2'] = "true"
        os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
        os.environ['LANGCHAIN_API_KEY'] = api_key

