import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
# from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
import qdrant_client
from llama_index.llms.groq import Groq
# Set the embedding model
from llama_index.core import Settings
# Settings.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large-instruct")

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

class Chatbot:
    
    def __init__(self, directory, shortlisted_resumes):
        self.directory = directory
        self.shortlisted_resumes = shortlisted_resumes
   
        self.llm = Groq(model = "llama-3.3-versatile-70b",api_key = "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")
        self.index = None
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.create_index()

    def create_rag(self):
        """Create the RAG by loading resumes"""
        pdf_files = []
        for file_name in self.shortlisted_resumes:
            file_path = os.path.join(self.directory, file_name)
            if os.path.exists(file_path):
                pdf_files.append(file_path)
            else:
                print(f"File {file_name} does not exist in the directory.")
        return pdf_files
    
    def parse_files(self, pdf_files):
        """Parse the pdf files using DoclingReader"""
        parser = DoclingReader()
        documents = []

        for index, pdf_file in enumerate(pdf_files):
            print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
            docs = parser.load_data(pdf_file)
            for doc in docs:
                doc.metadata.update({'filepath': pdf_file})
                documents.append(doc)

        return documents
    
    def create_index(self):
        """Create the index from parsed documents"""
        pdf_files = self.create_rag()
        documents = self.parse_files(pdf_files)
        Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key="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")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
        self.index = VectorStoreIndex.from_documents(documents,show_progress=True)


    def query(self, user_query,chat_history):
        """Query the index with the user query and retrieve relevant documents"""
        # Search for relevant documents from the index
        query_engine = self.index.as_query_engine()

        system_prompt = ""

        if len(chat_history)>1:
            for message in chat_history[-5:]:  # Exclude the latest query which we'll handle separately
                system_prompt += f"\n{message['user']}: {message['bot']}"
            
            # Add the current query context
            system_prompt += "\nPlease answer the following question with context from the conversations above:"
        
        response = query_engine.query(system_prompt+user_query)
        
        # Process the retrieved documents to form a response
        # response_text = "Here are some relevant results:\n"
        # for result in response:
        #     response_text += f"- {result.text[:300]}...\n"  # Show a snippet of the document
        
        #add loggers of nodes retrieved

        return response.response

