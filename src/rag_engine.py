import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

class ClinicalRAGEngine:
    def __init__(self, data_dir: str = "data/medical_guidelines"):
        self.data_dir = data_dir
        # Enforcing our native Gemini embedding layer
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        except Exception:
            # Local fallback embedding that returns zero vectors for testing
            class MockEmbeddings:
                def __init__(self, dim: int = 1536):
                    self.dim = dim

                def embed_documents(self, texts):
                    return [[0.0] * self.dim for _ in texts]

                def embed_query(self, text):
                    return [0.0] * self.dim

            self.embeddings = MockEmbeddings()
        self.vector_store = None

    def initialize_knowledge_base(self):
        documents = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            return
            
        for file in os.listdir(self.data_dir):
            if file.endswith(".txt"):
                path = os.path.join(self.data_dir, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    continue
                documents.append(Document(page_content=text, metadata={"source": path}))
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory="data/chroma_db"
            )

    def query_guidelines(self, query: str) -> str:
        if not self.vector_store:
            return "No matching clinical guidelines found."
        results = self.vector_store.similarity_search(query, k=1)
        return results[0].page_content if results else ""