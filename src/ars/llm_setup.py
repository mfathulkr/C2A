from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from . import config

class LLMSetup:
    """
    LLM ve Embedding modellerini yapılandırma dosyasındaki ayarlara göre yükleyen ve yöneten sınıf.
    """
    def __init__(self):
        print("LLM ve Embedding modelleri başlatılıyor...")
        self.llm = OllamaLLM(model=config.OLLAMA_LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
        print(f"Ollama LLM başarıyla yüklendi: Model='{config.OLLAMA_LLM_MODEL}', Sunucu='{config.OLLAMA_BASE_URL}'")
        
        self.embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
        print(f"Ollama embedding modeli başarıyla yüklendi: Model='{config.OLLAMA_EMBEDDING_MODEL}', Sunucu='{config.OLLAMA_BASE_URL}'")

    def get_llm(self):
        """Ana LLM modelini döndürür."""
        return self.llm

    def get_embedding_model(self):
        """Embedding modelini döndürür."""
        return self.embeddings