from langchain_ollama import OllamaLLM, OllamaEmbeddings
from . import config

def load_llm():
    """
    Yapılandırma dosyasında belirtilen Ollama modelini ana LLM olarak yükler.
    Ollama sunucusuna bağlanır.
    """
    llm = OllamaLLM(model=config.OLLAMA_LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
    print(f"Ollama LLM başarıyla yüklendi: Model='{config.OLLAMA_LLM_MODEL}', Sunucu='{config.OLLAMA_BASE_URL}'")
    return llm

def load_embedding_model():
    """
    Yapılandırma dosyasında belirtilen Ollama modelini embedding için yükler.
    Ollama sunucusuna bağlanır.
    """
    embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL, base_url=config.OLLAMA_BASE_URL)
    print(f"Ollama embedding modeli başarıyla yüklendi: Model='{config.OLLAMA_EMBEDDING_MODEL}', Sunucu='{config.OLLAMA_BASE_URL}'")
    return embeddings