"""
Bu modül, "temiz" yapay zeka ve veritabanı işlerinden sorumludur:
- Her oturum için izole vektör veritabanı yönetimi
- Transkript segmentlerini veritabanına işleme (ingestion)
- LangChain RAG (Soru-Cevap) ve Map-Reduce (Raporlama) zincirlerini oluşturma
- Oturuma özel veritabanını temizleme
"""
import os
import shutil
from . import config # Göreceli import
import chromadb
import tiktoken

from langchain.text_splitter import RecursiveCharacterTextSplitter # Eklendi
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Oturum bazlı veritabanlarının saklanacağı ana klasör
DB_SESSIONS_DIR = "db_sessions"

def format_docs(docs: list[Document]) -> str:
    """
    Retrieval'dan gelen Document nesnelerini temiz bir string olarak formatlar.
    Bu, LLM'e gönderilen context'in daha anlaşılır olmasını sağlar.
    """
    return "\n\n".join(
        f"Alıntı (Başlangıç: {doc.metadata.get('start', 'N/A')}s):\n{doc.page_content}"
        for doc in docs
    )

def merge_documents_for_summary(docs: list[Document], model_name: str = "llama3:8b", token_limit: int = 1000) -> list[Document]:
    """
    Küçük dökümanları, modelin token limitini aşmayacak şekilde akıllıca birleştirir.
    'tiktoken' kütüphanesini kullanarak kelime yerine token sayar.
    """
    try:
        # Modeller genellikle farklı tokenizasyon şemaları kullanır, ancak cl100k_base
        # çoğu modern model için iyi bir genel yaklaşımdır.
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Nadir durumlarda, encoding adı değişebilir. Güvenli bir varsayılan kullan.
        encoding = tiktoken.encoding_for_model("gpt-4")
        
    # Önce dökümanları sequence_id'ye göre sırala
    sorted_docs = sorted(docs, key=lambda x: x.metadata.get('sequence_id', 0))
    
    new_docs = []
    current_tokens = 0
    current_content = []

    for doc in sorted_docs:
        doc_content = doc.page_content
        # 'page_content' boşsa atla
        if not doc_content.strip():
            continue
            
        # Yeni dökümanın token sayısını hesapla
        doc_tokens = len(encoding.encode(doc_content))
        
        # Eğer bu dökümanı eklemek limiti aşacaksa, mevcut parçayı bitir ve yenisine başla
        if current_tokens + doc_tokens > token_limit and current_content:
            new_doc = Document(page_content=" ".join(current_content))
            new_docs.append(new_doc)
            # Mevcut parçayı sıfırla
            current_content = []
            current_tokens = 0
        
        # Dökümanı mevcut parçaya ekle
        current_content.append(doc_content)
        current_tokens += doc_tokens

    # Döngü bittiğinde kalan son parçayı da ekle
    if current_content:
        new_doc = Document(page_content=" ".join(current_content))
        new_docs.append(new_doc)
        
    print(f"{len(docs)} adet küçük döküman, token limitine göre {len(new_docs)} adet parçaya birleştirildi.")
    return new_docs


class VectorDBManager:
    def __init__(self, session_id: str):
        """
        VectorDBManager'ı belirli bir oturum ID'si için başlatır.
        Her oturum kendi izole veritabanı klasörüne sahip olur.
        """
        print(f"VectorDBManager: '{session_id}' oturumu için başlatılıyor.")
        self.session_id = session_id
        self.persist_directory = os.path.join(DB_SESSIONS_DIR, self.session_id)
        
        # Ensure session directory exists
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)
            
        self.llm = ChatOllama(base_url=config.OLLAMA_BASE_URL, model=config.LLM_MODEL, temperature=0)
        self.embeddings = OllamaEmbeddings(base_url=config.OLLAMA_BASE_URL, model=config.EMBEDDING_MODEL)
        
        # ChromaDB istemcisini ve koleksiyonu daha kontrollü yönetelim
        # Bu, kaynakları serbest bırakmamıza olanak tanır.
        self._client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma(
            client=self._client,
            collection_name=f"ars_collection_{self.session_id}",
            embedding_function=self.embeddings
        )
        print(f"VectorDBManager: Veritabanı '{self.persist_directory}' konumunda hazır.")

    def release(self):
        """
        ChromaDB istemcisini durdurarak kaynakları (dosya tanıtıcıları gibi) serbest bırakır.
        Bu, veritabanı dosyalarını güvenli bir şekilde silmeden önce çağrılmalıdır.
        """
        if hasattr(self, '_client') and self._client:
            try:
                print(f"'{self.session_id}' oturumu için ChromaDB istemcisi durduruluyor...")
                self._client.stop()
                self._client = None # Referansı kaldır
                print("   -> İstemci başarıyla durduruldu.")
            except Exception as e:
                print(f"   -> UYARI: ChromaDB istemcisi durdurulurken bir hata oluştu: {e}")

    def ingest_segments(self, segments: list, source_filename: str):
        """
        Transkript segmentlerini alır, gerekirse böler ve oturuma özel veritabanına işler.
        """
        print(f"'{source_filename}' için {len(segments)} adet segment işleniyor...")
        
        # Metinleri birleştirip sonra bölmek, daha tutarlı parçalar oluşturur.
        full_text = " ".join(segment.get('text', '').strip() for segment in segments if segment.get('text', '').strip())
        
        if not full_text:
            print("UYARI: Seste işlenecek metin bulunamadı.")
            return

        # Metni yönetilebilir parçalara böl
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        texts = text_splitter.split_text(full_text)
        
        # Orijinal segmentlerdeki metadata'yı burada kaybediyoruz ancak
        # uzun metinler için bölümleme yapmak daha kritik.
        # Basitlik adına, tüm parçalara aynı kaynak bilgisini ekliyoruz.
        documents = [
            Document(page_content=text, metadata={'source': source_filename}) 
            for text in texts
        ]

        if not documents:
            print("UYARI: Bölümleme sonrası işlenecek metin bulunamadı.")
            return
        
        self.vector_store.add_documents(documents)
        print(f"   -> Metin {len(documents)} parçaya bölündü ve başarıyla veritabanına eklendi.")


    def get_rag_chain(self):
        """Soru-cevap (chat) için RAG zincirini kurar ve döndürür."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        prompt_template = ChatPromptTemplate.from_template(config.RAG_PROMPT_TEMPLATE)
        
        rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} 
            | prompt_template 
            | self.llm 
            | StrOutputParser()
        )
        return rag_chain

    def get_report_chain(self):
        """Toplantı raporu için Map-Reduce zincirini kurar ve döndürür."""
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 1000})
        # LangChain'in özetleme zinciri, bir retriever yerine doğrudan döküman listesi bekler.
        # Bu yüzden burada zinciri değil, dökümanları ve zinciri ayrı ayrı döndüreceğiz.
        
        map_prompt = PromptTemplate.from_template(config.MAP_PROMPT_TEMPLATE)
        combine_prompt = PromptTemplate.from_template(config.COMBINE_PROMPT_TEMPLATE)
        
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False # Konsolu gereksiz yere doldurmaması için kapatıldı
        )
        return chain, retriever

    def generate_report(self) -> str:
        """
        Veritabanındaki tüm belgeleri alır, raporlama için birleştirir,
        map-reduce zincirini çalıştırır ve nihai rapor metnini döndürür.
        """
        try:
            report_chain, retriever = self.get_report_chain()
            all_docs = retriever.invoke("tüm belgeyi özetle")
            
            if not all_docs:
                return "Rapor oluşturmak için döküman bulunamadı."

            # Raporlama için dökümanları birleştir (artık token bazlı)
            merged_docs = merge_documents_for_summary(all_docs, model_name=config.LLM_MODEL, token_limit=7000)
            
            result = report_chain.invoke({"input_documents": merged_docs}, config={"max_concurrency": 4})
            return result.get('output_text', "Rapor metni alınamadı.")

        except Exception as e:
            print(f"HATA: Rapor oluşturulurken bir istisna oluştu: {e}")
            return "Rapor oluşturulurken beklenmedik bir hata oluştu. Lütfen konsol loglarını kontrol edin."

    def clear_database(self):
        """
        Bu oturuma ait veritabanı kaynaklarını serbest bırakır ve klasörü diskten siler.
        """
        print(f"Veritabanı temizleniyor: {self.persist_directory}")
        
        # Önce kaynakları serbest bırakmayı dene
        self.release()

        if os.path.exists(self.persist_directory):
            try:
                # Kaynaklar serbest bırakıldıktan sonra sil
                shutil.rmtree(self.persist_directory)
                print("   -> Veritabanı başarıyla silindi.")
            except Exception as e:
                print(f"   -> Veritabanı silinirken bir hata oluştu: {e}")
        else:
            print("   -> Silinecek veritabanı bulunamadı, işlem atlanıyor.") 