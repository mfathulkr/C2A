# Bu dosya, "temiz" yapay zeka işlerinden sorumlu olacaktır.
# LangChain kullanarak metin işleme, ChromaDB ve Neo4j veritabanlarını yönetme,
# ve sohbet/raporlama zincirlerini oluşturma gibi işlemler burada yer alacaktır. 

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from langchain_neo4j import Neo4jGraph

from . import config
# import . import neo4j_manager as nm # Bu import artık gerekli değil

# Üçlü çıkarma için kullanılacak prompt şablonu
TRIPLET_EXTRACTION_TEMPLATE = """
Sen bir metin analistisin. Görevin, verilen metin parçasından Bilgi Grafiği için Bilgi Üçlüleri (head, relation, tail) çıkarmaktır.
Her üçlü (baş varlık, ilişki, kuyruk varlık) formatında olmalıdır. Varlıkları ve ilişkileri olabildiğince kısa ve öz tut.
ÖNEMLİ: Çıktın SADECE geçerli bir JSON nesnesi içermelidir. Başka hiçbir metin, açıklama veya not ekleme.

Örnek:
Metin: "Elon Musk, elektrikli araç üreticisi olan Tesla'yı kurdu ve aynı zamanda SpaceX'in de CEO'sudur."
Çıktı:
```json
{{
  "triplets": [
    {{"head": "Elon Musk", "relation": "KURDU", "tail": "Tesla"}},
    {{"head": "Tesla", "relation": "TÜRÜ", "tail": "Elektrikli Araç Üreticisi"}},
    {{"head": "Elon Musk", "relation": "CEO", "tail": "SpaceX"}}
  ]
}}
```

Aşağıdaki metinden tüm anlamlı üçlüleri çıkar.
Metin:
---
{text}
---
"""

class AIManager:
    """
    Bu sınıf, tüm "temiz" yapay zeka işlemlerini yönetir.
    - Metinleri işleme ve bölme
    - ChromaDB ve Neo4j veritabanlarını doldurma
    """
    def __init__(self, llm, embedding_model):
        self.llm = llm
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=120)

    def _format_transcript(self, whisperx_result: dict) -> str:
        """WhisperX çıktısını düz bir metne dönüştürür."""
        df = pd.DataFrame(whisperx_result['segments'])
        # Konuşmacı bilgisi varsa metne ekle
        if 'speaker' in df.columns:
            return "\n".join([f"{row['speaker']}: {row['text']}" for _, row in df.iterrows()])
        else:
            return "\n".join([row['text'] for _, row in df.iterrows()])

    def populate_databases(self, whisperx_result: dict):
        """
        Transkripti işler ve hem ChromaDB'yi hem de Neo4j'yi doldurur.
        """
        full_transcript = self._format_transcript(whisperx_result)
        docs = [Document(page_content=full_transcript)]
        
        # Metni parçalara ayır
        chunks = self.text_splitter.split_documents(docs)
        print(f"Metin, {len(chunks)} parçaya (chunk) ayrıldı.")

        # --- ChromaDB'yi Doldur (Vektörler için) ---
        print("ChromaDB vektör veritabanı dolduruluyor...")
        Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_model, 
            persist_directory=config.CHROMA_DB_PATH
        )
        print("ChromaDB başarıyla oluşturuldu.")

        # --- Neo4j'yi Doldur (Graflar için) ---
        print("Neo4j graf veritabanı dolduruluyor...")
        # LLM ile üçlü çıkarma zinciri
        prompt = ChatPromptTemplate.from_template(TRIPLET_EXTRACTION_TEMPLATE)
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        all_triplets = []
        print("Metin parçalarından üçlüler çıkarılıyor...")
        for i, chunk in enumerate(chunks):
            print(f"Parça {i+1}/{len(chunks)} işleniyor...")
            try:
                extracted_json = chain.invoke({"text": chunk.page_content})
                all_triplets.extend(extracted_json.get("triplets", []))
            except Exception as e:
                print(f"Parça {i+1} işlenirken hata oluştu (atlandı): {e}")
                continue
        
        # Neo4j'ye göndermeden önce geçersiz (boş) üçlüleri filtrele
        print(f"Filtreleme öncesi üçlü sayısı: {len(all_triplets)}")
        valid_triplets = [
            t for t in all_triplets 
            if t.get("head") and t.get("tail") and t.get("relation")
        ]
        print(f"Filtreleme sonrası geçerli üçlü sayısı: {len(valid_triplets)}")

        # Üçlüleri Neo4j'e yaz
        if valid_triplets:
            graph = Neo4jGraph(
                url=config.NEO4J_URI,
                username=config.NEO4J_USERNAME,
                password=config.NEO4J_PASSWORD,
                refresh_schema=False # APOC çağrısını ve hatasını önlemek için şema anlama özelliğini kapat
            )
            # Üçlüleri Neo4j'e yazmak için Cypher sorgusu
            query = """
            UNWIND $triplets AS triplet
            MERGE (h:Entity {name: triplet.head})
            MERGE (t:Entity {name: triplet.tail})
            MERGE (h)-[r:RELATIONSHIP {type: triplet.relation}]->(t)
            """
            graph.query(query, params={'triplets': valid_triplets})
            print(f"{len(valid_triplets)} adet üçlü Neo4j'e başarıyla yazıldı.")
        else:
            print("Metinden çıkarılacak anlamlı bir üçlü bulunamadı.")
            
        return chunks # Raporlama için parçaları döndür 