# Bu dosya, "temiz" yapay zeka işlerinden sorumlu olacaktır.
# LangChain kullanarak metin işleme, ChromaDB ve Neo4j veritabanlarını yönetme,
# ve sohbet/raporlama zincirlerini oluşturma gibi işlemler burada yer alacaktır. 

import os
import json
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
from langchain_community.graphs import Neo4jGraph

from . import config
from .llm_setup import LLMSetup

class AIManager:
    """
    Bu sınıf, tüm "temiz" yapay zeka işlemlerini yönetir.
    - Metinleri işleme ve bölme
    - ChromaDB ve Neo4j veritabanlarını doldurma
    """
    def __init__(self):
        llm_setup = LLMSetup()
        self.llm = llm_setup.get_llm()
        self.embedding_model = llm_setup.get_embedding_model()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        # Neo4jGraph nesnesini burada merkezi olarak oluşturuyoruz.
        self.graph = Neo4jGraph(
            url=config.NEO4J_URI,
            username=config.NEO4J_USERNAME,
            password=config.NEO4J_PASSWORD,
            refresh_schema=False
        )

    def _extract_json_from_text(self, text: str) -> dict:
        """
        LLM'den gelen metin yanıtından JSON bloğunu ayıklar.
        Metin içindeki ilk '{' ve son '}' karakterlerini bularak çalışır.
        Eğer JSON bloğu bulunamazsa veya ayrıştırma hatası olursa boş bir sözlük döner.
        """
        try:
            # Markdown kod bloğunu (```json ... ```) temizle
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            
            # Alternatif olarak, sadece ilk '{' ve son '}' arasını al
            start_index = text.find('{')
            end_index = text.rfind('}')
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                json_str = text[start_index:end_index+1]
                return json.loads(json_str)
            else:
                return {} # JSON nesnesi bulunamadı
        except (json.JSONDecodeError, IndexError) as e:
            print(f"--> JSON ayrıştırma hatası: {e}. Metin: '{text[:100].strip()}...'")
            return {} # Hata durumunda boş sözlük dön

    def _format_transcript(self, whisperx_result: dict) -> str:
        """WhisperX çıktısını düz bir metne dönüştürür."""
        df = pd.DataFrame(whisperx_result['segments'])
        # Konuşmacı bilgisi varsa metne ekle
        if 'speaker' in df.columns:
            return "\n".join([f"{row['speaker']}: {row['text']}" for _, row in df.iterrows()])
        else:
            return "\n".join([row['text'] for _, row in df.iterrows()])

    def clear_databases(self):
        print("Neo4j veritabanı temizleniyor...")
        try:
            self.graph.query("MATCH (n) DETACH DELETE n")
            print("Neo4j veritabanı başarıyla temizlendi.")
        except Exception as e:
            print(f"Neo4j temizlenirken hata oluştu: {e}")

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

        # --- Neo4j'yi Doldur ---
        print("Neo4j graf veritabanı dolduruluyor...")
        
        # LLM zincirinden JSON parser'ı çıkarıyoruz, sadece ham metin üretecek
        prompt = ChatPromptTemplate.from_template(config.TRIPLET_EXTRACTION_TEMPLATE)
        chain = prompt | self.llm 

        all_triplets = []
        print("Metin parçalarından üçlüler çıkarılıyor...")
        for i, chunk in enumerate(chunks):
            print(f"Parça {i+1}/{len(chunks)} işleniyor...")
            llm_output = "" # Hata durumunda loglamak için boş string tanımlayalım
            try:
                # Zinciri çağırıp LLM'den ham metin cevabını alalım
                response = chain.invoke({"text": chunk.page_content})
                
                # Gelen cevap AIMessage nesnesi ise içeriğini, değilse string'e çevirerek al
                llm_output = response.content if hasattr(response, 'content') else str(response)

                # LLM çıktısından JSON'ı ayıkla
                extracted_json = self._extract_json_from_text(llm_output)
                
                if extracted_json and "triplets" in extracted_json:
                    all_triplets.extend(extracted_json["triplets"])
                else:
                    print(f"--> UYARI: Parça {i+1} için geçerli üçlü bulunamadı, atlanıyor. LLM çıktısı: '{llm_output[:100].strip()}...'")

            except Exception as e:
                # Diğer olası beklenmedik hatalar için
                print(f"--> HATA: Parça {i+1} işlenirken beklenmedik bir hata oluştu (atlandı): {e}")
                continue
        
        # --- Üçlüleri filtreleme ve Neo4j'e yazma kısmı ---
        print(f"Filtreleme öncesi üçlü sayısı: {len(all_triplets)}")
        valid_triplets = [
            t for t in all_triplets 
            if t.get("head") and t.get("tail") and t.get("relation")
        ]
        print(f"Filtreleme sonrası geçerli üçlü sayısı: {len(valid_triplets)}")

        if valid_triplets:
            # Merkezi `self.graph` nesnesini kullanıyoruz.
            self.graph.query(
                """
                UNWIND $triplets AS triplet
                MERGE (h:Entity {name: triplet.head})
                MERGE (t:Entity {name: triplet.tail})
                MERGE (h)-[r:RELATIONSHIP {type: triplet.relation}]->(t)
                """, 
                params={'triplets': valid_triplets}
            )
            print(f"{len(valid_triplets)} adet üçlü Neo4j'e başarıyla yazıldı.")
        else:
            print("Metinden çıkarılacak anlamlı bir üçlü bulunamadı.")
            
        return chunks 