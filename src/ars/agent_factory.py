from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from . import config, llm_setup

# Raporlama için kullanılacak basit zincir
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents import Document
from operator import itemgetter
from .llm_setup import LLMSetup
from .manager import AIManager # AIManager'ı import ediyoruz
from langchain.prompts import PromptTemplate

# Neo4j veritabanı şemasını manuel olarak tanımlıyoruz.
# Bu, LLM'in doğru ve etkili Cypher sorguları oluşturmasına yardımcı olur.
NEO4J_SCHEMA = """
Node properties are the following:
Entity {name: STRING}

Relationship properties are the following:
RELATIONSHIP {type: STRING}

The schema is as follows:
(:Entity)-[:RELATIONSHIP]->(:Entity)
"""

def create_map_reduce_chain(llm, graph: Neo4jGraph):
    """
    Uzun metinleri özetlemek için bir Map-Reduce zinciri oluşturur.
    Bu versiyon, "Reduce" adımını zenginleştirmek için Neo4j'den graf bilgileri çeker.
    """
    # Map adımı için istem
    map_template = """
Senin görevin, daha büyük bir metnin bir parçasını analiz etmektir.
Aşağıdaki metin parçasının ana fikirlerini, önemli argümanlarını ve bağlamını yakalayan yoğun bir özet oluştur.
Yapısal bir format kullanmak yerine, metnin ruhunu ve en kritik bilgilerini içeren, iyi yazılmış bir paragraf hazırla.
Bu özet, daha sonra diğer özetlerle birleştirilerek nihai bir rapor oluşturulmasında kullanılacaktır, bu yüzden eksiksiz ve anlaşılır olması çok önemlidir.

METİN:
"{text}"

YOĞUN TÜRKÇE ÖZET:
"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = map_prompt | llm | StrOutputParser()

    # Reduce adımı için istem
    reduce_template = config.REPORT_PROMPT
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    def get_graph_data(_) -> str:
        """
        Veritabanındaki tüm ilişkileri çeken ve formatlayan deterministik bir fonksiyondur.
        LLM tabanlı sorgu üretiminin güvenilmezliği nedeniyle bu yöntem tercih edilmiştir.
        """
        cypher_query = "MATCH (h:Entity)-[r:RELATIONSHIP]->(t:Entity) RETURN h.name AS head, r.type AS type, t.name AS tail LIMIT 25"
        try:
            results = graph.query(cypher_query)
            if not results:
                return "Metinden çıkarılan konular arasında belirgin bir ilişki bulunamadı."
            
            # Sonuçları madde imleri ile formatla
            formatted_triplets = []
            for res in results:
                formatted_triplets.append(f"- **{res['head']}** --[{res['type']}]--> **{res['tail']}**")
            
            return "\n".join(formatted_triplets)

        except Exception as e:
            print(f"Graf sorgusu sırasında hata: {e}")
            return "Metindeki ilişkiler analiz edilirken bir hata oluştu."

    def map_and_combine(docs: list[Document]) -> dict:
        """
        Belgeleri map_chain ile işler, birleştirir ve graf verilerini çeker.
        """
        summaries = map_chain.batch([{"text": doc.page_content} for doc in docs])
        combined_summary = "\n\n---\n\n".join(summaries)
        
        graph_data = get_graph_data(None) # Argüman artık kullanılmıyor

        return {
            "text": combined_summary,
            "graph_data": graph_data
        }

    # Nihai zincir, input olarak {"input_documents": [docs...]} şeklinde bir sözlük bekler.
    reduce_chain = (
        itemgetter("input_documents")     # Sözlükten döküman listesini çıkarır.
        | RunnableLambda(map_and_combine) # Fonksiyonu Runnable'a çevirir.
        | reduce_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return reduce_chain


# Agent için özel olarak tasarlanmış ReAct prompt şablonu (Örnekli ve Düzeltilmiş Versiyon)
AGENT_PROMPT_TEMPLATE = """Sen, bir doküman hakkındaki soruları yanıtlamakla görevli yardımcı bir asistansın.
Şu araçlara erişimin var:
{tools}

HER ZAMAN aşağıdaki formatı kullan:

Thought: Kullanıcının sorusunu analiz et ve hangi aracın en iyisi olduğuna karar ver.
Action: Kullanılacak araç. Bu, şu araçlardan biri OLMALIDIR: {tool_names}
Action Input: Araç için giriş sorgusu.
Observation: Araçtan gelen sonuç.
... (bu Thought/Action/Action Input/Observation döngüsü tekrarlanabilir)
Thought: Yeterli bilgiyi topladım. Şimdi nihai yanıtı oluşturacağım.
Final Answer: Kullanıcının sorusuna verilen nihai yanıt.

İşte başarılı bir etkileşim örneği:
---
Soru: Üretkenlik hakkında ne konuşuldu?
Thought: Kullanıcı bir konu hakkında genel bir soru soruyor. Anlamsal Arama bunun için en iyi araçtır.
Action: Semantic Search
Action Input: Üretkenlik hakkında ne konuşuldu?
Observation: Belgede üretken olmanın abartıldığı ve sıkılmanın faydaları olduğu belirtiliyor.
Thought: Cevap vermek için yeterli bilgiye sahibim.
Final Answer: Belge, üretken olmanın abartılabileceğini ve sıkıntıyı benimsemenin faydaları olabileceğini öne sürüyor.
---

Şimdi başla!

Soru: {input}
{agent_scratchpad}
"""

def create_agent(llm, embedding_model):
    """
    Neo4j ve ChromaDB araçlarını kullanarak bir ReAct agent oluşturur ve döndürür.
    Bu versiyon, LLM'in Cypher üretme zorluğunu ortadan kaldırmak için deterministik bir Graph aracı kullanır.
    """
    # 1. ChromaDB'den anlamsal arama aracı (Semantic Search Tool)
    vector_store = Chroma(
        persist_directory=config.CHROMA_DB_PATH, 
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs: list[Document]) -> str:
        """Belgeleri tek bir string'de birleştirir."""
        return "\n\n".join(doc.page_content for doc in docs)

    vector_store_retriever_tool = Tool(
        name="Semantic Search",
        func=lambda query: format_docs(retriever.invoke(query)),
        description="Transkriptin bölümleri arasında anlamsal arama yapmak için kullanılır. Belirli konular, kişiler veya olaylar hakkında genel bilgi bulmak için idealdir.",
    )

    # 2. Neo4j'den deterministik graf arama aracı
    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        refresh_schema=False
    )

    # LLM'in tek görevi sorudan anahtar kelimeleri çıkarmak olacak.
    entity_extraction_prompt = PromptTemplate.from_template(
        """Bir kullanıcı sorusundan, ilişki sorgusu için anlamlı olan en önemli özel isimleri (kişi, yer, organizasyon, proje vb.) çıkar.
        "ne", "hakkında", "video", "kimdir" gibi genel ve anlamsız kelimeleri YOK SAY.
        Eğer anlamlı bir özel isim bulamazsan, boş bir liste döndür.
        Sadece bulunan özel isimleri bir JSON listesi olarak formatla.
        Örnek 1: 'arda ile xabi arasındaki ilişki nedir?' -> ["arda", "xabi"]
        Örnek 2: 'real madrid hakkında bilgi ver.' -> ["real madrid"]
        Örnek 3: 'ne hakkında konuşuluyor?' -> []
        Soru: "{question}"
        """
    )
    
    entity_extraction_chain = entity_extraction_prompt | llm | JsonOutputParser()

    def graph_search_func(question: str) -> str:
        """Kullanıcı sorusundan varlıkları çıkarır ve bunları sabit bir Cypher sorgu şablonuna enjekte eder."""
        entities = entity_extraction_chain.invoke({"question": question})
        if not entities or len(entities) < 1:
            return "İlişki sorgulamak için lütfen daha spesifik bir soru sorun (örn: 'X ve Y arasındaki ilişki nedir?')."
        
        # Sorguyu büyük/küçük harf duyarsız hale getiriyoruz
        entity1 = entities[0]
        # Eğer ikinci bir varlık varsa onu da kullan, yoksa ilk varlığın geçtiği tüm ilişkileri bul
        if len(entities) > 1:
            entity2 = entities[1]
            cypher_query = f"""
            MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity)
            WHERE e1.name =~ '(?i).*{entity1}.*' AND e2.name =~ '(?i).*{entity2}.*'
            RETURN e1.name AS head, r.type AS type, e2.name AS tail LIMIT 10
            """
        else:
            cypher_query = f"""
            MATCH (e1:Entity)-[r:RELATIONSHIP]->(e2:Entity)
            WHERE e1.name =~ '(?i).*{entity1}.*' OR e2.name =~ '(?i).*{entity1}.*'
            RETURN e1.name AS head, r.type AS type, e2.name AS tail LIMIT 10
            """
        
        try:
            results = graph.query(cypher_query)
            if not results:
                return f"'{' ve '.join(entities)}' içeren bir ilişki bulunamadı."
            
            return "\n".join([f"- {res['head']} --[{res['type']}]--> {res['tail']}" for res in results])
        except Exception as e:
            return f"Veritabanı sorgulanırken bir hata oluştu: {e}"

    graph_search_tool = Tool(
        name="Graph Search",
        func=graph_search_func,
        description="""Varlıklar (kişiler, yerler, konular) arasındaki ilişkileri bulmak için kullanılır. Girdi olarak doğal dilde bir soru verin."""
    )

    tools = [vector_store_retriever_tool, graph_search_tool]

    # 3. ReAct Agent'ı oluşturma
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=15, 
        return_intermediate_steps=True
    )

    return agent_executor 

def create_conversational_agent(llm, ai_manager: AIManager):
    """
    Neo4j ve ChromaDB araçlarını kullanan bir konuşma agent'ı oluşturur.
    """
    # Neo4j aracını oluştur
    # graph nesnesini doğrudan AIManager'dan alıyoruz.
    graph = ai_manager.graph 
    
    neo4j_tool = Tool(
        name="Grafik Veritabanı Sorgulayıcı",
        description=(
            "Veritabanındaki kişiler, yerler, olaylar ve diğer varlıklar arasındaki "
            "ilişkiler ve bağlantılar hakkındaki soruları yanıtlamak için kullanılır. "
            "Örneğin: 'X ve Y arasında nasıl bir bağlantı var?', 'A projesinde kimler çalıştı?'"
        ),
        func=graph.query,
    )

    # ChromaDB (Vektör Deposu) aracını oluştur
    chroma_tool = Tool(
        name="Anlamsal Arama",
        description="Transkriptin bölümleri arasında anlamsal arama yapmak için kullanılır. Belirli konular, kişiler veya olaylar hakkında genel bilgi bulmak için idealdir.",
        func=lambda query: format_docs(ai_manager.vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)),
    )

    tools = [neo4j_tool, chroma_tool]

    # 3. ReAct Agent'ı oluşturma
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True
    )

    return agent_executor 