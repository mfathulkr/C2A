from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.vectorstores import Chroma
from langchain_core.tools import Tool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from . import config, llm_setup

# Raporlama için kullanılacak basit zincir
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from .llm_setup import LLMSetup
from .manager import AIManager # AIManager'ı import ediyoruz
from langchain.prompts import PromptTemplate

def create_reporting_chain(llm, prompt_template):
    """
    Sadece metin alıp, belirli bir şablona göre rapor üreten basit bir LLM zinciri oluşturur.
    """
    prompt = PromptTemplate.from_template(config.REPORT_PROMPT)
    
    chain = (
        {"context": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Uzun metinleri özetlemek için Map-Reduce zinciri
def create_map_reduce_chain(llm):
    """
    Uzun metinleri özetlemek için bir Map-Reduce zinciri oluşturur.
    `load_summarize_chain` kullanarak modern LangChain API'sini takip eder.
    """
    # Map adımı için istem
    map_template = """
Aşağıdaki metin parçası, daha büyük bir toplantı veya sunumun bir bölümüdür. Bu metni analiz et ve aşağıdaki bilgileri yapısal olarak çıkar:
- **Ana Konular:** Bu parçada tartışılan ana konu veya konular nelerdir?
- **Belirtilen Kararlar:** Eğer varsa, bu parçada alınan net bir karar var mı?
- **Eylem Adımları/Görevler:** Gelecekte yapılması gereken bir görev veya eylem adımı belirtiliyor mu?
- **Kilit Fikirler:** Konuşmanın genel özeti için önemli olan kilit fikirleri veya argümanları listele.

METİN:
"{text}"

ÇIKARILAN BİLGİLER:
"""
    map_prompt = PromptTemplate.from_template(map_template)

    # Reduce adımı için istem ve zincir
    # load_summarize_chain'in birleştirme adımı, özetleri "text" değişkeni altında bekler.
    # Mevcut REPORT_PROMPT şablonumuz "context" kullandığı için, bunu anında değiştiriyoruz.
    reduce_template = config.REPORT_PROMPT.replace("{context}", "{text}")
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # `load_summarize_chain` kullanarak zinciri oluştur.
    # Bu, eski manuel MapReduceDocumentsChain kurulumunun yerini alır.
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=reduce_prompt,
        verbose=True,
    )
    return chain

# Agent için özel olarak tasarlanmış ReAct prompt şablonu (Örnekli ve Düzeltilmiş Versiyon)
AGENT_PROMPT_TEMPLATE = """You are a helpful assistant. Your job is to answer questions about a document.
You have access to the following tools:
{tools}

ALWAYS use the following format:

Thought: Analyze the user's question and decide which tool is best.
Action: The tool to use. This MUST be one of: {tool_names}
Action Input: The input query for the tool.
Observation: The result from the tool.
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I have gathered enough information. I will now form the final answer.
Final Answer: The final answer to the user's question.

Here is an example of a successful interaction:
---
Question: What was discussed about productivity?
Thought: The user is asking a general question about a topic. Semantic Search is the best tool for this.
Action: Semantic Search
Action Input: What was discussed about productivity?
Observation: The document mentions that being productive is overrated and that boredom has benefits.
Thought: I have enough information to answer.
Final Answer: The document suggests that being productive might be overrated and that embracing boredom can have benefits.
---

Now, begin!

Question: {input}
{agent_scratchpad}
"""

def create_agent(llm, embedding_model):
    """
    Neo4j ve ChromaDB araçlarını kullanarak bir ReAct agent oluşturur ve döndürür.
    """
    # 1. ChromaDB'den anlamsal arama aracı (Semantic Search Tool)
    vector_store = Chroma(
        persist_directory=config.CHROMA_DB_PATH, 
        embedding_function=embedding_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Aracı formatlamak için yardımcı fonksiyon
    def format_docs(docs: list[Document]) -> str:
        """Belgeleri tek bir string'de birleştirir."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Vektör deposundan belgeleri almak ve formatlamak için bir araç oluştur
    vector_store_retriever_tool = Tool(
        name="Semantic Search",
        func=lambda query: format_docs(retriever.invoke(query)),
        description="Transkriptin bölümleri arasında anlamsal arama yapmak için kullanılır. Belirli konular, kişiler veya olaylar hakkında bilgi bulmak için idealdir.",
    )

    # 2. Neo4j'den graf arama aracı (Graph Search Tool)
    # db_manager = neo4j_manager.Neo4jManager() # Artık buna gerek yok, Neo4jGraph doğrudan hallediyor.
    graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
        refresh_schema=False # APOC çağrısını ve hatasını önlemek için şema anlama özelliğini kapat
    )
    
    graph_search_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph, # Doğrudan driver yerine Neo4jGraph nesnesini kullan
        verbose=True, # Hangi Cypher sorgularının çalıştığını görmek için
        validate_cypher=True,
        allow_dangerous_requests=True # Güvenlik uyarısını onaylamak için
    )

    graph_search_tool = Tool(
        name="Graph Search",
        func=graph_search_chain.invoke,
        description="""Use this tool to understand the relationships and connections between entities.
        It is ideal for specific questions about connections, like: 'Who said what to whom?', 'Which company is associated with which project?', or 'What is the link between X and Y?'.
        The input should be a natural language question about relationships.
        """
    )

    tools = [vector_store_retriever_tool, graph_search_tool]

    # 3. ReAct Agent'ı oluşturma
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)
    
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, # Olası hataları yakalamak için
        max_iterations=5 # Sonsuz döngüleri önlemek için
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
        func=lambda query: graph.query(query),
    )

    # ChromaDB (Vektör Deposu) aracını oluştur
    chroma_tool = Tool(
        name="Anlamsal Arama",
        description="Transkriptin bölümleri arasında anlamsal arama yapmak için kullanılır. Belirli konular, kişiler veya olaylar hakkında bilgi bulmak için idealdir.",
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
        handle_parsing_errors=True, # Olası hataları yakalamak için
        max_iterations=5 # Sonsuz döngüleri önlemek için
    )

    return agent_executor 