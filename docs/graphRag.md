# Proje: Mevcut RAG Uygulamasına Graph RAG ve Agentic Yetenekler Ekleme

## 1. Genel Bakış ve Amaç

Bu projenin amacı, halihazırda Streamlit arayüzü ve ChromaDB vektör veritabanı ile çalışan mevcut RAG (Retrieval-Augmented Generation) uygulamasına, ilişkisel ve yapısal verileri sorgulamak için bir **Bilgi Grafiği (Knowledge Graph)** eklemektir. Nihai hedef, kullanıcı sorgularını analiz ederek en uygun bilgi kaynağını (Vektör Veritabanı veya Bilgi Grafiği) akıllıca seçebilen veya her ikisini birden kullanabilen bir **"Agentic RAG"** mimarisi oluşturmaktır. Bu, daha isabetli, kapsamlı ve bağlamı anlayan raporlar üretilmesini sağlayacaktır.

**Mevcut Durum:**
- **Arayüz:** Streamlit (`streamlit_app.py`)
- **Veri Depolama:** Yüklenen dokümanlar metin olarak işleniyor.
- **Bilgi Erişimi:** Anlamsal arama için ChromaDB vektör veritabanı kullanılıyor.
- **Dil Modeli:** Llama 3.1 (yerelde çalışıyor).

**Hedef Mimari:**
1.  **Kullanıcı Sorgusu:** Streamlit arayüzünden alınır.
2.  **Router (Yönlendirici) Ajan:** Sorgunun niteliğini analiz eder.
    - "Bu soru anlamsal bir arama mı gerektiriyor?" -> Vektör Veritabanına yönlendir.
    - "Bu soru varlıklar ve ilişkileri hakkında mı?" -> Bilgi Grafiğine yönlendir.
    - "Bu soru her ikisini de gerektiren karmaşık bir analiz mi?" -> Her iki kaynağı da kullan.
3.  **Bilgi Getirme (Retrieval):**
    - **ChromaDB:** Metin parçacıklarını döndürür.
    - **Bilgi Grafiği:** Varlık düğümlerini ve aralarındaki ilişkileri döndürür.
4.  **Sentezleme (Synthesis):** Getirilen tüm bilgiler (metinler ve grafik verileri) Llama 3.1 modeline tek bir "context" olarak sunulur ve nihai, zenginleştirilmiş cevap/rapor oluşturulur.

---

## 2. Kurulumlar: Gerekli Kütüphaneler

Mevcut `requirements.txt` dosyasına ek olarak aşağıdaki kütüphanelerin kurulması gerekmektedir. Bu kütüphaneler, ajan mantığı, bilgi grafiği oluşturma ve yönetimi için kullanılacaktır.

```bash
pip install langchain langchain-community langchain-experimental networkx pyvis
```

- **`langchain`, `langchain-community`:** Agent (ajan), router (yönlendirici) ve farklı veri kaynaklarını birbirine bağlayan zincirleri (chains) oluşturmak için temel çerçeve.
- **`langchain-experimental`:** Bilgi grafiği ile entegrasyon için deneysel modüller içerir.
- **`networkx`:** Bilgi grafiğini bellekte oluşturmak, yönetmek ve sorgulamak için standart Python kütüphanesi.
- **`pyvis`:** Geliştirme aşamasında oluşturulan grafiği görselleştirmek ve doğrulamak için son derece faydalı bir araç.

---

## 3. Adım Adım Entegrasyon Planı

### Adım 3.1: Bilgi Grafiği Oluşturma Modülü

Projenin `src/ars/` klasörüne `graph_builder.py` adında yeni bir modül ekleyin. Bu modülün görevi, mevcut dokümanlardan varlıkları (entities) ve ilişkileri (relations) çıkararak bir `networkx` grafiği oluşturmak ve bunu daha sonra kullanmak üzere bir dosyaya kaydetmektir.

**`src/ars/graph_builder.py` için görevler:**

1.  **Dokümanları Yükle:** Dokümanları işlemek için mevcut veri yükleme fonksiyonlarınızı kullanın.
2.  **Varlık ve İlişki Çıkarımı için LLM Prompt'u:** Llama 3.1'i kullanarak metinden yapısal bilgi çıkaracak bir fonksiyon yazın. Bu fonksiyon, metin parçalarını alıp (Varlık1, İlişki, Varlık2) formatında üçlüler (triplets) döndürmelidir.

    *Örnek Prompt Şablonu:*
    ```
    Bir metin verilecektir. Görevin, metindeki ana varlıkları ve aralarındaki ilişkileri (Varlık1, İlişki, Varlık2) formatında bir liste olarak çıkarmaktır. Sadece en önemli ve net ilişkileri listele. Varlıklar Şirket, Kişi, Ürün veya Yer olabilir.

    Metin: 'Apple, 2023 yılında Vision Pro adında yeni bir başlık çıkardı. Tim Cook, lansman sırasında ürünün devrim yaratacağını söyledi.'

    Çıktı:
    [
        ("Apple", "çıkardı", "Vision Pro"),
        ("Tim Cook", "CEO'su", "Apple"),
        ("Tim Cook", "bahsetti", "Vision Pro")
    ]
    ```

3.  **Grafiği İnşa Et:** Çıkarılan bu üçlüleri kullanarak bir `networkx.Graph` nesnesi oluşturun. Her bir üçlü, iki düğüm (Varlık1, Varlık2) ve onları birleştiren bir kenar (İlişki) olarak grafa eklenecektir.
4.  **Grafiği Kaydet:** Oluşturulan grafiği, her seferinde yeniden hesaplamamak için `graph.gpickle` formatında `db_sessions` klasörüne kaydedin.

### Adım 3.2: Agentic Router ve Araçların (Tools) Tanımlanması

Mevcut `processor.py` modülünü, sorguları yönetecek bir "ajan" içerecek şekilde güncelleyin veya `agent.py` adında yeni bir modül yaratın.

1.  **Araçları (Tools) Tanımla:** LangChain'in `Tool` konseptini kullanarak ajanın erişebileceği iki ana araç tanımlayın:
    * **`VectorSearchTool`:** Kullanıcının mevcut ChromaDB'ye anlamsal arama yapmasını sağlayan bir araç. Bu, mevcut arama fonksiyonlarınızı sarmalayacaktır.
    * **`GraphSearchTool`:** `graph_builder.py` ile oluşturulan ve kaydedilen bilgi grafiği üzerinde sorgulama yapacak yeni bir araç. Bu araç, "X'in Y ile ilişkisi nedir?" gibi sorgulara cevap vermek için `networkx` fonksiyonlarını kullanır.

2.  **Router'ı Oluştur (LLMChain):** LangChain kullanarak, kullanıcı sorgusunu alıp hangi aracın (`VectorSearchTool` veya `GraphSearchTool`) daha uygun olduğuna karar veren bir "router chain" oluşturun. Bu zincir, Llama 3.1'e sorgunun doğasını sorarak karar verir.

    *Örnek Router Prompt'u:*
    ```
    Kullanıcı sorgusuna göre en uygun aracı seçin.
    'vector_search': Bir kişi, olay veya kavram hakkında genel bilgi veya özet arandığında kullanılır. Örnek: 'Yapay zeka etiği hakkında bana bilgi ver.'
    'graph_search': İki veya daha fazla varlık arasındaki net ilişkiler sorulduğunda kullanılır. Örnek: 'Google, DeepMind'ı ne zaman satın aldı?'

    Sorgu: '{kullanici_sorgusu}'
    Seçilen Araç:
    ```

3.  **Agent'ı Yarat:** Tanımlanan araçlar ve router ile bir LangChain Agent (örneğin, `create_react_agent`) oluşturun. Bu ajan, router'dan gelen karara göre ilgili aracı çalıştıracak ve sonucunu alacaktır.

### Adım 3.3: Streamlit Arayüzünün Entegrasyonu

`streamlit_app.py` dosyasını, doğrudan ChromaDB'yi çağırmak yerine yeni oluşturulan bu ajanı çağıracak şekilde güncelleyin.

1.  Kullanıcıdan gelen sorguyu alın.
2.  Bu sorguyu `agent.invoke()` gibi bir komutla ajana gönderin.
3.  Ajanın çalışması sonucunda Llama 3.1 tarafından üretilen nihai cevabı ekranda gösterin.

---

## 4. Önerilen Yeni Dosya Yapısı

```
c2a/
├── db_sessions/
│   ├── [session_id]/
│   │   ├── chroma.sqlite3
│   │   └── knowledge_graph.gpickle  # <-- Yeni eklenecek grafik dosyası
├── docs/
├── src/
│   └── ars/
│       ├── __init__.py
│       ├── config.py
│       ├── manager.py
│       ├── processor.py             # <-- Agentic mantık buraya eklenebilir
│       ├── agent.py                 # <-- VEYA yeni bir dosyada tanımlanabilir
│       └── graph_builder.py         # <-- Yeni eklenecek modül
├── .gitignore
├── requirements.txt
└── streamlit_app.py                 # <-- Agent'ı çağıracak şekilde güncellenecek
```

Bu adımlar, projenizi modern ve güçlü bir Graph RAG mimarisine dönüştürmek için sağlam bir yol haritası sunmaktadır. Yapay zeka kodlayıcının bu plana göre hareket etmesi, başarılı bir entegrasyon sağlayacaktır.