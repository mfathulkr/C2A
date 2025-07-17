# Akıllı Raporlama Sistemi (ARS) - Gelişmiş Graf ve Map-Reduce Entegrasyonu

Bu proje, bir YouTube bağlantısını veya yerel bir ses dosyasını analiz ederek, içeriğini metne dönüştüren ve bu metin üzerinden yapay zeka tabanlı, **hibrit (Vektör + Graf)** sorgulama ve **uzun metinler için özetleme/raporlama** imkanı sunan bir web uygulamasıdır. Tüm işlemler, veri gizliliğini ön planda tutarak kullanıcının kendi bilgisayarında yerel olarak çalışır.

## Teknoloji Mimarisi

- **Arayüz:** Streamlit
- **Arka Plan Mantığı:** Python
- **Video/Ses İndirme:** `yt-dlp`
- **STT (Sesi Metne Çevirme):** `openai-whisper`
- **Yapay Zeka (LLM & Embeddings):** `langchain`, `transformers` ve `sentence-transformers` (Hugging Face tabanlı)
- **Hibrit Veri Depolama:**
    - **Vektör Veritabanı:** `ChromaDB` (Anlamsal arama için)
    - **Graf Veritabanı:** `Neo4j` (İlişkisel sorgulama için)
- **Orkestrasyon & Mimariler:** `LangChain` (ReAct Agent, Map-Reduce Chain)

---

## Kurulum Adımları

Bu uygulamanın çalışması için bazı ön gereksinimler ve dikkatli bir kurulum süreci gerekmektedir. Lütfen adımları sırasıyla takip edin.

### 1. Ön Gereksinimler

- **NVIDIA Sürücüleri & CUDA (GPU Kullanıcıları için):** Eğer uygulamanın yapay zeka ve transkripsiyon hızından tam olarak faydalanmak istiyorsanız, NVIDIA ekran kartınızın güncel sürücülerinin ve [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)'in kurulu olduğundan emin olun. Bu, `torch` kütüphanesinin GPU'nuzu tanımasını sağlar ve işlemleri büyük ölçüde hızlandırır.
- **ffmpeg:** Whisper'ın çeşitli ses formatlarını işleyebilmesi ve `yt-dlp`'nin indirme sonrası ses formatı dönüşümü yapabilmesi için gereklidir.
  - **Windows (Chocolatey ile):** `choco install ffmpeg`
  - **Linux (apt ile):** `sudo apt-get install ffmpeg`
  - **macOS (Homebrew ile):** `brew install ffmpeg`
- **Neo4j Desktop:** Neo4j veritabanını yerel olarak çalıştırmak için [Neo4j Desktop](https://neo4j.com/download/)'ı kurun.
  1. Kurduktan sonra yeni bir proje oluşturun.
  2. Proje içinde "Add Database" seçeneği ile yeni bir veritabanı (DBMS) oluşturun.
  3. Oluşturduğunuz bu veritabanının **aktif (running)** durumda olduğundan emin olun.
  4. Bu veritabanını kurarken belirlediğiniz şifreyi, projedeki `src/ars/config.py` dosyasında bulunan `NEO4J_PASSWORD` değişkenine atayın. URI ve kullanıcı adı varsayılan (`neo4j://localhost:7687` ve `neo4j`) olarak bırakılmıştır.

### 2. Python Ortamı ve Bağımlılıklar

Projenin, kütüphane çakışmalarını önlemek için izole bir sanal ortamda çalıştırılması şiddetle tavsiye edilir.

1.  **Ortam Oluşturma ve Aktive Etme (Örnek: Conda):**
    ```bash
    conda create --name ars python=3.10 -y
    conda activate ars
    ```

2.  **Gerekli Kütüphaneleri Kurma:**
    Proje ana dizininde olduğunuzdan emin olarak, `requirements.txt` dosyasındaki tüm bağımlılıkları `pip` ile kurun:

    ```bash
    pip install -r requirements.txt
    ```
    *Not: `torch` kütüphanesinin kurulumu donanımınıza (CPU/GPU) göre farklılık gösterebilir. En iyi performans için PyTorch'un resmi sitesinden sisteminize uygun (özellikle CUDA versiyonunuza) `torch` kurulum komutunu alıp onu çalıştırmanız önerilir.*

---

## Uygulamayı Çalıştırma

Tüm kurulum adımları tamamlandıktan sonra, Python ortamınızın aktif olduğundan emin olun ve proje ana dizininde aşağıdaki komutu çalıştırın:

```bash
streamlit run streamlit_app.py
```

Bu komut, varsayılan web tarayıcınızda uygulamanın arayüzünü açacaktır.

---

## Teknik İş Akışı ve Kullanım

### Analiz Süreci
1.  **Girdi Sağlama:** Kenar çubuğundaki (sidebar) metin kutusuna bir YouTube linki yapıştırın veya bir ses/video dosyası yükleyin.
2.  **Analizi Başlatma:** `Yeni Analiz Başlat` butonuna tıklayın. Bu işlem, mevcut oturum verilerini (varsa) temizleyecek ve yeni bir analiz süreci başlatacaktır. Arka planda aşağıdaki adımlar otomatik olarak yürütülür:
    - **Ses Elde Etme (`processor.py`):** YouTube linki verilmişse `yt-dlp` ile en iyi kalitedeki ses indirilir. Dosya yüklenmişse doğrudan kullanılır.
    - **Transkripsiyon (`processor.py`):** `openai-whisper` modeli kullanılarak ses dosyası metne dönüştürülür.
    - **Veri İşleme ve Depolama (`manager.py`):**
        - Oluşturulan uzun metin, `LangChain` kullanılarak daha küçük, yönetilebilir parçalara (`chunks`) bölünür.
        - Her bir metin parçasından, bir LLM (Dil Modeli) aracılığıyla anlamsal ilişkileri temsil eden **bilgi üçlüleri** (Özne-İlişki-Nesne) çıkarılır.
        - Bu üçlüler temizlenir (boş değerler ayıklanır) ve Neo4j veritabanına `MERGE` sorgularıyla yüklenir, böylece bir bilgi grafiği (knowledge graph) oluşturulur.
        - Aynı metin parçaları, bir `embedding` modeli kullanılarak vektörleştirilir ve anlamsal arama için ChromaDB veritabanına yüklenir.

### Etkileşim Süreci
Analiz tamamlandığında arayüzde iki ana fonksiyon aktif hale gelir:

1.  **Rapor Oluşturma:**
    - **Ne işe yarar?** Tüm metnin genel bir özetini veya raporunu almak için kullanılır. Özellikle uzun (saatler süren) ses dosyaları için idealdir.
    - **Nasıl çalışır? (`agent_factory.py` - `create_map_reduce_chain`):** Bu özellik, `Map-Reduce` mimarisini kullanır.
        - **Map Adımı:** Metnin her bir parçası (chunk) ayrı ayrı LLM'e gönderilerek özetlenir.
        - **Reduce Adımı:** Elde edilen bu ara özetler birleştirilir ve nihai, tutarlı bir rapor oluşturmak için tekrar LLM'e gönderilir. Bu yöntem, LLM'lerin bağlam penceresi (context window) limitini aşma sorununu çözer.

2.  **Hibrit Agent ile Sohbet:**
    - **Ne işe yarar?** Metin içeriği hakkında spesifik sorular sormak için kullanılır. Agent, sorunuzun doğasına göre en uygun aracı kendi seçer.
    - **Nasıl çalışır? (`agent_factory.py` - `create_conversational_agent`):** Bu, `LangChain ReAct` (Reasoning and Acting) ajanıdır ve iki güçlü araca sahiptir:
        - **Graf Aracı (Neo4j):** "X ve Y arasındaki ilişki nedir?", "A projesinde kimler çalıştı?" gibi net, ilişkisel ve yapısal sorular için Neo4j graf veritabanını sorgular.
        - **Vektör Aracı (ChromaDB):** "Yapay zekanın etik sorunları hakkında ne gibi yorumlar yapıldı?" gibi anlamsal veya kavramsal sorular için ChromaDB'de vektör araması yapar.
    - Agent, sorduğunuz soruyu anlar, hangi aracın en doğru cevabı vereceğini planlar, o aracı kullanır ve gelen sonucu size anlamlı bir cevap olarak sunar. Bazen her iki araçtan gelen bilgiyi birleştirerek daha kapsamlı bir yanıt da oluşturabilir.

---

## Proje Mimarisi

Proje, görevleri mantıksal modüllere ayıran modüler bir yapıya sahiptir (`src/ars/` altında):

- `streamlit_app.py`: Kullanıcı arayüzünü oluşturan ve `manager` ile `agent_factory` modüllerini çağırarak tüm iş akışını yöneten ana betik.
- `config.py`: Tüm yapılandırma ayarlarının (veritabanı bağlantıları, model adları, dosya yolları, istemler/prompts) bulunduğu merkezi dosya.
- `llm_setup.py`: Hugging Face'den LLM ve embedding modellerini (örn: `sentence-transformers`) yükleyen ve yapılandıran modül.
- `processor.py`: `yt-dlp` ile ses indirme ve `openai-whisper` ile sesi metne dönüştürme (transkripsiyon) işlemlerini yürütür.
- `manager.py`: Transkripsiyon sonrası tüm veri işleme hattını yönetir. Metni parçalara ayırır, LLM kullanarak üçlüleri çıkartır ve hem Neo4j'yi hem de ChromaDB'yi doldurur.
- `agent_factory.py`: LangChain kullanarak uygulamanın iki ana zeka merkezini oluşturan modül:
    1.  Uzun metinler için `Map-Reduce` özetleme zinciri (`create_map_reduce_chain`).
    2.  Hibrit, araç-kullanabilen Q&A ajanı (`create_conversational_agent`). 