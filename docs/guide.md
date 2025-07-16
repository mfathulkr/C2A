
Bu doküman, projenin son halini ve kullanımını açıklayan bir kılavuz niteliğindedir.

-----

### **Kullanım Kılavuzu: Yerel Akıllı Raporlama Sistemi (ARS) - v1.0**

**Proje Amacı:** Komut satırından çalışan, ses dosyalarını yerel olarak işleyip sorgulayan ve bu dosyalardan detaylı, yapılandırılmış toplantı raporları üreten bir araçtır.

-----

### **Bölüm 1: Proje Felsefesi ve Teknoloji Mimarisi**

Bu projenin temelinde **veri gizliliği** ve **modülerlik** yatar. Tüm ses işleme ve yapay zeka operasyonları, kullanıcının kendi makinesinde, tamamen açık kaynaklı teknolojilerle gerçekleştirilir. Hiçbir veri dışarıya gönderilmez.

**Kullanılan Teknolojiler ve Kütüphaneler:**

*   **Ortam Yönetimi:** **Conda** - Bağımlılıkları ve Python ortamını izole ve tutarlı bir şekilde yönetmek için.
*   **Orkestrasyon:** **LangChain** - RAG (Soru-Cevap) ve Map-Reduce (Raporlama) gibi karmaşık yapay zeka iş akışlarını yönetmek için.
*   **STT (Konuşmayı Metne Çevirme):** **WhisperX** - Standart Whisper'a göre daha doğru zaman damgaları sunan gelişmiş bir kütüphane.
*   **LLM & Embedding:** **Ollama** - Llama 3 gibi güçlü dil modellerini ve gömme modellerini yerel makinede çalıştırmak için.
*   **Vektör Veritabanı:** **ChromaDB** - Metin gömme (embedding) verilerini yerel olarak saklamak için kalıcı ve verimli bir vektör deposu.
*   **Komut Satırı Arayüzü:** **Argparse** - Python'un standart bir kütüphanesi olan, güçlü bir CLI aracı.
*   **Ses ve Veri İşleme:** **PyTorch** (CPU versiyonu), **Pandas**.

-----

### **Bölüm 2: Kurulum Adımları**

#### **Adım 2.1: Conda Ortamını Oluşturma**

Proje, `environment.yml` dosyası aracılığıyla tüm bağımlılıklarını yönetir. Ortamı kurmak ve aktive etmek için aşağıdaki komutları çalıştırın.

```bash
# Eğer daha önceden 'ars_env' adında bir ortam kurduysanız, önce onu kaldırın:
# conda env remove --name ars_env

# 1. environment.yml dosyasından yeni ortamı oluşturun:
conda env create -f environment.yml

# 2. Yeni oluşturulan ortamı aktive edin:
conda activate ars_env
```

Bu komutlar tamamlandığında, terminal satırınızın başında `(ars_env)` ifadesini göreceksiniz. Artık tüm işlemler bu izole ortamda gerçekleşecektir.

**`environment.yml` Dosyasının İçeriği:**
```yaml
# environment.yml
name: ars_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  # Temel Conda Paketleri
  - python=3.10
  - ffmpeg
  - pandas
  - git
  - pip
  # PyTorch (CPU Versiyonu)
  - pytorch
  - torchaudio
  - cpuonly 
  # Pip ile Kurulacaklar
  - pip:
    - langchain
    - langchain-core
    - langchain-chroma
    - langchain-ollama
    - ollama
    - git+https://github.com/m-bain/whisperX.git
    - soundfile
```

#### **Adım 2.2: Ollama Modellerini İndirme**

Ollama'nın çalıştığından emin olduktan sonra uygulama için gerekli olan modelleri indirin:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
```

-----

### **Bölüm 3: Proje Yapısı ve Kod Açıklamaları**

Uygulamanın dosya yapısı aşağıdaki gibidir:

```
/C2A/
|-- ars.py                  # Ana uygulama mantığı ve CLI kodları
|-- config.py               # Konfigürasyon ve prompt şablonları
|-- environment.yml         # Conda ortam tanımı
|-- /audio_files/           # Ses dosyalarınızı koyacağınız klasör
|-- /ars_db/                # ChromaDB verilerinin saklandığı klasör
|-- /docs/                  # Proje dokümantasyonu
```

#### **`config.py` - Ayarlar ve Şablonlar**
Bu dosya, projenin tüm yapılandırma ayarlarını ve dil modeline gönderilen talimat (prompt) şablonlarını içerir. `LLM_MODEL`, `EMBEDDING_MODEL`, veritabanı yolu gibi ayarlar buradan yönetilir.

#### **`ars.py` - Ana Uygulama Mantığı**
Bu dosya, `ARSManager` adında bir sınıf içerir ve uygulamanın tüm işlevselliğini yönetir.

*   **`ingest_audio(self, audio_file_path)` Metodu:**
    1.  Bir ses dosyasını alır.
    2.  `WhisperX` kullanarak sesi metne çevirir (`transcribe`). Bu işlem sırasında konsolda bir ilerleme çubuğu gösterilir.
    3.  Metnin zaman damgalarını hassaslaştırmak için hizalama (`align`) yapar.
    4.  Konuşmayı kronolojik segmentlere ayırır.
    5.  Her bir segmenti, kaynak dosya adı, başlangıç/bitiş zamanı ve raporlama için kritik olan bir **sıra numarası (`sequence_id`)** içeren bir metadata ile `Document` nesnesine dönüştürür.
    6.  Bu dökümanları, `OllamaEmbeddings` kullanarak vektörlere çevirir ve `ChromaDB` veritabanına kaydeder. Güncel ChromaDB versiyonu sayesinde veriler diske otomatik olarak kaydedilir, bu yüzden `.persist()` çağrısı kullanılmaz.

*   **`ask_question(self, question)` Metodu:**
    1.  Kullanıcıdan bir soru alır.
    2.  Bu soruyu kullanarak veritabanında anlamsal olarak en ilgili metin parçalarını bulur (`as_retriever`).
    3.  Bulunan metin parçalarını (kontekst) ve kullanıcının sorusunu, `config.py`'deki `RAG_PROMPT_TEMPLATE` şablonunu kullanarak `llama3:8b` modeline gönderir.
    4.  Modelden gelen cevabı ekrana yazdırır.

*   **`generate_report(self, audio_file_path)` Metodu:**
    1.  Raporu istenen ses dosyasının adını alır.
    2.  Veritabanından o dosyaya ait **tüm** metin segmentlerini, `sequence_id`'ye göre sıralanmış bir şekilde çeker.
    3.  LangChain'in `load_summarize_chain` fonksiyonunu `map_reduce` stratejisi ile kullanır:
        *   **Map Adımı:** Her bir metin segmentini ayrı ayrı `llama3:8b` modeline göndererek özetler.
        *   **Reduce Adımı:** Tüm bu bireysel özetleri birleştirir ve `config.py`'deki `COMBINE_PROMPT_TEMPLATE` şablonunda belirtilen yapılandırılmış (Genel Özet, Ana Konular vb.) nihai raporu oluşturur.
    4.  Raporu hem ekrana yazdırır hem de `.md` uzantılı bir dosyaya kaydeder.

-----

### **Bölüm 4: Kullanım Komutları**

`ars_env` ortamı aktifken, aşağıdaki komutları kullanarak uygulamayı çalıştırabilirsiniz:

1.  **Bir ses dosyasını veritabanına işleyin:**
    (Bu işlem, dosyanın uzunluğuna göre zaman alabilir.)
    ```bash
    python ars.py ingest --file ./audio_files/podcast.mp3
    ```

2.  **Veritabanına bir soru sorun:**
    ```bash
    python ars.py ask --question "Podcast'te hangi konulardan bahsedildi?"
    ```

3.  **İşlenmiş dosyadan bir rapor oluşturun:**
    ```bash
    python ars.py report --file ./audio_files/podcast.mp3
    ```
    Bu komutun sonunda `report_podcast.md` adında, belirttiğiniz formatta bir rapor dosyası oluşmalıdır.