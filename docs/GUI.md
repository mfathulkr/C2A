### **Yapay Zeka Geliştirici İçin Proje Görev Tanımı (Prompt)**

**Proje Adı:** Akıllı Raporlama Sistemi (ARS) - Streamlit Web Arayüzü

**Genel Hedef:** Mevcut komut satırı tabanlı ARS projesini, kullanıcı dostu, etkileşimli ve tek doküman odaklı bir Streamlit web uygulamasına dönüştür. Uygulama, kullanıcının bir ses/video kaynağı sunmasını, bu kaynağın işlenmesini beklemesini ve ardından aynı arayüz üzerinden hem sohbet (soru-cevap) yapabilmesini hem de detaylı bir rapor indirebilmesini sağlamalıdır.

-----

### **Faz 1: Proje Yapısının Yeniden Düzenlenmesi (Refactoring)**

Mevcut kod yapısı, tek bir `ars.py` dosyası etrafında şekillenmiştir. Web uygulaması geliştirmeden önce, kodun sürdürülebilirliğini ve test edilebilirliğini artırmak için aşağıdaki modern proje yapısını oluştur. Bu yapı, sorumlulukların ayrılması (Separation of Concerns) ilkesine dayanır.

**Hedef Proje Yapısı:**

```
/ars_streamlit_project/
|
|-- streamlit_app.py        # Ana Streamlit arayüz kodu
|
|-- src/
|   |-- ars/
|   |   |-- __init__.py         # Bu klasörün bir Python paketi olduğunu belirtir (boş dosya)
|   |   |-- processor.py      # Ses/video indirme, işleme ve transkripsiyon mantığı
|   |   |-- manager.py        # Veritabanı yönetimi, RAG ve Map-Reduce zincirleri
|   |   |-- config.py         # Mevcut konfigürasyon dosyası (model adları, ayarlar)
|
|-- /audio_cache/             # YouTube'dan indirilen veya yüklenen ses dosyaları için geçici önbellek
|
|-- /db_sessions/             # Her kullanıcı oturumu için oluşturulan geçici ChromaDB veritabanları
|
|-- requirements.txt          # Gerekli Python kütüphaneleri
|
|-- README.md                 # Proje açıklaması
```

**Yapılacaklar:**

1.  Yukarıdaki dosya ve klasör yapısını oluştur.
2.  Mevcut `config.py` dosyasını `src/ars/` altına taşı.
3.  Mevcut `ars.py` dosyasındaki mantığı, aşağıda detaylandırılacak olan `processor.py` ve `manager.py` dosyalarına ayır.

-----

### **Faz 2: Çekirdek Mantık Modüllerinin Geliştirilmesi (`processor.py` ve `manager.py`)**

**A. `src/ars/processor.py` Dosyası:**

Bu modül, "kirli işlerden" sorumludur: dosya indirme, kaydetme ve transkripsiyon.

  * **`AudioProcessor` adında bir sınıf oluştur.**
  * **`process_youtube_url(url: str) -> str` metodu:**
      * Parametre olarak bir YouTube URL'si alır.
      * `yt-dlp` kütüphanesini kullanarak videonun sesini indirir.
      * İndirilen ses dosyasını `./audio_cache/` klasörüne kaydeder.
      * Kaydedilen dosyanın tam yolunu (`str`) döndürür.
  * **`process_uploaded_file(uploaded_file) -> str` metodu:**
      * Streamlit'in `UploadedFile` nesnesini parametre olarak alır.
      * Bu dosyanın içeriğini `./audio_cache/` klasörüne yeni bir dosya olarak yazar.
      * Kaydedilen dosyanın tam yolunu (`str`) döndürür.
  * **`run_transcription(audio_path: str) -> list` metodu:**
      * Parametre olarak bir ses dosyası yolu alır.
      * Mevcut `ars.py` dosyasındaki `whisperX` mantığını kullanarak sesi metne çevirir. **Konuşmacı tespiti (`diarization`) bu versiyonda devre dışı bırakılacaktır** (CPU uyumluluğu ve basitlik için).
      * Zaman damgaları ve metin içeren segmentlerin bir listesini (`list`) döndürür.

**B. `src/ars/manager.py` Dosyası:**

Bu modül, "temiz" yapay zeka ve veritabanı işlerinden sorumludur. **Uygulamanın "tek doküman" mantığını burada kuracağız.**

  * **`VectorDBManager` adında bir sınıf oluştur.**
  * **`__init__(self, session_id: str)` metodu:**
      * Her kullanıcı oturumu için benzersiz bir `session_id` alır.
      * Bu ID'yi kullanarak `./db_sessions/{session_id}/` adında özel bir veritabanı yolu oluşturur. Bu, her dokümanın kendi izole veritabanında işlenmesini sağlar.
      * Bu özel yola sahip bir ChromaDB nesnesi başlatır.
  * **`ingest_segments(self, segments: list)` metodu:**
      * `processor.py`'dan gelen transkript segmentlerini alır.
      * Her segment için zengin `metadata`'ya sahip LangChain `Document` nesneleri oluşturur.
      * Bu dökümanları kendi oturumuna özel ChromaDB'ye ekler ve kalıcı hale getirir (`persist`).
  * **`get_rag_chain(self)` metodu:**
      * Soru-cevap (chat) için RAG zincirini kurar ve döndürür.
  * **`get_report_chain(self)` metodu:**
      * Toplantı raporu için Map-Reduce zincirini kurar ve döndürür.
  * **`clear_database(self)` metodu:**
      * Kendi oturumuna ait veritabanı klasörünü (`./db_sessions/{session_id}/`) diskten tamamen siler. Bu, yeni bir dosya işlenmeden önce çağrılacaktır.

-----

### **Faz 3: Streamlit Arayüzünün Geliştirilmesi (`streamlit_app.py`)**

Bu dosya, kullanıcı arayüzünü ve iş akışını yönetir. **Streamlit'in `st.session_state` özelliği, uygulamanın durumunu (hangi dosyanın işlendiği, sohbet geçmişi vb.) korumak için kritik öneme sahiptir.**

**Yapılacaklar:**

1.  **Başlangıç ve Durum Yönetimi:**
      * Uygulama ilk açıldığında `st.session_state` içinde `processing_complete`, `db_manager`, `messages` gibi anahtar kelimeleri `None` veya `False` olarak başlat.
2.  **Arayüz Düzeni:**
      * `st.title()` ve `st.markdown()` ile bir başlık ve açıklama ekle.
      * `st.columns(2)` ile ekranı ikiye böl:
          * **Sol Sütun:** `st.file_uploader()` ile bir dosya yükleme bileşeni oluştur. Sadece tek dosya yüklenmesine izin ver.
          * **Sağ Sütun:** `st.text_input()` ile bir YouTube link giriş kutusu oluştur.
      * Bu iki sütunun altına bir `st.button("İşle ve Analize Başla")` butonu koy.
3.  **İşleme Mantığı:**
      * Kullanıcı butona tıkladığında ve bir dosya veya link sağlandığında:
        1.  `st.spinner("Lütfen bekleyin... Ses dosyası işleniyor...")` ile kullanıcıya meşguliyet bildirimi göster.
        2.  `st.session_state`'te mevcut bir `db_manager` varsa, `db_manager.clear_database()` metodunu çağırarak eski veritabanını temizle.
        3.  `AudioProcessor`'ı kullanarak dosyayı indir/kaydet ve transkriptini oluştur.
        4.  Benzersiz bir `session_id` oluştur (örn: `uuid.uuid4().hex`).
        5.  Bu ID ile yeni bir `VectorDBManager` nesnesi oluştur ve `st.session_state.db_manager`'a ata.
        6.  Transkript segmentlerini `db_manager.ingest_segments()` ile veritabanına işle.
        7.  İşlem bitince `st.session_state.processing_complete = True` ve `st.session_state.messages = []` olarak ayarla. Uygulama arayüzü bu duruma göre kendini yeniden çizecektir.
4.  **Sohbet Arayüzü:**
      * Bu bölüm, sadece `if st.session_state.processing_complete:` koşulu doğru ise görünmelidir.
      * `st.expander("Sohbet Geçmişi")` içinde, `st.session_state.messages` listesindeki her mesajı `st.chat_message` ile göster.
      * `st.chat_input("Döküman hakkında bir soru sorun...")` ile kullanıcıdan yeni sorular al.
      * Yeni bir soru geldiğinde, `st.session_state.db_manager.get_rag_chain()`'i çağır, cevabı al, hem soruyu hem cevabı `st.session_state.messages`'a ekle ve arayüzü yeniden çiz.
5.  **Raporlama ve İndirme:**
      * Bu bölüm de sadece `if st.session_state.processing_complete:` doğru ise görünmelidir.
      * `st.button("Detaylı Rapor Oluştur")` butonu ekle.
      * Butona tıklandığında, `st.spinner` göster, `st.session_state.db_manager.get_report_chain()`'i çağır ve dönen rapor metnini `st.session_state.report_text`'e kaydet.
      * Eğer `st.session_state.report_text` dolu ise, `st.markdown(st.session_state.report_text)` ile raporu ekranda göster **VE** `st.download_button()` ile bu metnin `.md` dosyası olarak indirilmesini sağla.
