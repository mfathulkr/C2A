# Akıllı Raporlama Sistemi (ARS)

Bu proje, bir ses/video dosyasını veya YouTube linkini analiz ederek, içeriğini metne dönüştüren ve bu metin üzerinden yapay zeka tabanlı detaylı raporlama ve sohbet imkanı sunan bir web uygulamasıdır. Tüm işlemler, veri gizliliğini ön planda tutarak kullanıcının kendi bilgisayarında yerel olarak çalışır.

![Uygulama Arayüzü](docs/GUI.md)

## Teknoloji Mimarisi

- **Arayüz:** Streamlit
- **Arka Plan Mantığı:** Python
- **STT (Sesi Metne Çevirme):** `WhisperX` (Kelime-seviyesinde zaman damgası için)
- **Yapay Zeka (LLM & Embeddings):** `Ollama` (Llama 3 ile)
- **Vektör Veritabanı:** `ChromaDB`
- **Orkestrasyon:** `LangChain`

---

## Kurulum Adımları

Bu uygulamanın çalışması için bazı ön gereksinimler ve dikkatli bir kurulum süreci gerekmektedir. Lütfen adımları sırasıyla takip edin.

### 1. Ön Gereksinim: NVIDIA Sürücüleri (GPU Kullanıcıları için)

Eğer uygulamanın transkripsiyon hızından tam olarak faydalanmak istiyorsanız, NVIDIA ekran kartınızın güncel sürücülerinin kurulu olduğundan emin olun.

### 2. Ollama Kurulumu ve Modelleri

Uygulamanın yapay zeka yetenekleri için Ollama gereklidir.

1.  **Ollama'yı İndirin:** [Ollama'nın resmi web sitesine](https://ollama.com/) gidin ve işletim sisteminize uygun versiyonu indirip kurun.
2.  **Modelleri İndirin:** Kurulum tamamlandıktan sonra, terminali (veya PowerShell/CMD'yi) açın ve aşağıdaki komutları çalıştırarak gerekli yapay zeka modellerini bilgisayarınıza indirin:

    ```bash
    ollama pull llama3:8b
    ollama pull nomic-embed-text
    ```

    Bu işlem, modellerin boyutuna ve internet hızınıza bağlı olarak zaman alabilir.

### 3. Conda Ortamı ve Bağımlılıklar

Proje, izole bir Conda ortamında çalışacak şekilde tasarlanmıştır.

1.  **Conda'yı Kurun:** Eğer sisteminizde Conda kurulu değilse, [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (önerilen) veya Anaconda'yı kurun.

2.  **Conda Ortamını Oluşturun:**
    Terminalde proje ana dizinine gidin ve aşağıdaki komutla `ars` adında yeni bir Conda ortamı oluşturun:

    ```bash
    conda create --name ars python=3.10 -y
    ```

3.  **Ortamı Aktive Edin:**

    ```bash
    conda activate ars
    ```

4.  **GPU için Kritik Kütüphaneleri Kurun (PyTorch & CUDA):**
    WhisperX'in GPU üzerinde verimli çalışabilmesi için PyTorch'un doğru CUDA versiyonu ile kurulması çok önemlidir.
    - **NVIDIA GPU'nuzun desteklediği CUDA sürümünü kontrol edin (`nvidia-smi` komutu ile).**
    - PyTorch'un [resmi web sitesine](https://pytorch.org/get-started/locally/) gidin ve sisteminize uygun kurulum komutunu (genellikle `conda install ...` ile başlar) alın. Örneğin, CUDA 12.1 için komut genellikle şöyledir:
    
    ```bash
    # Örnek (Kendi CUDA sürümünüze göre bunu güncelleyin!)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    ```
    Bu komut, PyTorch ile birlikte doğru `cudatoolkit` ve `cudnn` versiyonlarını da kuracaktır. **Bu adımı atlamayın!**

5.  **Gerekli Diğer Kütüphaneleri Kurun:**
    Proje ana dizininde olduğunuzdan emin olarak, `requirements.txt` dosyasındaki diğer bağımlılıkları `pip` ile kurun:

    ```bash
    pip install -r requirements.txt
    ```

6.  **`ffmpeg` Kurulumu:**
    Ses dosyalarını işlemek için `ffmpeg` gereklidir. Bunu da Conda ile kolayca kurabilirsiniz:
    
    ```bash
    conda install -c conda-forge ffmpeg -y
    ```

---

## Uygulamayı Çalıştırma

Tüm kurulum adımları tamamlandıktan sonra, `ars` Conda ortamının aktif olduğundan emin olun ve proje ana dizininde aşağıdaki komutu çalıştırın:

```bash
streamlit run streamlit_app.py
```

Bu komut, varsayılan web tarayıcınızda uygulamanın arayüzünü açacaktır.

## Nasıl Kullanılır?

1.  **Kaynak Belirleme:**
    - **Dosya Yükle:** Bilgisayarınızdan bir ses veya video dosyası (`mp3`, `wav`, `mp4` vb.) seçin.
    - **YouTube Linki:** Analiz etmek istediğiniz videonun YouTube linkini yapıştırın.

2.  **Analizi Başlatma:**
    - `İşle ve Analize Başla` butonuna tıklayın. Bu işlem, kaynağın uzunluğuna ve donanımınızın gücüne (özellikle GPU) bağlı olarak birkaç dakika sürebilir. Arka planda ses metne dönüştürülür ve veritabanı oluşturulur.

3.  **Sonuçları İnceleme:**
    - İşlem tamamlandığında sayfanın alt kısmında iki ana bölüm belirir:
      - **📝 Raporlama:** `Detaylı Rapor Oluştur` butonu ile tüm metnin kapsamlı bir analizini alabilirsiniz. Rapor, içeriğe göre "Alınan Kararlar" veya "Ulaşılan Sonuçlar" gibi başlıkları akıllıca doldurur.
      - **💬 Döküman ile Sohbet Et:** Metin içeriği hakkında spesifik sorular sorabileceğiniz bir sohbet arayüzüdür. Örneğin: "Projenin bütçesi hakkında ne konuşuldu?" veya "Sam Altman'ın AGI hakkındaki temel argümanı neydi?".

4.  **Yeni Analiz:**
    - Farklı bir dosyayı analiz etmek için kenar çubuğundaki `Yeni Analiz Başlat` butonuna tıklayarak mevcut oturumu ve verileri temizleyebilirsiniz.

---

## Proje Mimarisi

Proje, görevleri mantıksal modüllere ayıran bir yapıya sahiptir:

- `streamlit_app.py`: Kullanıcı arayüzünü oluşturan ve iş akışını yöneten ana betik.
- `src/ars/config.py`: Tüm yapılandırma ayarlarının (model adları, prompt şablonları vb.) bulunduğu merkezi dosya.
- `src/ars/processor.py`: "Kirli" işlerden sorumlu modül. YouTube'dan indirme, dosya kaydetme ve `WhisperX` ile transkripsiyon işlemlerini yürütür.
- `src/ars/manager.py`: "Temiz" yapay zeka işlerinden sorumlu modül. `ChromaDB` ile vektör veritabanını yönetir, metinleri işler ve `LangChain` aracılığıyla raporlama ve sohbet zincirlerini oluşturur. 