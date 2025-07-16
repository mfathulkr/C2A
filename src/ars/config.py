"""
Proje için yapılandırma ayarları.
Bu dosyada, yollar, model adları, API anahtarları gibi sık sık değiştirilebilecek
veya ortamlar arasında farklılık gösterebilecek tüm sabitler merkezi olarak yönetilir.
"""
import os

# Projenin temel dizini
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ses dosyalarının bulunduğu klasör
AUDIO_FILES_DIR = os.path.join(BASE_DIR, "audio_files")

# --- Model Yapılandırmaları ---

# Özetleme ve Soru-Cevap için kullanılacak LLM modelinin adı.
# Bu modelin Ollama üzerinde çalışıyor olması gerekir.
# Örnek: "llama3:8b", "mistral", "gemma:7b"
LLM_MODEL = "llama3:8b"

# Metin gömme (embedding) için kullanılacak modelin adı.
# Bu modelin Ollama üzerinde çalışıyor olması gerekir.
# Örnek: "nomic-embed-text", "mxbai-embed-large"
EMBEDDING_MODEL = "nomic-embed-text"

# WhisperX için kullanılacak modelin boyutu.
# "large-v3" en doğru olanıdır, ancak daha fazla kaynak gerektirir.
# Diğer seçenekler: "base", "small", "medium", "large", "large-v2"
WHISPER_MODEL_SIZE = "large-v3"

# WhisperX modelinin çalışacağı hesaplama türü.
# GPU için: "float16" (önerilen) veya "float32"
# CPU için: "int8"
COMPUTE_TYPE = "int8"

# WhisperX transkripsiyon işlemi için toplu iş boyutu (batch size).
# Donanımınıza göre ayarlayın. GPU belleği yüksekse bu değeri artırabilirsiniz.
BATCH_SIZE = 4

# --- Veritabanı ve API Yapılandırmaları ---

# ChromaDB vektör veritabanının diskte saklanacağı klasör.
CHROMA_PERSIST_DIRECTORY = os.path.join(BASE_DIR, "ars_db")

# Ollama sunucusunun çalıştığı adres.
# Eğer Ollama aynı makinede çalışıyorsa bu ayarı değiştirmeyin.
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Prompt Şablonları ---

# Raporlama (map-reduce) için kullanılacak GÜNCELLENMİŞ ve ESNEK prompt'lar
MAP_PROMPT_TEMPLATE = """
Sen profesyonel bir içerik analistisin. Görevin, aşağıda verilen metin parçasını analiz etmektir.
Bu metin bir iş toplantısı, bir söyleşi, bir ders veya herhangi bir konuşma dökümü olabilir.
Lütfen metni dikkatlice oku ve aşağıdaki bilgileri yapılandırılmış ve **ayrıntılı** bir formatta çıkar:

- **Ana Konular ve Fikirler:** Bu bölümde tartışılan ana fikirleri, argümanları veya temaları **detaylandırarak** açıkla.
- **Önemli Detaylar:** Varsa bahsedilen kilit rakamları, isimleri, tarihler, projeler veya spesifik bilgileri **liste halinde ve açıklamalarıyla** sun.
- **Kararlar veya Sonuçlar:** Eğer metinde net bir karar belirtilmişse, onu yaz. Eğer yoksa, bu bölümde ulaşılan ana sonuçları, varılan fikir birliğini veya çıkarımları **kapsamlı bir şekilde** özetle.
- **Aksiyonlar veya Öneriler:** Eğer bir görev veya aksiyon tanımlandıysa ("yapılacak", "edilmesi gereken" gibi), onu bir madde olarak ekle. Eğer net bir aksiyon yoksa, metinde sunulan önerileri, gelecek adımları veya tavsiyeleri bu başlık altına **açıklamalarıyla birlikte** yaz.

Eğer bir başlık için metinde HİÇBİR ilgili bilgi (ne doğrudan ne de dolaylı) yoksa, o başlığı tamamen boş bırak. Cevabını **olabildiğince detaylı ve kapsamlı** oluştur, özetlemekten kaçın.

Metin:
---
{text}
---

Yapılandırılmış ve Detaylı Özet:
"""

COMBINE_PROMPT_TEMPLATE = """
Sen, bir dizi analiz özetinden nihai bir rapor oluşturan uzman bir rapor yazıcısısın.
Aşağıda, bir konuşmanın farklı bölümlerinden çıkarılmış yapılandırılmış özetler bulunmaktadır.
Görevin, bu parçaları birleştirerek **son derece detaylı, kapsamlı,** profesyonel ve iyi organize edilmiş bir rapor hazırlamaktır.
Raporun formatı, içeriğin doğasına (toplantı, söyleşi vb.) uyum sağlamalıdır. **Her bölümü olabildiğince ayrıntılı yazmaya özen göster.**

Raporun, aşağıdaki başlıkları içermeli ve her bölümü, sağlanan özetlerden sentezlediğin bilgilerle doldurmalısın.

# Detaylı Analiz Raporu

## 1. Genel Bakış
Konuşmanın ana amacını ve genel seyrini **en az 2-3 doyurucu paragrafta** özetle. Tartışılan en önemli konuları ve temaları, arka plan bilgisi vererek zenginleştir.

## 2. Tartışılan Ana Konular ve Temalar
Görüşülen her bir ana konuyu veya temayı bir alt başlık olarak listele. Her konunun altını, tartışmanın kilit noktaları, sunulan karşı argümanlar ve önemli detaylar ile **ayrıntılı bir şekilde** doldur.

## 3. Alınan Kararlar ve Ulaşılan Sonuçlar
Toplantı boyunca alınan tüm somut kararları madde madde sırala. Eğer net kararlar yoksa, konuşma boyunca varılan ana sonuçları, çıkarımları veya üzerinde anlaşılan fikirleri **paragraflar halinde detaylandırarak** bu bölümde özetle.

## 4. Aksiyon Maddeleri ve Öneriler
Belirlenen tüm görevleri, (varsa) kimin sorumlu olduğunu ve (varsa) tamamlanma tarihlerini içeren bir liste halinde sun. Eğer net aksiyonlar yoksa, konuşmada sunulan önemli önerileri, gelecek adımlar için tavsiyeleri veya potansiyel planları bu bölümde **her bir maddeyi açıklayarak** listele.

## 5. Öne Çıkan Diğer Notlar
Yukarıdaki kategorilere girmeyen ancak önemli olan diğer gözlemleri, fikirleri veya riskleri bu bölümde **ayrıntılı olarak** belirt.

Eğer bir bölüm için sağlanan özetlerde HİÇBİR bilgi yoksa, o bölümü raporuna dahil etme veya "Bu konuda not alınmamıştır." şeklinde belirt.

Özetlenmiş Bölümler:
---
{text}
---

YUKARIDAKİ BİLGİLERE GÖRE OLUŞTURULMUŞ PROFESYONEL VE **SON DERECE DETAYLI** RAPOR:
"""

# Soru-cevap (RAG) için kullanılacak prompt
RAG_PROMPT_TEMPLATE = """
Sen ses dökümanlarını analiz eden akıllı bir asistansın.
Görevin, sadece sağlanan bağlamı kullanarak kullanıcının sorusunu yanıtlamaktır.
Lütfen aşağıdaki kurallara uy:
1. Cevabını oluşturmak için sağlanan tüm bağlamı dikkatlice analiz et.
2. Soruyu, sorulduğu dilde yanıtla (Örneğin, soru Türkçe ise cevap da Türkçe olmalı).
3. Eğer soru genel bir soruysa (örn. "bu ne hakkında?", "özetle"), bağlamın kısa bir özetini sun.
4. Eğer cevap bağlamda bulunmuyorsa, sorunun diline göre "Bu soruya belgede bir cevap bulamadım." veya "I could not find an answer to that in the provided document." şeklinde yanıt ver. KESİNLİKLE bilgi uydurma.
5. Yardımsever ve profesyonel bir dil kullan.

Belgeden Alınan Bağlam:
---
{context}
---

Kullanıcının Sorusu: {question}

Cevap:
""" 