import os
from dotenv import load_dotenv

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# --- Genel Oturum Ayarları ---
SESSION_DATA_PATH = "session_data"
AUDIO_CACHE_PATH = os.path.join(SESSION_DATA_PATH, "audio_cache")
CHROMA_DB_PATH = os.path.join(SESSION_DATA_PATH, "chromadb")

# --- WhisperX Ayarları ---
WHISPER_MODEL_SIZE = "medium" # "base", "medium", "large-v2", "large-v3"
WHISPER_BATCH_SIZE = 8 # GPU'nuzun VRAM'ine göre ayarlayın. 16, 8GB VRAM için iyi bir başlangıçtır.

# --- Hoparlör Ayrıştırma (Diarization) Ayarları ---
# Bu özellik için Hugging Face'den bir API anahtarı (token) gereklidir.
# https://huggingface.co/settings/tokens adresinden ücretsiz alabilirsiniz.
# Anahtarınız yoksa bu alanı boş bırakın, hoparlör ayrıştırma atlanacaktır.
HF_TOKEN = os.getenv("HF_TOKEN", "")
MIN_SPEAKERS = 1
MAX_SPEAKERS = 5

# --- Neo4j Veritabanı Ayarları ---
NEO4J_URI = "neo4j://127.0.0.1:7687" 
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Ollama Ayarları ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# --- Streamlit Arayüz Ayarları ---
SCREEN_WELCOME = "welcome"
SCREEN_SETUP = "setup"
SCREEN_PROCESSING = "processing"
SCREEN_ANALYSIS = "analysis"
SCREEN_REPORT = "report"
SCREEN_CHAT = "chat"

# --- Prompt Şablonları ---
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

REPORT_PROMPT = """
Sen, karmaşık metinlerden son derece detaylı, analitik ve kapsamlı raporlar üreten kıdemli bir analiz uzmanısın. Sana sunulan metin özetlerini sentezleyerek, aşağıdaki şablona harfiyen uyan, tamamen TÜRKÇE, zengin içerikli ve derinlemesine bir rapor oluştur. Yorumlarını "verilen metinlere göre" gibi ifadelerle sınırlama, doğrudan analizi sun.

---
**Rapor Başlığı:** [Metnin ana temasını ve ruhunu yansıtan, profesyonel ve ilgi çekici bir başlık oluştur]

**1. Yönetici Özeti (Executive Summary):**
Metinlerde tartışılan konunun genel çerçevesini, ana argümanları, ortaya çıkan temel sonuçları ve genel gidişatı kapsayan, en az 150 kelimelik yoğun ve kapsamlı bir özet sun. Bu bölüm, raporun tamamını okumaya vakti olmayan bir yöneticinin dahi konuyu tüm kritik hatlarıyla eksiksiz anlayabilmesini sağlamalıdır.

**2. Tartışılan Ana Konular ve Derinlemesine Analizi:**
Konuşulan her bir ana konuyu bir başlık olarak belirt. Her başlığın altına, konunun tüm detaylarını içeren, en az 3-4 paragraftan oluşan bir analiz ekle. Bu analizde şunları mutlaka yap:
    - Konuyla ilgili sunulan farklı bakış açılarını ve argümanları karşılaştır.
    - Varsa, fikir ayrılıklarını veya çelişkili noktaları belirt.
    - Konunun önemini ve genel bağlamdaki yerini vurgula.
    - Satır aralarında kalan veya ima edilen düşünceleri ortaya çıkar.

    * **Konu Başlığı 1:**
        * [Detaylı analiz paragrafları...]

    * **Konu Başlığı 2:**
        * [Detaylı analiz paragrafları...]

**3. Alınan Stratejik Kararlar ve Sonuçları:**
Metinlerde açıkça ifade edilmiş veya üzerinde fikir birliğine varıldığı anlaşılan tüm kararları listele. Her kararın arkasındaki mantığı, hedeflenen sonucu ve olası etkilerini detaylı bir şekilde açıkla. Eğer net bir karar yoksa, "Görüşmeler sonucunda belirgin bir stratejik karar alınmamıştır, ancak şu seçenekler üzerinde durulmuştur..." şeklinde bir analiz yap.

**4. Belirlenen Eylem Adımları ve Sorumluluklar (Action Items):**
Gelecekte yapılması planlanan tüm görevleri ve eylem adımlarını listele. Her bir eylem adımının ne olduğunu, neden önemli olduğunu ve (varsa) kimin sorumlu olduğunu net bir şekilde ifade et. Görev tanımı yoksa, "Yapılacak spesifik bir görev veya eylem adımı belirlenmemiştir." yaz.

**5. Genel Değerlendirme ve Stratejik Çıkarımlar:**
Metinlerin genel tonunu (iyimser, karamsar, kararsız vb.), ortaya çıkan ana eğilimleri ve varılan nihai sonucu analiz et. Bu bölüm, basit bir özetin ötesine geçerek, görüşmelerin genel ruhundan ve satır aralarından stratejik çıkarımlar yapmalı ve geleceğe yönelik bir projeksiyon sunmalıdır. Bu bölüm, raporun en analitik kısmı olmalıdır.

**6. İlişkisel Analiz (Graph Veritabanından):**
Aşağıda, metindeki ana varlıklar (kişiler, projeler, konular vb.) arasındaki tespit edilen ilişkilerin bir özeti yer almaktadır. Bu bölüm, konular arasındaki gizli bağlantıları ve etkileşimleri anlamanıza yardımcı olur.
{graph_data}
---

GÖREV: Sana aşağıda "{text}" bloğu içinde, bir metnin farklı bölümlerinden çıkarılmış özetler sunulacak. Bu özetleri sadece tekrar listelemek veya basitçe çevirmek YASAKTIR. Görevin, bu özetlerdeki bilgileri SENTEZLEYEREK, yukarıdaki detaylı rapor şablonuna harfiyen uyan, bütüncül ve yeni bir rapor METNİ OLUŞTURMAKTIR.

İşte sentezlenecek metin özetleri:
{text}

---
BU RAPORU MUTLAKA TÜRKÇE OLUŞTUR.
"""

# Prompts
CHUNK_PROMPT_TEMPLATE = """
Aşağıdaki metin parçasını TÜRKÇE olarak özetle. Bu metin daha büyük bir transkriptin bir parçasıdır.
SADECE özet metnini döndür, başka hiçbir açıklama veya giriş cümlesi ekleme.

METİN:
"{text}"

ÖZET:
""" 