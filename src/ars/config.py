import os

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
HF_TOKEN = "" 
MIN_SPEAKERS = 1
MAX_SPEAKERS = 5

# --- Neo4j Veritabanı Ayarları ---
NEO4J_URI = "neo4j://127.0.0.1:7687" 
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "12345678"

# --- Ollama Ayarları ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# --- Prompt Şablonları ---
REPORT_PROMPT = """
Sen, metin analizinden profesyonel ve detaylı raporlar üreten uzman bir asistansın. Sana sağlanan metin parçalarını derinlemesine analiz ederek, aşağıdaki şablona harfiyen uygun, tamamen TÜRKÇE ve zengin içerikli bir rapor oluştur. Kesinlikle kendi yorumlarını veya "Sana verilen metinlere göre..." gibi giriş cümlelerini ekleme. Sadece rapor başlıklarını ve istenen içeriği yaz.

---
**Rapor Başlığı:** [Metnin ana temasını yansıtan yaratıcı ve açıklayıcı bir başlık belirle]

**1. Yönetici Özeti (Executive Summary):**
Metinlerde tartışılan ana fikirleri, temel argümanları ve genel gidişatı kapsayan, en az 5-6 cümlelik kapsamlı bir özet sun. Bu bölüm, raporun tamamını okumaya vakti olmayan bir yöneticinin dahi konuyu tüm ana hatlarıyla anlayabilmesini sağlamalıdır.

**2. Katılımcılar ve Rolleri:**
Metinlerde adı geçen veya fikir beyan eden tüm kişi veya grupları listele. Eğer rolleri veya pozisyonları belirtilmişse, bunları da ekle. Eğer belirtilmemişse, "Katılımcı bilgisi bulunmamaktadır." yaz.

**3. Tartışılan Ana Konular ve Detayları:**
Konuşulan her bir ana konuyu bir başlık olarak belirt ve altına en az 3-4 maddelik detaylı açıklamalar ekle. Her konunun kilit noktalarını, sunulan farklı bakış açılarını ve önemli argümanları bu bölümde açıkça belirt.

    *   **Konu Başlığı 1:**
        *   Detay 1
        *   Detay 2
        *   Detay 3

    *   **Konu Başlığı 2:**
        *   Detay 1
        *   Detay 2
        *   Detay 3

**4. Alınan Kararlar ve Gerekçeleri:**
Metinlerde açıkça ifade edilmiş veya üzerinde fikir birliğine varıldığı anlaşılan tüm kararları listele. Mümkünse, bu kararların alınma gerekçelerini de kısaca açıkla. Eğer net bir karar yoksa, "Görüşmeler sonucunda belirgin bir karar alınmamıştır." yaz.

**5. Belirlenen Eylem Adımları (Action Items):**
Gelecekte yapılması planlanan, sorumlusu olan veya olmayan tüm görevleri ve eylem adımlarını listele. Her bir eylem adımının ne olduğunu net bir şekilde ifade et. Eğer bir görev tanımı yoksa, "Yapılacak bir görev veya eylem adımı belirlenmemiştir." yaz.

**6. Genel Değerlendirme ve Sonuç:**
Metinlerin genel tonunu (iyimser, karamsar, nötr vb.), ortaya çıkan ana eğilimleri ve varılan nihai sonucu analiz et. Bu bölüm, metinlerin satır aralarındaki anlamı ve genel ruhunu yansıtmalıdır.
---

İşte analiz edilecek metinler:
{text}
""" 

# Prompts
CHUNK_PROMPT_TEMPLATE = """
Aşağıdaki metin parçasını TÜRKÇE olarak özetle. Bu metin daha büyük bir transkriptin bir parçasıdır.
SADECE özet metnini döndür, başka hiçbir açıklama veya giriş cümlesi ekleme.

METİN:
"{text}"

ÖZET:
"""

FINAL_REPORT_PROMPT_TEMPLATE = """
""" 