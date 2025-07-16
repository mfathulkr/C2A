"""
Bu modül, ses işleme ile ilgili "kirli" işlerden sorumludur:
- YouTube'dan ses indirme
- Kullanıcının yüklediği dosyaları kaydetme
- Ses dosyalarını metne çevirme (transkripsiyon)
"""
import os
import whisperx
import torch
import gc
from . import config  # Göreceli import
import yt_dlp
import uuid
import glob

# Geçici olarak indirilen/yüklenen dosyaların saklanacağı klasör
AUDIO_CACHE_DIR = "audio_cache"

class AudioProcessor:
    def __init__(self):
        """AudioProcessor'ı başlatır."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"AudioProcessor: Cihaz '{self.device.upper()}' olarak ayarlandı.")
        # Ensure cache directory exists
        if not os.path.exists(AUDIO_CACHE_DIR):
            os.makedirs(AUDIO_CACHE_DIR)

    def process_youtube_url(self, url: str) -> str:
        """
        Verilen YouTube URL'sinden sesi indirir, MP3 olarak kaydeder ve
        dosya yolunu döndürür.
        """
        print(f"YouTube URL'si işleniyor: {url}")
        sanitized_title = str(uuid.uuid4()) # Benzersiz bir dosya adı oluştur
        
        # 'outtmpl' parametresine '%(ext)s' ekleyerek dosyanın doğru uzantıyla
        # kaydedilmesini sağlıyoruz.
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(AUDIO_CACHE_DIR, f'{sanitized_title}.%(ext)s'),
            'quiet': False
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Dosya uzantısı değişken olabileceğinden (m4a, webm, vb.), dosyayı glob ile bul.
        files = glob.glob(os.path.join(AUDIO_CACHE_DIR, f"{sanitized_title}.*"))
        if not files:
            raise FileNotFoundError("İndirilen YouTube ses dosyası bulunamadı.")
            
        output_path = files[0]
        print(f"Ses dosyası başarıyla indirildi: {output_path}")
        return output_path

    def process_uploaded_file(self, uploaded_file) -> str:
        """
        Kullanıcının yüklediği dosyayı geçici klasöre kaydeder ve yolunu döndürür.
        """
        print(f"Yüklenen dosya işleniyor: {uploaded_file.name}")
        file_path = os.path.join(AUDIO_CACHE_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        print(f"Dosya başarıyla kaydedildi: {file_path}")
        return file_path

    def run_transcription(self, audio_path: str) -> list:
        """
        Verilen ses dosyasını metne dönüştürür ve segment listesini döndürür.
        """
        print(f"Transkripsiyon başlatılıyor: {audio_path}")
        model = None
        model_a = None
        metadata = None
        # Cihaza göre en uygun hesaplama türünü seç
        compute_type = "float16" if self.device == "cuda" else "int8"
        print(f"Hesaplama türü: {compute_type}")
        
        try:
            # Modeli ve sesi yükle
            model = whisperx.load_model(
                config.WHISPER_MODEL_SIZE, 
                self.device, 
                compute_type=compute_type
            )
            audio = whisperx.load_audio(audio_path)

            # Transkripsiyon
            print("Ses dosyası metne çevriliyor... (Bu işlem zaman alabilir)")
            result = model.transcribe(audio, batch_size=config.BATCH_SIZE, verbose=True)

            # Transkripti hizalama
            print("Zaman damgaları hizalanıyor...")
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )
            aligned_result = whisperx.align(
                result["segments"], 
                model_a, 
                metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            print("Transkripsiyon ve hizalama tamamlandı.")
            return aligned_result["segments"]

        except Exception as e:
            print(f"Transkripsiyon sırasında bir hata oluştu: {e}")
            import traceback
            traceback.print_exc()
            return [] # Hata durumunda boş liste dön
        finally:
            # Belleği temizle
            print("Bellek temizleniyor...")
            del model
            del model_a
            del metadata
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print("Temizlik tamamlandı.") 