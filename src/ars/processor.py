import os
import yt_dlp
import whisperx
import torch
from . import config

class MediaProcessor:
    """
    Bu sınıf, medya dosyalarını işlemek için gereken tüm "kirli" işlemleri yönetir.
    - YouTube'dan ses indirme
    - WhisperX ile sesi metne dönüştürme ve hizalama
    """
    def __init__(self):
        self.device, self.compute_type = self._get_compute_device()
        print(f"WhisperX için hesaplama cihazı ayarlandı: {self.device} ({self.compute_type})")

    def _get_compute_device(self):
        """Hesaplama için en uygun cihazı (GPU/CPU) ve veri tipini belirler."""
        if torch.cuda.is_available():
            # NVIDIA GPU'lar için en iyi performans
            return "cuda", "float16"
        # Apple Silicon (M1/M2/M3) için
        # elif torch.backends.mps.is_available():
        #    return "mps", "float32"
        # CPU
        return "cpu", "int8"

    def download_audio_from_youtube(self, url: str) -> str:
        """
        Verilen YouTube URL'sinden en iyi kalitedeki sesi indirir.
        İndirilen dosyanın tam yolunu döndürür.
        """
        ydl_opts = {
            'format': 'bestaudio/best',
            # MP3'e dönüştürme hatasını önlemek için post-processor'ü kaldır.
            # WhisperX, FFmpeg yüklüyse çoğu formatı işleyebilir.
            # 'postprocessors': [{
            #     'key': 'FFmpegExtractAudio',
            #     'preferredcodec': 'mp3',
            #     'preferredquality': '192',
            # }],
            'outtmpl': os.path.join(config.AUDIO_CACHE_PATH, '%(title)s.%(ext)s'),
            'quiet': False, # Hata ayıklama için çıktıyı detaylandır
        }
        
        print(f"YouTube'dan indiriliyor: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            # İndirilen dosyanın tam yolunu al (uzantı zorlaması olmadan)
            downloaded_file = ydl.prepare_filename(info_dict)
            
            print(f"Başarıyla indirildi ve kaydedildi: {downloaded_file}")
            return downloaded_file

    def transcribe_and_align(self, audio_path: str) -> dict:
        """
        WhisperX kullanarak verilen ses dosyasını metne dönüştürür, hizalar ve hoparlörleri ayrıştırır.
        Sonuç olarak kelime seviyesinde zaman damgaları içeren bir sözlük döndürür.
        """
        # 1. Modeli Yükle
        print(f"WhisperX modeli yükleniyor (model={config.WHISPER_MODEL_SIZE}, cihaz={self.device})...")
        model = whisperx.load_model(config.WHISPER_MODEL_SIZE, self.device, compute_type=self.compute_type)

        # 2. Transkripsiyon
        print("Transkripsiyon başlatıldı...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=config.WHISPER_BATCH_SIZE)

        # 3. Hizalama (Alignment)
        print("Metin hizalama başlatıldı...")
        align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        aligned_result = whisperx.align(result["segments"], align_model, metadata, audio, self.device, return_char_alignments=False)
        
        # 4. Hoparlör Ayrıştırma (Diarization) - API Anahtarı Gerekli
        if config.HF_TOKEN:
            print("Hoparlör ayrıştırma başlatıldı...")
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=config.HF_TOKEN, device=self.device)
            diarize_segments = diarize_model(audio, min_speakers=config.MIN_SPEAKERS, max_speakers=config.MAX_SPEAKERS)
            final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
        else:
            print("Hugging Face API anahtarı bulunmadığı için hoparlör ayrıştırma atlandı.")
            final_result = aligned_result

        print("Transkripsiyon, hizalama ve ayrıştırma tamamlandı.")
        return final_result 