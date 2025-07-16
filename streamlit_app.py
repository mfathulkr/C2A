import streamlit as st
import os
import shutil
import uuid
from src.ars.processor import AudioProcessor
from src.ars.manager import VectorDBManager

# --- Uygulama Başlangıç Yapılandırması ---

# Gerekli klasörlerin var olduğundan emin ol.
# NOT: Artık başlangıçta klasörleri silmiyoruz. Temizlik işlemi
# kullanıcı tarafından tetiklenecektir.
AUDIO_CACHE_DIR = "audio_cache"
DB_SESSIONS_DIR = "db_sessions"
for dir_path in [AUDIO_CACHE_DIR, DB_SESSIONS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def reset_session():
    """
    Tüm oturum durumunu, önbelleği ve veritabanını temizler.
    Bu, "Yeni Analiz Başlat" düğmesi tarafından tetiklenir.
    """
    print("Oturum sıfırlanıyor...")
    
    # 1. Adım: Veritabanı yöneticisini güvenli bir şekilde kapat ve sil
    if 'db_manager' in st.session_state and st.session_state.db_manager:
        print("Mevcut veritabanı yöneticisi temizleniyor...")
        st.session_state.db_manager.clear_database() # Bu metod artık önce kaynakları serbest bırakıyor

    # 2. Adım: Ses önbelleğini temizle
    if os.path.exists(AUDIO_CACHE_DIR):
        shutil.rmtree(AUDIO_CACHE_DIR)
    os.makedirs(AUDIO_CACHE_DIR)

    # 3. Adım: Streamlit oturum durumunu temizle
    # 'db_manager' dahil tüm anahtarları sil
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]
    
    print("Oturum başarıyla sıfırlandı. Arayüz yeniden başlatılıyor.")
    st.rerun()

# --- Sayfa Yapılandırması ---
st.set_page_config(
    page_title="Akıllı Raporlama Sistemi (ARS)",
    page_icon="🤖",
    layout="wide"
)

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("🤖 Akıllı Raporlama Sistemi")
    st.markdown("""
        Bu uygulama, ses dosyalarınızı veya YouTube videolarınızı metne dönüştürür,
        içeriği özetler ve sorularınızı yanıtlar.
    """)
    
    # Yeni Analiz butonu
    st.button(
        "Yeni Analiz Başlat",
        on_click=reset_session,
        type="primary",
        use_container_width=True,
        help="Mevcut oturumu, sohbeti ve yüklenen dosyayı temizleyerek uygulamayı sıfırlar."
    )

    st.markdown("---")
    
    # Oturum durumu değişkenlerini, sadece tanımlı değillerse başlat.
    # Bu, sayfa yenilendiğinde durumun kaybolmasını önler.
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'report_text' not in st.session_state:
        st.session_state.report_text = ""
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = ""
    if 'source_info' not in st.session_state:
        st.session_state.source_info = ""


# --- Ana Başlık ve Açıklama ---
st.title("🤖 Akıllı Raporlama Sistemi (ARS)")
st.markdown("""
Bu uygulama, sağladığınız bir ses/video kaynağından otomatik olarak metin transkripsiyonu oluşturur, 
ardından bu metin üzerinden sorularınızı yanıtlar ve detaylı bir özet rapor sunar.

**Nasıl Kullanılır?**
1.  Aşağıdaki seçeneklerden birini kullanarak bir kaynak belirtin:
    *   **Dosya Yükle:** Bilgisayarınızdan bir ses veya video dosyası (`mp3`, `wav`, `mp4` vb.) seçin.
    *   **YouTube Linki:** Analiz etmek istediğiniz videonun YouTube linkini yapıştırın.
2.  `İşle ve Analize Başla` butonuna tıklayın. İşlem süresi, kaynağın uzunluğuna bağlıdır.
3.  İşlem tamamlandıktan sonra, alt kısımda belirecek olan sohbet arayüzünü ve raporlama seçeneklerini kullanın.
""")


# --- Girdi Alanları ---
with st.container(border=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Ses/Video Dosyası Yükleyin", 
            type=['mp3', 'mp4', 'm4a', 'wav', 'flac']
        )
    with col2:
        youtube_url = st.text_input("Veya YouTube Linki Yapıştırın")
    
    st.write("") # Boşluk için
    
    if st.button("İşle ve Analize Başla", type="primary", use_container_width=True):
        source_provided = uploaded_file is not None or (youtube_url and youtube_url.strip())
        
        if not source_provided:
            st.error("Lütfen bir dosya yükleyin veya bir YouTube linki girin.")
        else:
            with st.spinner("Lütfen bekleyin... Bu işlem kaynağın uzunluğuna göre zaman alabilir."):
                # 1. Önceki oturumu temizle
                if st.session_state.db_manager is not None:
                    st.session_state.db_manager.clear_database()
                
                # 2. Yeni oturum için başlangıç
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.processing_complete = False
                st.session_state.messages = []
                st.session_state.report_text = ""
                
                audio_processor = AudioProcessor()

                # 3. Kaynağı işle
                if uploaded_file:
                    st.session_state.audio_path = audio_processor.process_uploaded_file(uploaded_file)
                    st.session_state.source_info = uploaded_file.name
                elif youtube_url:
                    st.session_state.audio_path = audio_processor.process_youtube_url(youtube_url)
                    st.session_state.source_info = youtube_url
                
                # 4. Transkripsiyonu çalıştır
                segments = audio_processor.run_transcription(st.session_state.audio_path)
                
                if segments:
                    # 5. Veritabanını oluştur ve verileri işle
                    st.session_state.db_manager = VectorDBManager(st.session_state.session_id)
                    st.session_state.db_manager.ingest_segments(segments, os.path.basename(st.session_state.audio_path))
                    st.session_state.processing_complete = True
                    st.success("Dosya başarıyla işlendi! Artık soru sorabilir veya rapor oluşturabilirsiniz.")
                else:
                    st.error("Ses dosyasından metin çıkarılamadı. Lütfen dosyayı kontrol edin.")
            
            # Sayfayı yeniden çizerek sohbet/rapor alanlarını göster
            st.rerun()


# --- Analiz Sonrası Arayüz (Sadece işlem bittiyse görünür) ---
if st.session_state.get('processing_complete', False):
    st.divider()
    st.subheader(f"Analiz Edilen Kaynak: `{st.session_state.get('source_info', 'Bilinmiyor')}`")

    # --- Raporlama ve İndirme Bölümü ---
    with st.container(border=True):
        st.subheader("📝 Raporlama")
        
        if st.button("Detaylı Rapor Oluştur", use_container_width=True):
            with st.spinner("Rapor oluşturuluyor... (Bu işlem birkaç dakika sürebilir)"):
                if st.session_state.get('db_manager'):
                    st.session_state.report_text = st.session_state.db_manager.generate_report()
                else:
                    st.session_state.report_text = "Veritabanı yöneticisi bulunamadı. Lütfen tekrar deneyin."
        
        if st.session_state.get('report_text'):
            st.markdown(st.session_state.report_text)
            
            # Kaynak adından güvenli bir dosya adı oluştur
            source_name = st.session_state.get('source_info', 'kaynak')
            safe_filename = "".join([c for c in source_name if c.isalpha() or c.isdigit() or c in (' ', '-')]).rstrip()
            
            st.download_button(
                label="Raporu .md olarak İndir",
                data=st.session_state.report_text,
                file_name=f"report_{safe_filename}.md",
                mime="text/markdown",
                use_container_width=True
            )

    # --- Sohbet Arayüzü Bölümü ---
    with st.container(border=True):
        st.subheader("💬 Döküman ile Sohbet Et")

        # Sohbet geçmişini göster
        for message in st.session_state.get('messages', []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Kullanıcıdan yeni soru al
        if prompt := st.chat_input("Döküman hakkında bir soru sorun..."):
            st.session_state.setdefault('messages', []).append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI cevabını al ve göster
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    if st.session_state.get('db_manager'):
                        rag_chain = st.session_state.db_manager.get_rag_chain()
                        response = rag_chain.invoke(prompt)
                        st.markdown(response)
                    else:
                        response = "Sohbet başlatılamadı, veritabanı yöneticisi hazır değil."
                        st.markdown(response)
            
            # Cevabı sohbet geçmişine ekle
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun() # Sohbet sonrası arayüzü yenile 