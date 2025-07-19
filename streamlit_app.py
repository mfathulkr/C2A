import streamlit as st
import os
import shutil
from src.ars import config
from src.ars.processor import MediaProcessor
from src.ars.manager import AIManager
from src.ars import agent_factory

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Akıllı Raporlama Sistemi", layout="wide")

# --- Kaynakları ve Oturum Durumlarını Yükleme ---

# AIManager'ı ve MediaProcessor'ı oturum durumunda saklayarak yeniden oluşturulmasını engelliyoruz.
# AIManager artık LLM ve embedding modellerini kendi içinde başlatıyor.
if 'manager' not in st.session_state:
    with st.spinner("Yapay zeka modelleri ve kaynaklar hazırlanıyor..."):
        st.session_state.manager = AIManager()

if 'processor' not in st.session_state:
    st.session_state.processor = MediaProcessor()

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'report_chain' not in st.session_state:
    st.session_state.report_chain = None
    
if 'transcript' not in st.session_state:
    st.session_state.transcript = None

if 'chunks' not in st.session_state:
    st.session_state.chunks = None
    
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 'screen' durumunu oturumda sakla
if 'screen' not in st.session_state:
    st.session_state.screen = config.SCREEN_WELCOME

# --- Oturum Temizleme ---
def end_analysis_session():
    """Tüm oturum durumunu ve geçici dosyaları temizler."""
    try:
        # ChromaDB bağlantılarını kapat
        if 'agent_executor' in st.session_state:
            # Agent'ın vector store'unu temizle
            if hasattr(st.session_state.agent_executor, 'vectorstore'):
                try:
                    st.session_state.agent_executor.vectorstore._client.reset()
                except:
                    pass
        
        # AIManager'ın ChromaDB bağlantılarını temizle
        if 'manager' in st.session_state:
            try:
                # ChromaDB client'ını sıfırla
                import chromadb
                chromadb.PersistentClient(path=config.CHROMA_DB_PATH).reset()
            except:
                pass
        
        # Session state'i temizle
        st.session_state.clear()
        
        # Dosyaları sil (Windows için güvenli silme)
        if os.path.exists(config.SESSION_DATA_PATH):
            try:
                # Önce dosyaları kullanımdan çıkar
                import time
                time.sleep(1)  # Kısa bir bekleme
                
                # Güvenli silme işlemi
                def safe_remove(path):
                    try:
                        if os.path.isfile(path):
                            os.unlink(path)
                        elif os.path.isdir(path):
                            shutil.rmtree(path)
                    except PermissionError:
                        # Dosya kullanımdaysa atla
                        pass
                
                # Recursive olarak tüm dosyaları sil
                for root, dirs, files in os.walk(config.SESSION_DATA_PATH, topdown=False):
                    for file in files:
                        safe_remove(os.path.join(root, file))
                    for dir in dirs:
                        safe_remove(os.path.join(root, dir))
                
                # Ana klasörü sil
                safe_remove(config.SESSION_DATA_PATH)
                
            except Exception as e:
                st.warning(f"Bazı dosyalar silinemedi: {e}")
        
        # Screen'i welcome'a döndür
        st.session_state.screen = config.SCREEN_WELCOME
        
    except Exception as e:
        st.error(f"Oturum temizlenirken hata oluştu: {e}")
        st.session_state.screen = config.SCREEN_WELCOME

# --- Ana Uygulama ---

llm, embedding_model, media_processor, ai_manager = st.session_state.manager.llm, st.session_state.manager.embedding_model, st.session_state.processor, st.session_state.manager

# 1. KARŞILAMA EKRANI
if st.session_state.screen == config.SCREEN_WELCOME:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>ARS</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Akıllı Raporlama Sistemi</h3>", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Yeni Analiz Başlat", use_container_width=True, type="primary"):
            st.session_state.screen = config.SCREEN_SETUP
            st.rerun()

# 2. ANALİZ KURULUM EKRANI
elif st.session_state.screen == config.SCREEN_SETUP:
    st.title("1. Adım: Analiz Kaynağını Belirleyin")
    source_option = st.radio("Kaynak Tipi:", ["Bilgisayardan Dosya Yükle", "YouTube Linki Kullan"], horizontal=True, label_visibility="collapsed")
    
    uploaded_file = None
    youtube_url = None

    if source_option == "Bilgisayardan Dosya Yükle":
        uploaded_file = st.file_uploader("Desteklenen Formatlar: mp3, mp4, m4a, wav", type=["mp3", "mp4", "m4a", "wav"], label_visibility="collapsed")
    else:
        youtube_url = st.text_input("YouTube video linkini buraya yapıştırın:")

    st.write("---")

    if (uploaded_file or (youtube_url and ("youtube.com" in youtube_url or "youtu.be" in youtube_url))):
        if st.button("Analizi Başlat", type="primary", use_container_width=True):
            st.session_state.uploaded_file = uploaded_file
            st.session_state.youtube_url = youtube_url
            st.session_state.screen = config.SCREEN_PROCESSING
            st.rerun()
            
    if st.button("Geri"):
        st.session_state.screen = config.SCREEN_WELCOME
        st.rerun()

# 3. İŞLEME EKRANI
elif st.session_state.screen == config.SCREEN_PROCESSING:
    with st.status("Analiz süreci yürütülüyor...", expanded=True) as status:
        try:
            # Temizlik - Güvenli silme işlemi
            status.update(label="Eski analiz verileri temizleniyor...")
            if os.path.exists(config.SESSION_DATA_PATH):
                try:
                    # Önce ChromaDB bağlantılarını kapat
                    import chromadb
                    try:
                        chromadb.PersistentClient(path=config.CHROMA_DB_PATH).reset()
                    except:
                        pass
                    
                    # Kısa bir bekleme
                    import time
                    time.sleep(1)
                    
                    # Güvenli silme
                    def safe_remove(path):
                        try:
                            if os.path.isfile(path):
                                os.unlink(path)
                            elif os.path.isdir(path):
                                shutil.rmtree(path)
                        except PermissionError:
                            pass
                    
                    for root, dirs, files in os.walk(config.SESSION_DATA_PATH, topdown=False):
                        for file in files:
                            safe_remove(os.path.join(root, file))
                        for dir in dirs:
                            safe_remove(os.path.join(root, dir))
                    
                    safe_remove(config.SESSION_DATA_PATH)
                    
                except Exception as e:
                    st.warning(f"Bazı eski dosyalar silinemedi: {e}")
            
            os.makedirs(config.AUDIO_CACHE_PATH, exist_ok=True)
            
            # Kaynak işleme (İndirme veya Kaydetme)
            audio_path = ""
            uploaded_file = st.session_state.get("uploaded_file")
            youtube_url = st.session_state.get("youtube_url")

            if uploaded_file:
                status.update(label=f"'{uploaded_file.name}' dosyası kaydediliyor...")
                file_path = os.path.join(config.AUDIO_CACHE_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                audio_path = file_path
            elif youtube_url:
                status.update(label="YouTube'dan ses indiriliyor...")
                audio_path = media_processor.download_audio_from_youtube(youtube_url)

            # Transkripsiyon ve Hizalama
            status.update(label="Ses metne dönüştürülüyor (WhisperX)...")
            whisperx_result = media_processor.transcribe_and_align(audio_path)
            
            # Veritabanlarını Doldurma ve Metni Parçalama
            status.update(label="Metin işleniyor ve veritabanları oluşturuluyor...")
            # populate_databases şimdi parçalanmış dökümanları döndürecek
            st.session_state.chunks = ai_manager.populate_databases(whisperx_result) 

            # Agent ve Raporlama Zincirini Oluşturma
            status.update(label="Analiz araçları hazırlanıyor...")
            st.session_state.agent_executor = agent_factory.create_agent(llm, embedding_model)
            # Map-reduce zincirini graph nesnesiyle birlikte oluştur
            st.session_state.reporting_chain = agent_factory.create_map_reduce_chain(
                llm=llm, 
                graph=ai_manager.graph
            )
            st.session_state.messages = [{"role": "assistant", "content": "Analiz tamamlandı. Kayıt hakkında sorularınızı sorabilir veya bir rapor oluşturmasını isteyebilirsiniz."}]
            
            status.update(label="Analiz başarıyla tamamlandı!", state="complete")
            st.session_state.screen = config.SCREEN_ANALYSIS
            st.rerun()

        except Exception as e:
            st.error(f"Analiz sırasında bir hata oluştu: {e}")
            if st.button("Başa Dön"):
                end_analysis_session()
                st.rerun()

# 4. ANALİZ EKRANI (CHAT & RAPORLAMA)
elif st.session_state.screen == config.SCREEN_ANALYSIS:
    st.sidebar.title("Kontrol Paneli")
    if st.sidebar.button("Yeni Analiz Yap", use_container_width=True):
        end_analysis_session()
        st.rerun()
    
    st.title("Akıllı Raporlama Sistemi - Analiz Ekranı")

    with st.expander("Tutanak Raporu Oluşturucu", expanded=False):
        if st.button("Toplantı Tutanağı Oluştur", use_container_width=True):
            with st.spinner("Tutanak raporu hazırlanıyor (Bu işlem uzun sürebilir)..."):
                # Zincire, beklediği formatta {"input_documents": ...} bir sözlük veriyoruz.
                response = st.session_state.reporting_chain.invoke({"input_documents": st.session_state.chunks})
                # Zincir doğrudan metin (str) döndürdüğü için anahtar aramıyoruz.
                st.session_state.report = response
    
    if "report" in st.session_state:
        st.markdown("---")
        st.subheader("Oluşturulan Rapor")
        st.markdown(st.session_state.report)
        st.download_button("Raporu İndir (.md)", st.session_state.report, "tutanak_raporu.md", "text/markdown")
        
    st.markdown("---")
    st.subheader("Kayıt ile Sohbet Et")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                response = st.session_state.agent_executor.invoke({"input": prompt})
                st.markdown(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]}) 