"""
Kendi Vektörleştirme (Embedding) ve İndeksleme Çalışmamız
Bu kod, az önce parçaladığımız metinleri Ollama üzerinden vektöre çevirir ve Chroma'ya kaydeder.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import DATA_DIR, ROOT

def embedding_ve_vektor_olustur():
    # 1. Makaleyi oku
    dosya_yolu = DATA_DIR / "makale.txt"
    makale_metni = dosya_yolu.read_text(encoding="utf-8")

    # 2. Aynı chunk ayarlarını burada da uygulayalım (Chunk Size: 1000, Overlap: 100)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    parcalar = text_splitter.create_documents([makale_metni])
    print(f"Toplam {len(parcalar)} parça hazırlandı. Vektörleştirme başlatılıyor...")

    # 3. Ollama Embedding modelini tanımla (embeddinggemma kullanıyoruz)
    embedding_model = OllamaEmbeddings(model="embeddinggemma")

    # 4. Parçaları Chroma veritabanına vektör olarak kaydet
    # Veritabanının kaydedileceği klasör adını kendi çalışman için belirleyebilirsin (örn: chroma_makale_db)
    db_klasoru = ROOT / "chroma_makale_db"
    
    vectorstore = Chroma.from_documents(
        documents=parcalar,
        embedding=embedding_model,
        persist_directory=str(db_klasoru)
    )

    print(f"\n[BAŞARILI] 84 parça başarıyla vektörleştirildi!")
    print(f"Vektör veritabanı şu klasöre kaydedildi: {db_klasoru}")

if __name__ == "__main__":
    embedding_ve_vektor_olustur()
    