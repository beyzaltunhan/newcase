"""
Tokenizer ile Parçalama, Parçaları Ekrana Yazdırma, Embedding ve ChromaDB Insert Akışı
"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import ROOT

def tokenizer_ile_parcala_ve_kaydet():
    print("1. Makale dosyası aranıyor ve okunuyor...")
    
    dosya_yolu = ROOT / "data2" / "makale.txt"
    if not dosya_yolu.exists():
        dosya_yolu = Path("data2/makale.txt")

    if not dosya_yolu.exists():
        raise FileNotFoundError(f"'{dosya_yolu}' adında makale.txt bulunamadı! Lütfen yolu kontrol et.")

    loader = TextLoader(str(dosya_yolu), encoding="utf-8")
    docs = loader.load()

    print("2. Uygun Tokenizer (Hugging Face) yükleniyor ve metin parçalanıyor...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=500,     
        chunk_overlap=50    
    )
    chunks = text_splitter.split_documents(docs)
    print(f"-> Toplam {len(chunks)} token tabanlı parça oluşturuldu.\n")

    # --- HOCANIN İSTEDİĞİ YER: Parçaları ekrana yazdırıyoruz ---
    print("=" * 50)
    print(" OLUŞTURULAN METİN PARÇALARI (CHUNKS):")
    print("=" * 50)
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- [PARÇA {i} / {len(chunks)}] ---")
        print(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
    print("=" * 50 + "\n")
    # -----------------------------------------------------------

    print("3. embeddinggemma embedding modeli yükleniyor...")
    embedding_model = OllamaEmbeddings(model="embeddinggemma")

    print("4. Parçalar ChromaDB veritabanına ekleniyor (Insert)...")
    db_klasoru = ROOT / "chroma_makale_db"
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(db_klasoru)
    )
    sonuc = vectorstore.get(include=["embeddings", "documents"])
    print("Kayıtlı vektör sayısı:", len(sonuc["ids"]))
    print("Vektör boyutu:", len(sonuc["embeddings"][0]))
    print("İlk vektörün ilk 5 değeri:", sonuc["embeddings"][0][:5])
    
    print(f"BAŞARILI! Tokenizer ile bölünen veriler şu konuma kaydedildi: {db_klasoru}")

if __name__ == "__main__":
    tokenizer_ile_parcala_ve_kaydet()

   