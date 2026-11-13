"""Makale dosyasını doğrudan okuyup Chroma veritabanına 'makale.txt' kaynağı ile kaydeder."""
import os
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from config import DB_DIR, COLLECTION
from d5_03_embeddings import embedding_client
from quality_utils import hit_at_k, rerank_candidates

def index_and_test_makale():
    dosya_adi = "makale.txt"
    
    if not os.path.exists(dosya_adi):
        print("Hata: 'makale.txt' dosyası ana dizinde bulunamadı!")
        return

    print("1. 'makale.txt' dosyası okunuyor...")
    with open(dosya_adi, "r", encoding="utf-8") as f:
        icerik = f.read()

    # LangChain Document nesnesine dönüştür
    doc = Document(page_content=icerik, metadata={"source": "makale.txt"})

    # Metni parçalara (chunks) böl
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents([doc])

    # Her chunka doğru metadata etiketlerini ekle
    for idx, chunk in enumerate(chunks, 1):
        chunk.metadata["source"] = "makale.txt"
        chunk.metadata["chunk_id"] = f"chunk_{idx}"

    print(f"Toplam {len(chunks)} parça oluşturuldu. Veritabanına kaydediliyor...")

    # Chroma veritabanına kaydet
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_client(),
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION
    )
    print("Başarılı! Makale veritabanına kaydedildi.\n")

    # --- TEST AŞAMASI ---
    print("=" * 75)
    print("MAKALE DÖKÜMANI RAG & RERANK TEST SONUÇLARI:")
    print("=" * 75)
    
    CASES = [
        ("Makalede kullanılan veri analizi ve modelleri nelerdir?", "makale.txt"),
        ("Araştırmada uygulanan istatistiksel yöntemler nasıl değerlendirilmiştir?", "makale.txt"),
        ("Çalışmanın kapsamı ve veri seti hakkında hangi bilgiler verilmektedir?", "makale.txt"),
    ]

    hits = top1 = 0
    from d5_05_search import search
    
    for query, expected in CASES:
        docs = search(query, k=8)
        hits += hit_at_k(docs[:3], expected)
        
        ranked = rerank_candidates(query, docs, top_n=3)
        found = ranked[0][0].metadata.get("source", "Bilinmiyor") if ranked else None
        
        top1 += int(found == expected)
        
        durum = "OK (Doğru Dosya)" if found == expected else "KONTROL (Farklı Dosya)"
        print(f"Soru: {query}")
        print(f" -> Durum: {durum} | Beklenen: {expected} | Bulunan Kaynak (Top1): {found}\n")

    n = len(CASES)
    print("=" * 75)
    print(f"Hit@3 (Beklenen kaynağın ilk 3'te olma başarısı): {hits}/{n}")
    print(f"Rerank Top1 isabet oranı: {top1}/{n}")
    print("=" * 75)

if __name__ == "__main__":
    index_and_test_makale() 