"""
Kendi Makalem İçin Retrieve (k=8) -> Rerank (top_n=3) Akışı
"""
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import ROOT

# 1. Kendi makale veritabanımıza bağlanma fonksiyonu
def get_makale_store():
    db_klasoru = ROOT / "chroma_makale_db"
    if not db_klasoru.exists():
        raise FileNotFoundError("Önce makaleni veritabanına indeksleyen kodunu çalıştırmalısın!")
    
    # Kendi embedding modelimizle Chroma veritabanımızı bağlıyoruz
    embedding_model = OllamaEmbeddings(model="embeddinggemma")
    return Chroma(
        persist_directory=str(db_klasoru),
        embedding_function=embedding_model
    )

# 2. Arama fonksiyonu (Veritabanından k adet parça çeker)
def makale_search(question, k=8):
    store = get_makale_store()
    return store.similarity_search(query=question, k=k)

# 3. Basit bir kelime/skor bazlı rerank simülasyonu (veya FlashRank entegrasyonu)
def rerank_candidates(question, candidates, top_n=3):
    # Soru kelimelerini küçük harfe çevirip anahtar kelime havuzu oluşturuyoruz
    query_words = set(question.lower().split())
    
    scored_candidates = []
    for doc in candidates:
        text = doc.page_content.lower()
        # Parçanın içinde soruyla eşleşen kelime sayısını bulup basit bir skor üretiyoruz
        score = sum(1 for word in query_words if word in text)
        scored_candidates.append((doc, score))
    
    # Skora göre büyükten küçüğe sıralayıp en iyi top_n tanesini alıyoruz
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates[:top_n]

# 4. Ana Çalıştırma Alanı
if __name__ == "__main__":
    # Kendi makalene uygun ingilizce veya türkçe bir soru yazabilirsin
    q = "What are the essential soft skills for data scientists?"
    
    print(f"[SORU]: {q}\n" + "-"*50)
    
    # Önce veritabanından en ilgili 8 adayı çekiyoruz (Retrieve)
    candidates = makale_search(q, k=8)
    print(f"Veritabanından çekilen ham parça sayısı: {len(candidates)}")
    
    # Rerank işleminden geçirip en iyi 3 tanesini süzüyoruz
    ranked = rerank_candidates(q, candidates, top_n=3)
    
    print(f"\nRerank Sonrası Süzülen En İyi 3 Parça:\n" + "="*50)
    for i, (doc, score) in enumerate(ranked, 1):
        source_name = doc.metadata.get("source", "makale.txt")
        print(f"\n[{i}. En İyi Parça] (Skor: {score}) | Kaynak: {source_name}")
        print(f"İçerik:\n{doc.page_content}\n" + "-"*40)

        