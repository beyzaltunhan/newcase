"""
Oluşturduğumuz Chroma Veritabanından Soru Sorma (Retrieval) Testi
Bu kod, makalene soru sorarak en alakalı parçaları veritabanından çeker.
"""
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from config import ROOT

def veritabanindan_bilgi_cek():
    # 1. Kaydettiğimiz özel veritabanımızın klasör yolu
    db_klasoru = ROOT / "chroma_makale_db"
    
    print(f"Veritabanı yükleniyor: {db_klasoru}")
    
    # 2. Ollama Embedding modelini ve Chroma veritabanını bağlıyoruz
    embedding_model = OllamaEmbeddings(model="embeddinggemma")
    
    vectorstore = Chroma(
        persist_directory=str(db_klasoru),
        embedding_function=embedding_model
    )
    
    # 3. Makalene sormak istediğin soruyu buraya yazabilirsin (İstediğin gibi değiştirebilirsin)
    soru = "Which countries did the job postings come from?"
    print(f"\n[SORULAN SORU]: {soru}\n" + "-"*50)
    
    # 4. Veritabanında soruyla en çok eşleşen en yakın 2 parçayı buluyoruz (Similarity Search)
    ilgili_parcalar = vectorstore.similarity_search(soru, k=2)
    
    # 5. Sonuçları ekrana yazdıralım
    print(f"Bulunan en ilgili parça sayısı: {len(ilgili_parcalar)}\n")
    for i, parca in enumerate(ilgili_parcalar):
        print(f"[{i+1}. Bulunan Parça İçeriği]:\n{parca.page_content}\n" + "="*50)

if __name__ == "__main__":
    veritabanindan_bilgi_cek()