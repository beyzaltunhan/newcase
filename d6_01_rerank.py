"""6. gün / Lab 1: Retrieve(k=8) -> lexical rerank(top_n=3)."""
from d5_05_search import search
from quality_utils import rerank_candidates, lexical_score

def retrieve_and_rerank(question, k=8, top_n=3, lexical_weight=0.4):
    # 1. Chroma veritabanından dense (anlamsal) adayları çek
    candidates = search(question, k=k)
    # 2. quality_utils ile lexical rerank (yeniden sıralama) yap
    ranked_results = rerank_candidates(question, candidates, top_n=top_n, lexical_weight=lexical_weight)
    return candidates, ranked_results

if __name__ == "__main__":
    # Makalene uygun gerçek bir soru yazmalısın
    q = "Makalede kullanılan veri bilimi, istatistiksel modelleme ve analiz yöntemleri nelerdir?"
    k_degeri = 8
    lexical_agirlik = 0.4  # Sabitlediğimiz lexical (kelime) ağırlığı
    
    print(f"Soru: {q}\n")
    print("=" * 75)
    print("CHROMA'DAN GELEN İLK HAM SIRALAMA (Dense Search):")
    print("=" * 75)
    
    ham_adaylar = search(q, k=k_degeri)
    for sira, doc in enumerate(ham_adaylar, 1):
        chunk_id = doc.metadata.get("id", f"chunk_{sira}")
        kaynak = doc.metadata.get("source", "makale.txt")
        print(f"Chroma Sırası: {sira:>2} | Chunk ID: {chunk_id} | Kaynak: {kaynak}")

    print("\n" + "=" * 75)
    print(f"LEXICAL RERANK SONRASI (Ağırlık: {lexical_agirlik}, En İyi 3 Sonuç):")
    print("=" * 75)
    
    _, ranked = retrieve_and_rerank(q, k=k_degeri, top_n=3, lexical_weight=lexical_agirlik)
    
    for i, (doc, total_skor) in enumerate(ranked, 1):
        lex_puan = lexical_score(q, doc.page_content)
        chunk_id = doc.metadata.get("id", "Bilinmiyor")
        
        # Sütun formatında düzenli çıktı
        print(f"[{i}. Sıra] Total Skor: {total_skor:.3f} | Lexical Puan: {lex_puan:.3f} | Chunk ID: {chunk_id}")
        print(f"          Metin Özet: {doc.page_content[:150].strip()}...\n")


