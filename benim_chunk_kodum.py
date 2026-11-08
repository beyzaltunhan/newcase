"""
Kendi Ayarladığımız Chunk ve Overlap Lab Çalışması
Bu kod, makaleni belirlediğin karakter boyutunda ve kesişim payıyla parçalar.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DATA_DIR

def makaleyi_parcala():
   
    dosya_yolu = DATA_DIR / "makale.txt"
    
    if not dosya_yolu.exists():
        print(f"Hata: {dosya_yolu} bulunamadı! Lütfen dosya yolunu kontrol et.")
        return
    
    makale_metni = dosya_yolu.read_text(encoding="utf-8")
    print(f"Makale başarıyla okundu. Toplam karakter uzunluğu: {len(makale_metni)}")

    # 2. Adım: Chunk Size ve Overlap Değerlerini BURADAN KENDİN BELİRLE
    # İstediğin gibi bu sayıları değiştirip (Örn: chunk_size=800, chunk_overlap=150) test edebilirsin.
    SECilen_CHUNK_SIZE = 1000
    SECilen_OVERLAP = 100

    print(f"-> Ayarlanan Chunk Size: {SECilen_CHUNK_SIZE} | Chunk Overlap: {SECilen_OVERLAP}")

    # 3. Adım: Metni parçalama (chunking) işlemini gerçekleştir
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=SECilen_CHUNK_SIZE,
        chunk_overlap=SECilen_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    parcalar = text_splitter.create_documents([makale_metni])

    # 4. Adım: Sonuçları ekrana yazdır
    print(f"\n[SONUÇ] Toplam Parça (Chunk) Sayısı: {len(parcalar)}")
    print("-" * 50)
    
    # İlk 3 parçanın baş halini ekranda görelim
    for i, parca in enumerate(parcalar[:3]):
        print(f"Parça {i}: {parca.page_content[:150]}...\n")

if __name__ == "__main__":
    makaleyi_parcala()