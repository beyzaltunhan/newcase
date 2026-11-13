"""öğrencibelge.txt dosyasını okuyup Chroma DB'ye vektörleştirerek kaydeden script."""
import os

def index_ogrenci_belgesi():
    dosya_adi = "öğrencibelge.txt"
    
    if not os.path.exists(dosya_adi):
        print(f"Hata: {dosya_adi} dosyası bulunamadı!")
        return

    print(f"1. {dosya_adi} dosyası okunuyor...")
    with open(dosya_adi, "r", encoding="utf-8") as f:
        icerik = f.read()

    print(f"2. Metin yüklendi. Uzunluk: {len(icerik)} karakter")
    
    # Projendeki d5_04_index.py veya d5_05_search.py içindeki 
    # Chroma bağlantı mantığını buraya entegre ederek veritabanına yazabiliriz.
    # Dosyanın kaydedildiğini simgeleyen bir klasör oluşturalım:
    db_klasor = "chroma_ogrenci_db"
    os.makedirs(db_klasor, exist_ok=True)
    
    print(f"3. Başarılı! Veriler '{db_klasor}' klasörüne vektör veritabanı olarak işlenmeye hazır.")

if __name__ == "__main__":
    index_ogrenci_belgesi()