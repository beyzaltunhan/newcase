from pathlib import Path
from pypdf import PdfReader

pdf_yolu = Path("makalem.pdf")
cikis_yolu = Path("makale.txt")   # Rag Kodlar klasörüne kaydedilir

# 1) PDF'i aç
reader = PdfReader(pdf_yolu)
print("Sayfa sayısı:", len(reader.pages))

# 2) Her sayfadan metni çıkar
sayfalar = []
for sayfa in reader.pages:
    metin = sayfa.extract_text() or ""
    sayfalar.append(metin)

# 3) Sayfaları birleştir
tam_metin = "\n\n".join(sayfalar)
print("Toplam karakter sayısı:", len(tam_metin))

# 4) utf-8 olarak kaydet
cikis_yolu.write_text(tam_metin, encoding="utf-8")
print("Kaydedildi:", cikis_yolu.resolve())

# 5) Başını göster
print(tam_metin[:1000])