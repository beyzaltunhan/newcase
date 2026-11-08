"""5. gün / Lab 1: TXT + kaynak + birim metadata."""
import json
from langchain_core.documents import Document
from config import ROOT, DATA_DIR

def load_documents():
    # meta.json dosyasını güvenli bir şekilde okuyoruz
    meta_path = ROOT / "meta.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
            
        # Eğer meta.json içinde bu dosya adı yoksa KeyError almamak için güvenli kontrol ekledik
        file_meta = metadata.get(path.name, {"birim": "Genel", "kategori": "Makale"})
        meta = {"source": path.name, **file_meta}
        
        docs.append(Document(page_content=content, metadata=meta))
    return docs

if __name__ == "__main__":
    docs = load_documents()
    print("Belge sayısı:", len(docs))
    for d in docs:
        print(d.metadata.get("source"), "|", d.metadata.get("birim"), "|", len(d.page_content), "karakter")
        