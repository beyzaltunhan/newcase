from langchain_ollama import OllamaEmbeddings

# 1. Ollama üzerinden embedding modelini başlatıyoruz
embeddings = OllamaEmbeddings(
    model="embeddinggemma",
    base_url="http://localhost:11434"
)

# 2. Embedding yapmak istediğin yeni cümleyi buraya tırnak içinde yazabilirsin
cumle = "Veri bilimi ve yapay zeka projeleri üzerinde çalışıyorum."

# 3. Cümleyi vektöre dönüştürüyoruz
vector = embeddings.embed_query(cumle)

# 4. Sonuçları ekrana yazdırıyoruz
print("Seçilen Cümle:", cumle)
print("Vektör Boyutu:", len(vector))
print("İlk 5 Sayısal Değer:", [round(x, 4) for x in vector[:5]])

