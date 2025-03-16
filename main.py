import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Inisialisasi stemmer Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r'\W+', ' ', text)  # Hapus tanda baca
    words = text.split()
    words = [stemmer.stem(word) for word in words]  # Stemming
    return ' '.join(words)

def search_tfidf(corpus, query, threshold=0.1):
    # Preprocessing corpus
    processed_corpus = [preprocess_text(doc) for doc in corpus]
    
    # Inisialisasi vectorizer
    vectorizer = TfidfVectorizer()
    
    # Konversi corpus menjadi representasi TF-IDF
    tfidf_matrix = vectorizer.fit_transform(processed_corpus)
    
    # Preprocessing query
    query = preprocess_text(query)
    
    # Transform query ke dalam representasi TF-IDF yang sama
    query_vector = vectorizer.transform([query])
    
    # Hitung kemiripan antara query dan setiap dokumen dalam corpus
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    
    # Urutkan hasil berdasarkan kemiripan
    similarity_scores = cosine_similarities[0]
    sorted_indices = similarity_scores.argsort()[::-1]
    
    # Tampilkan hasil pencarian
    results = []
    for idx in sorted_indices:
        if similarity_scores[idx] > threshold:
            results.append((similarity_scores[idx], corpus[idx]))
    
    return results

if __name__ == "__main__":
    corpus = [
        "Saya suka belajar pemrograman.",
        "Pemrograman dengan Python sangat menyenangkan.",
        "Belajar mesin pencari menggunakan TF-IDF.",
        "TF-IDF digunakan untuk pencarian teks."
    ]
    
    query = "belajar pemrograman dengan TF-IDF"
    results = search_tfidf(corpus, query)
    
    print("Query:", query)
    print("Hasil Pencarian:")
    for score, doc in results:
        print(f"Skor: {score:.4f} | Dokumen: {doc}")