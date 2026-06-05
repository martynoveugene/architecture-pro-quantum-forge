import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "doc_base"

start_time = time.perf_counter()

embeddings = HuggingFaceEmbeddings(model_name="DeepPavlov/rubert-base-cased-sentence")

db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME
)

execution_time_ms = (time.perf_counter() - start_time) * 1000
print("Connections takes {execution_time_ms:.2f} мс.")

def search(query: str, filter: dict = None):

    print("\nTest query: "+query + "   filter: "+str(filter))
    start_time = time.perf_counter()

    results = db.similarity_search_with_score(
        query,
        k=3,
        filter=filter
    )

    execution_time_ms = (time.perf_counter() - start_time) * 1000

    for doc, score in results:
        print(f"Score: {score:.4f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content: {doc.page_content[:100].replace("\n\n", " ")}\n----------")

    print(f"Search takes {execution_time_ms:.2f} мс.")

if __name__ == "__main__":
    search("Василиса машина времени")
    print("success!")

    search("машина времени открыта")
    print("success!")

    search("машина времени открыта", {"part": "2"})
    print("success!")

