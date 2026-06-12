import os
import glob
import time
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_text_splitters import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag import HateSpeechDetector

SOURCE_DIR = "./docs/knowledge_base"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "doc_base"

print("load rubert-base-cased-sentence ...")
embeddings = HuggingFaceEmbeddings(
    model_name="DeepPavlov/rubert-base-cased-sentence",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 400 и 50 - попробовать
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,        # max symbols in the chunk
    chunk_overlap=300,      # overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""] # Paragraph -> Line -> Word
)

#text_splitter = SemanticChunker(
#    embeddings,
#    breakpoint_threshold_type="percentile",
#    breakpoint_threshold_amount=95  # процент схожести
#)

detector = HateSpeechDetector()

start_time = time.perf_counter()

def extract_chapter_title(text, default_title):
    lines = [line.strip() for line in text[:5000].splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("Глава") and i + 1 < len(lines):
            for next_line in lines[i+1:i+5]:
                if next_line:
                    return next_line
    return default_title

if __name__ == "__main__":

    documents_to_store = []
    file_paths = glob.glob(os.path.join(SOURCE_DIR, "malicious.txt"))

    if not file_paths:
        print(f"No txt files in the dir '{SOURCE_DIR}'")
        exit()

    print(f"Number of found files : {len(file_paths)}")

    for file_path in file_paths:
        path_obj = Path(file_path)
        file_title = path_obj.stem

        print(f"processing file: {path_obj.name}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

        chapter_title = extract_chapter_title(full_text, default_title=path_obj.stem)
        print(f"-> title: '{chapter_title}'")

        chunks = text_splitter.split_text(full_text)

        for index, chunk_text in enumerate(chunks):
            #if not detector.is_safe(chunk_text, threshold=0.6):
            #    print(f"Чанк #{index} из файла {path_obj.name} заблокирован (обнаружен hate-speech)!")
            #    continue

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source_path": str(path_obj.resolve()),
                    "file_name": path_obj.name,
                    "title": chapter_title,
                    "chunk_id": f"{path_obj.stem}_{index}",
                    "position": index,
                    "part": path_obj.stem[0]
                }
            )
            documents_to_store.append(doc)

    print(f"\nTotal created: {len(documents_to_store)}")
    print(f"Start writing into ChromaDB ({CHROMA_DB_DIR})...")

    chunk_ids = [doc.metadata["chunk_id"] for doc in documents_to_store]

    db = Chroma.from_documents(
        documents=documents_to_store,
        ids=chunk_ids,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"} # косинусово сходство
    )

    execution_time = time.perf_counter() - start_time

    print(f"Success! All embeddings are saved. It takes {execution_time:.2f} с.")


