import os
import glob
import time
import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import logging
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

current_date = datetime.now().strftime("%Y-%m-%d")
LOG_FILE_NAME = f"rag_update_{current_date}.log"

logger = logging.getLogger("RAG_Updater")
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler(LOG_FILE_NAME, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

SOURCE_DIR = "./docs/knowledge_base"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "doc_base"
STATE_FILE = "./chroma_db/file_state.json"

logger.info("Загрузка модели эмбедингов...")
embeddings = HuggingFaceEmbeddings(
    model_name="DeepPavlov/rubert-base-cased-sentence",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def send_log_email(log_filepath, status="Успех"):
    SMTP_SERVER = "smtp.yandex.ru"
    SMTP_PORT = 465
    SENDER_EMAIL = "your_email@yandex.ru"
    SENDER_PASSWORD = "your_password"
    RECIPIENT_EMAIL = "rag_admin@yandex.ru"

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = f"[{status}] Лог об обновлении документов RAG от {datetime.now().strftime('%d.%m.%Y')}"

    body = f"\nПроцесс обновления векторного индекса завершен со статусом: {status}.\nЛог во вложении."
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with open(log_filepath, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(log_filepath)}",
            )
            msg.attach(part)

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"Лог-файл успешно отправлен на почту {RECIPIENT_EMAIL}")
    except Exception as e:
        logger.error(f"Не удалось отправить лог на почту. Ошибка: {e}")



def extract_chapter_title(text, default_title):
    lines = [line.strip() for line in text[:5000].splitlines()]
    for i, line in enumerate(lines):
        if line.startswith("Глава") and i + 1 < len(lines):
            for next_line in lines[i+1:i+5]:
                if next_line:
                    return next_line
    return default_title

# Загрузка состояния файлов
def load_file_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.info(f"Ошибка чтения файла состояния, создаем новый: {e}")
    return {}

# Сохранение состояния файлов
def save_file_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    status = "Успех"
    try:
        start_time = time.perf_counter()
        logger.info(f"Сканирование источника: '{SOURCE_DIR}'")

        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            collection_metadata={"hnsw:space": "cosine"}
        )

        old_state = load_file_state()
        new_state = {}

        file_paths = glob.glob(os.path.join(SOURCE_DIR, "*.txt"))

        if not file_paths:
            logger.info(f"Файлы .txt не найдены в директории '{SOURCE_DIR}'")
            exit()

        files_to_delete_from_db = []
        documents_to_add = []
        chunk_ids_to_add = []

        logger.info(f"Найдено файлов: {len(file_paths)}")

        # Анализ изменения файлов
        for file_path in file_paths:
            path_obj = Path(file_path)
            file_name = path_obj.name

            # время последнего изменения файла
            mtime = os.path.getmtime(file_path)
            new_state[file_name] = mtime

            # изменился ли файл или появился новый
            if file_name not in old_state:
                logger.info(f"  [Новый файл] > {file_name}")
            elif old_state[file_name] != mtime:
                logger.info(f"  [Измененный файл] > {file_name} > требуется переиндексация")
                files_to_delete_from_db.append(file_name)
            else:
                # Файл не изменился, пропускаем
                continue

            # разбиваем на чанки только новые/измененные файлы
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            chapter_title = extract_chapter_title(full_text, default_title=path_obj.stem)
            chunks = text_splitter.split_text(full_text)

            for index, chunk_text in enumerate(chunks):
                chunk_id = f"{path_obj.stem}_{index}"
                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        "source_path": str(path_obj.resolve()),
                        "file_name": file_name,
                        "title": chapter_title,
                        "chunk_id": chunk_id,
                        "position": index,
                        "part": path_obj.stem[0]
                    }
                )
                documents_to_add.append(doc)
                chunk_ids_to_add.append(chunk_id)

        # удаленные файлы
        for old_file_name in old_state.keys():
            if old_file_name not in new_state:
                logger.info(f"  Удаленный файл > {old_file_name} > будет удален из БД")
                files_to_delete_from_db.append(old_file_name)

        # очистка чанков удаленных файлов из БД
        if files_to_delete_from_db:
            logger.info(f" Удаление чанков для файлов: {files_to_delete_from_db}")
            collection = db._collection

            for file_name_to_del in files_to_delete_from_db:
                results = collection.get(where={"file_name": file_name_to_del})
                ids_to_del = results.get("ids", [])
                if ids_to_del:
                    deleted_ids = collection.delete(ids=ids_to_del)
                    actual_deleted_count = len(deleted_ids) if deleted_ids is not None else 0
                    logger.info(f"  - Удалено чанков: {actual_deleted_count} для файла {file_name_to_del}")
                else:
                    logger.info(f"  - Чанки для файла {file_name_to_del} не найдены")

        # добавление новых документов
        if documents_to_add:
            logger.info(f" Добавление {len(documents_to_add)} новых чанков...")
            db.add_documents(documents=documents_to_add, ids=chunk_ids_to_add)
            logger.info(" Новые эмбеддинги успешно добавлены.")
        else:
            logger.info("Изменений не обнаружено.")

        save_file_state(new_state)

        execution_time = time.perf_counter() - start_time
        logger.info(f"[OK] Процесс обновления завершен. Время работы: {execution_time:.2f} с.")
        logger.info("="*60)

    except Exception as ex:
        status = "Ошибка"
        logger.critical(f"Ошибка при обновлении индекса: {ex}", exc_info=True)

    finally:
        logger.info("отправка отчета на электронную почту...")
        send_log_email(LOG_FILE_NAME, status=status)
