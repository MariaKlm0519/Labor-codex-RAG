import os
import re
import pickle
from typing import List, Dict
from docx import Document
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from settings import INDEX_PATH, EMBEDDING_MODEL_NAME, DOCX_PATH, CHUNK_MAX_LEN

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# загрузка документа
def load_docx(path: str) -> str:
    doc = Document(path)
    print(doc)
    return "\n".join([p.text for p in doc.paragraphs])

def split_document(path: str, chunk_size: int=CHUNK_MAX_LEN) -> List[Dict]:
    text = load_docx(path)
    results = split_hierarchy(text, level="part", max_chunk_len=chunk_size)
    for i, chunk in enumerate(results):
        chunk["metadata"]["uid"] = i
    return results


def split_hierarchy(text: str, level: str = "document", max_chunk_len: int = CHUNK_MAX_LEN) -> List[Dict]:
    """
    Рекурсивно парсит иерархию: Часть → Раздел → Глава → Статья.
    Возвращает список листьев с метаданными.
    """

    # Паттерны для каждого уровня
    patterns = {
        "part": r"^Часть\s+[а-я]+\s*$",
        "section": r"^Раздел\s+[IVXLC]+.*$",
        "chapter": r"^Глава\s+\d+.*$",
        "article": r"^Статья\s+\d+.*$",
    }

    # Определяем текущий уровень и следующий уровень вложенности
    level_order = ["part", "section", "chapter", "article"]

    if level not in level_order:
        chunks = split_into_chunks(text, max_chunk_len)
        return [{"text": chunk, "metadata": {}} for chunk in chunks]

    current_idx = level_order.index(level)
    next_level = level_order[current_idx + 1] if current_idx + 1 < len(level_order) else None

    sections = extract_between(text, patterns[level])

    if not sections:
        # Заголовков нет — либо спускаемся на уровень ниже, либо это лист
        if next_level:
            return split_hierarchy(text, next_level, max_chunk_len)
        else:
            chunks = split_into_chunks(text, max_chunk_len)
            return [{"text": chunk, "metadata": {}} for chunk in chunks]

    result = []
    for section in sections:
        metadata = {level: section["title"]}
        body = section["body"]

        if next_level and body:
            # Есть уровень вложенности — идem
            children = split_hierarchy(body, next_level, max_chunk_len)
            for child in children:
                # Объединяем метаданные: текущий уровень + родительские
                combined_meta = {**metadata, **child.get("metadata", {})}
                result.append({
                    "text": child["text"],
                    "metadata": combined_meta
                })
        else:
            # Нет вложенности, чанкуем текст
            chunks = split_into_chunks(body, max_chunk_len)
            for chunk in chunks:
                result.append({
                    "text": chunk,
                    "metadata": metadata
                })

    return result


def extract_between(text: str, pattern: str) -> List[Dict]:
    """
    Находит все заголовки по паттерну и возвращает список:
    [{"title": "Глава 1 ...", "body": "Текст до следующей главы"},]
    """
    # Ищем все заголовки с их позициями
    matches = []
    for m in re.finditer(pattern, text, re.MULTILINE):
        matches.append({
            "title": m.group(0),
            "start": m.start(),
            "end": m.end()
        })

    if not matches:
        return [{"title": None, "body": text.strip()}]

    result = []
    for i, m in enumerate(matches):
        start = m["end"]
        end = matches[i + 1]["start"] if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        result.append({
            "title": m["title"],
            "body": body
        })

    return result


def split_into_chunks(text: str, max_len: int = CHUNK_MAX_LEN) -> List[str]:
    chunks = []
    buffer = ""

    for p in re.split(r"\.\s*\n+", text):
        p = p.strip()
        if not p:
            continue
        p += "."

        if len(buffer) + len(p) < max_len:
            buffer += " " + p if buffer else p
        else:
            if buffer:
                chunks.append(buffer)
            buffer = p

    if buffer:
        chunks.append(buffer)

    return chunks

#эмбединги
def embed_texts(texts: List[str], model) -> np.ndarray:
    texts = [f"passage: {t}" for t in texts]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


def main():
    if os.path.exists(INDEX_PATH):
        logger.info("Найден индекс")
        return

    logger.info("Индекс не найден, идет пересчет...")
    chunks = split_document(DOCX_PATH)
    logger.info(f"Создано чанков: {len(chunks)}")

    logger.info("Загрузка модели...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    logger.info("Создание эмбеддингов...")
    embeddings = embed_texts([c["text"] for c in chunks], model)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)

    logger.info(f"Индекс сохранен: {INDEX_PATH}")


if __name__ == "__main__":
    main()