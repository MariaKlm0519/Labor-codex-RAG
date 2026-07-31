import pickle
from collections import defaultdict
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from groq import Groq
from rank_bm25 import BM25Okapi
from settings import GROQ_API_KEY, INDEX_PATH, EMBEDDING_MODEL_NAME, GROQ_MODEL

ALPHA = 0.8   # вес cosine similarity

_client = Groq(api_key=GROQ_API_KEY)


def _normalize(scores: np.ndarray) -> np.ndarray:
    low, high = scores.min(), scores.max()
    if high == low:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(a, b.T)


class CodexRag:
    def __init__(self, index_path: str = INDEX_PATH, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

        with open(index_path, "rb") as f:
            index = pickle.load(f)
        self.chunks: List[Dict] = index["chunks"]
        self.embeddings: np.ndarray = index["embeddings"]

        tokenized = [c["text"].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def transform_query(self, user_query: str) -> str:
        """Переформулирует бытовой вопрос в юридическую терминологию ТК РФ,
        чтобы улучшить качество поиска. Используется ТОЛЬКО для retrieval,
        не для генерации финального ответа."""
        system_prompt = (
            "Пользователь задал вопрос по трудовому праву. "
    "Напиши короткий абзац (3-5 предложений) в стиле нормативного текста, "
    "который мог бы быть ответом на этот вопрос. "
    "Не ссылайся на конкретные статьи — ты их не знаешь. "
    "Пиши нейтрально, как справочный текст: факты, условия, термины. "
    "Никаких 'согласно законодательству', 'в соответствии с ТК' — только суть.\n\n"
    "Примеры:\n"
    "Вопрос: могут ли меня уволить пока я болею?\n"
    "Ответ: в период временной нетрудоспособности работника расторжение "
    "трудового договора по инициативе работодателя не допускается. "
    "Исключение составляют случаи ликвидации организации. "
    "После окончания периода нетрудоспособности ограничение снимается.\n\n"
    "Вопрос: сколько дней отпуска мне положено?\n"
    "Ответ: ежегодный основной оплачиваемый отпуск предоставляется работникам "
    "продолжительностью 28 календарных дней. Отдельным категориям работников "
    "предусмотрен удлинённый отпуск. Отпуск может быть разделён на части "
    "по соглашению сторон.\n"
        )
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            max_tokens=80,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def bm25_search(self, query: str, top_k: int = 10):
        tokenized = query.lower().split()
        raw_scores = self.bm25.get_scores(tokenized)
        norm_scores = _normalize(raw_scores)
        top_idx = np.argsort(norm_scores)[::-1][:top_k]
        return [(self.chunks[i], float(norm_scores[i])) for i in top_idx]

    def search(self, query: str = "", top_k: int = 10, final_k: int = 5,
            use_query_transform: bool = True) -> Dict:

        query_before = str(query)
        if use_query_transform:
            query = self.transform_query(query)

        query_emb = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]
        emb_scores = cosine_similarity(query_emb, self.embeddings)
        emb_norm = _normalize(emb_scores)
        emb_top_idx = np.argsort(emb_norm)[::-1][:top_k]
        emb_results = [(self.chunks[i], float(emb_norm[i])) for i in emb_top_idx]

        bm25_results = self.bm25_search(query, top_k=top_k)
        score_dict = {}

        for chunk, score in emb_results:
            uid = chunk["metadata"]["uid"]
            score_dict[uid] = {"chunk": chunk, "score": ALPHA * score}

        for chunk, score in bm25_results:
            uid = chunk["metadata"]["uid"]
            if uid in score_dict:
                score_dict[uid]["score"] += (1 - ALPHA) * score
            else:
                score_dict[uid] = {"chunk": chunk, "score": (1 - ALPHA) * score}

        combined = [(v["chunk"], v["score"]) for v in score_dict.values()]

        # === АГРЕГАЦИЯ ПО СТАТЬЯМ ===
        article_scores = defaultdict(list)
        article_examples = {}

        for chunk, score in combined:
            article = chunk["metadata"]["article"]
            article_scores[article].append(score)
            if article not in article_examples or score > article_examples[article]["score"]:
                article_examples[article] = {
                    "text": chunk["text"],
                    "meta": chunk,
                    "score": score,
                }

        results = [
            {
                "article": article,
                "score": max(scores_list),
                "text": article_examples[article]["text"],
                "meta": article_examples[article]["meta"],
            }
            for article, scores_list in article_scores.items()
        ]

        results.sort(key=lambda r: r["score"], reverse=True)

        return {
            "original_query": query_before,
            "transformed_query": query,
            "results": results[:final_k],
        }


def ask_llm(user_prompt: str) -> str:
    system_prompt = f"""
    Ты юридический ассистент по Трудовому кодексу РФ.
ЗАДАЧА: дать точный ответ на вопрос на основе приведённых статей.
ТРЕБОВАНИЯ:
- Если информация распределена по нескольким статьям — объедини их.
- Не придумывай ничего от себя.
- Используй формулировки максимально близкие к тексту закона.
- Если информация распределена по нескольким статьям — объедини их.
- Не делай выводов, которых нет в тексте.
- В конце укажи статьи в формате: (статья XXX, статья YYY ТК РФ).
"""
    response = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ],
        max_tokens=400,
        temperature=0.4
    )
    return response.choices[0].message.content

def build_prompt(query: str, results: List[Dict]) -> str:
    context = "\n\n".join([f"{r['article']}: {r['text']}" for r in results])
    return f"""КОНТЕКСТ:{context}
    ВОПРОС:{query}"""

def main():
    rag = CodexRag()
    while True:
        print("Введите вопрос:")
        query = input("> ")

        output = rag.search(query=query)
        print(f"\n[Query Transform]\n  Исходный:       {output['original_query']}")
        print(f"  Преобразованный: {output['transformed_query']}")

        print("\nРелевантные статьи:\n")
        for r in output["results"]:
            print("-" * 80)
            print(f"{r['article']}")
            print(f"Score: {r['score']:.4f}")
            print(f"Chunk: {r['text'][:400]}")
            print()

        prompt = build_prompt(output["original_query"], output["results"])
        answer = ask_llm(prompt)
        print("\n" + "-" * 80 + "\n")
        print("Ответ LLM:")
        print(answer)

if __name__ == "__main__":
    main()