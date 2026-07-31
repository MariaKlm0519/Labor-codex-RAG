"""
Метрики:
Hit Rate @K — нужная статья находится в топ-к результатов
MRR @K — Mean Reciprocal Rank: насколько высоко стоит нужная статья
"""

import re
import json
from typing import List, Dict
from settings import TEST_PATH
from rag_query import CodexRag, ask_llm, build_prompt

_ARTICLE_NUM_RE = re.compile(r"\d+")

def _article_number(value: str) -> str:
    match = _ARTICLE_NUM_RE.search(value)
    return match.group(0) if match else value

def load_test_set(path=TEST_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def hit_at_k(results: List[Dict], expected: str, k: int) -> bool:
    expected_num = _article_number(expected)
    return any(_article_number(r["article"]) == expected_num for r in results[:k])


def reciprocal_rank(results: List[Dict], expected: str, k: int) -> float:
    expected_num = _article_number(expected)
    for rank, r in enumerate(results[:k], start=1):
        if _article_number(r["article"]) == expected_num:
            return 1.0 / rank
    return 0.0


def evaluate(codex: CodexRag, k: int = 5, use_transform: bool = True) -> None:
    hits, rr_scores = [], []
    test_set = load_test_set()

    print(f"RAG Benchmark  |  K={k}  |  Query Transform: {use_transform}")
    print(f"{'=' * 70}\n")

    for i, item in enumerate(test_set, start=1):
        query = item["query"]
        expected = item["expected_article"]

        output = codex.search(query=query, final_k=k, use_query_transform=use_transform)
        results = output["results"]

        hit = hit_at_k(results, expected, k)
        rr = reciprocal_rank(results, expected, k)

        hits.append(hit)
        rr_scores.append(rr)

        status = "[Успех]" if hit else "[Нет]"
        print(f"[{i:02d}] {status}  Вопрос: {query}")
        print(f"      Ожидалось: {expected}")
        if use_transform:
            print(f"      Запрос после трансформации: {output['transformed_query']}")
        found = [r["article"] for r in results]
        print(f"      Найдено (топ-{k}): {found}")
        print()

    hit_rate = sum(hits) / len(hits)
    mrr = sum(rr_scores) / len(rr_scores)

    print(f"{'=' * 70}")
    print(f"Hit Rate @{k}: {hit_rate:.2%}  ({sum(hits)}/{len(hits)})")
    print(f"MRR @{k}: {mrr:.4f}")
    print(f"{'=' * 70}\n")

def main() -> None:
    print("Загрузка модели и индекса...")
    codex = CodexRag()

    evaluate(codex, k=5, use_transform=False)
    evaluate(codex, k=5, use_transform=True)

if __name__ == "__main__":
    main()