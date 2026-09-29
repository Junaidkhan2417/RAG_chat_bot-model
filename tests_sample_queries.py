import json
import time
import sys
from typing import List, Dict, Any

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME, has_live_credentials

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "What is Agentic AI according to the eBook?",
        "expected_type": "grounded_answer",
        "notes": "Validates core definition and conceptual understanding of Agentic AI.",
    },
    {
        "id": 2,
        "query": "How do AI agents differ from traditional automation systems?",
        "expected_type": "grounded_answer",
        "notes": "Validates contrast with static robotic process automation / rules.",
    },
    {
        "id": 3,
        "query": "What are the core components of an Agentic Architecture?",
        "expected_type": "grounded_answer",
        "notes": "Validates extraction of anatomy (perception, reasoning, memory, tools).",
    },
    {
        "id": 4,
        "query": "What role does memory play in Agentic AI workflows?",
        "expected_type": "grounded_answer",
        "notes": "Validates episodic vs semantic memory roles in stateful execution.",
    },
    {
        "id": 5,
        "query": "Who won the 2022 FIFA World Cup?",
        "expected_type": "refusal",
        "notes": "Out-of-domain query: verifies strict hallucination prevention and refusal.",
    },
]


def run_benchmark_tests() -> List[Dict[str, Any]]:
    print("=" * 80)
    print("RUNNING RAG AGENTIC AI BENCHMARK TEST SUITE")
    print(f"Mode: {'[LIVE] Pinecone + OpenAI' if has_live_credentials() else '[LOCAL FALLBACK] Grounded Chunks'}")
    print(f"Target Index: {PINECONE_INDEX_NAME}")
    print("=" * 80)

    graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
    results = []

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        q = item["query"]
        qtype = item["expected_type"]

        print(f"\n[Test Query #{qid}] {q}")
        print(f"Expected Behavior: {qtype.upper()} ({item['notes']})")
        print("-" * 80)

        start_time = time.time()
        initial_state = {
            "question": q,
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        res = graph.invoke(initial_state)
        elapsed = time.time() - start_time

        answer = res.get("answer", "")
        context = res.get("context", [])
        score = float(res.get("score", 0.0))

        # Check validation criteria
        refusal_keywords = ["cannot answer", "not contain", "not mentioned", "provided document"]
        is_refusal = any(k in answer.lower() for k in refusal_keywords) or score == 0.0

        if qtype == "refusal":
            passed = is_refusal
            status_str = "PASSED (Strict Refusal Triggered)" if passed else "FAILED (Expected refusal)"
        else:
            passed = (not is_refusal and len(context) > 0)
            status_str = "PASSED (Grounded Answer Generated)" if passed else "FAILED"

        print(f"Answer:\n   {answer}")
        print(f"Confidence Score: {score:.2f}")
        print(f"Latency: {elapsed:.2f}s")
        print(f"Retrieved Chunks Count: {len(context)}")
        if context:
            snippet = context[0].replace("\n", " ").strip()[:180]
            print(f"Top Chunk Snippet: \"{snippet}...\"")
        print(f"Test Status: {status_str}")

        results.append({
            "id": qid,
            "query": q,
            "answer": answer,
            "score": score,
            "context_count": len(context),
            "passed": passed,
            "latency": elapsed,
        })

    print("\n" + "=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    all_passed = all(r["passed"] for r in results)
    for r in results:
        print(f"Query {r['id']}: {'[PASS]' if r['passed'] else '[FAIL]'} | Score: {r['score']:.2f} | Chunks: {r['context_count']}")

    print(f"\nFinal Verdict: {'ALL BENCHMARK TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return results


if __name__ == "__main__":
    results = run_benchmark_tests()
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Saved benchmark results to benchmark_results.json")
