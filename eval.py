import os
import sys
import time
from pathlib import Path

# Ensure backend directory is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from rag_manager import rag_instance
from main import create_chatbot
from langchain_core.messages import HumanMessage


# ==================================================
# EVALUATION TEST BENCHMARK DATASET
# ==================================================

KNOWLEDGE_QUESTIONS = [
    {
        "query": "Where is the global headquarters of Anavi Chatbot Enterprises?",
        "expected_keyword": "Bangalore",
    },
    {
        "query": "What is the standard return window for physical products?",
        "expected_keyword": "30",
    },
    {
        "query": "What is the warranty period for Anavi AI Gateway Pro?",
        "expected_keyword": "3-Year",
    },
    {
        "query": "What are the domestic shipping rates in India?",
        "expected_keyword": "Free",
    },
    {
        "query": "What deployment options are supported for Anavi Enterprise RAG Suite?",
        "expected_keyword": "Docker",
    },
    {
        "query": "What payment methods are accepted?",
        "expected_keyword": "Credit",
    },
    {
        "query": "Are bulk or volume discounts available?",
        "expected_keyword": "discounts",
    },
    {
        "query": "What is the support coverage for enterprise plan subscribers?",
        "expected_keyword": "24/7",
    },
    {
        "query": "How does the chatbot protect customer data privacy?",
        "expected_keyword": "ChromaDB",
    },
    {
        "query": "Where is the regional office in Europe located?",
        "expected_keyword": "London",
    },
]

OUT_OF_DOMAIN_QUESTIONS = [
    {
        "query": "What is the authentic recipe for cooking Italian lasagna?",
    },
    {
        "query": "Who won the FIFA World Cup in 1970?",
    },
    {
        "query": "What are the official rules for quantum chess on Mars?",
    },
    {
        "query": "What is the secret formula for Coca-Cola?",
    },
    {
        "query": "How do I repair a 1995 Honda Civic transmission?",
    },
]

TOOL_PRESERVATION_QUESTIONS = [
    {
        "query": "What is 15% of 850?",
        "expected_type": "math",
        "expected_keyword": "127.5",
    },
    {
        "query": "Which city has the highest sales?",
        "expected_type": "sales",
        "expected_keyword": "Bangalore",
    },
    {
        "query": "What is the total sales for Electronics?",
        "expected_type": "sales",
        "expected_keyword": "153",
    },
    {
        "query": "Say hello to Anavi",
        "expected_type": "greeting",
        "expected_keyword": "Hello Anavi",
    },
]


# ==================================================
# EVALUATION SUITE RUNNER
# ==================================================

def run_evaluation():
    print("=" * 70)
    print("      ANAVI CHATBOT RAG & GUARDRAILS EVALUATION BENCHMARK")
    print("=" * 70)

    # 1. Report Knowledge Base Stats
    print(f"\n[1/4] KNOWLEDGE BASE INDEX STATS:")
    print(f"  * Total Knowledge Base Documents Loaded: {rag_instance.total_docs}")
    print(f"  * Total Chunks Persisted in ChromaDB:    {rag_instance.total_chunks}")
    print(f"  * Vector Database Location:              {rag_instance.persist_dir}")
    print(f"  * Relevance Threshold Guardrail:         0.40 Similarity Score")

    # 2. Evaluate Knowledge Base Retrieval & Grounded Precision
    print(f"\n[2/4] EVALUATING GROUNDED KNOWLEDGE QUERIES (10 Questions)...")
    knowledge_hits = 0
    retrieval_scores = []

    for i, item in enumerate(KNOWLEDGE_QUESTIONS, 1):
        q = item["query"]
        expected_kw = item["expected_keyword"]
        result = rag_instance.query_with_guardrail(q)

        success = result.get("success", False)
        best_score = result.get("best_score", 0.0)
        context = result.get("context", "")

        hit = success and (expected_kw.lower() in context.lower())
        if hit:
            knowledge_hits += 1
        retrieval_scores.append(best_score)

        status_icon = "[PASSED]" if hit else "[FAILED]"
        print(f"  [{i:02d}] '{q}'")
        print(f"       Status: {status_icon} | Score: {best_score:.4f} | Source Chunks: {result.get('chunks_retrieved', 0)}")

    knowledge_retrieval_rate = (knowledge_hits / len(KNOWLEDGE_QUESTIONS)) * 100
    avg_retrieval_score = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0

    # 3. Evaluate Hallucination Guardrail (Out-of-Domain Detection)
    print(f"\n[3/4] EVALUATING HALLUCINATION GUARDRAILS (5 Irrelevant Questions)...")
    guardrail_triggers = 0

    for i, item in enumerate(OUT_OF_DOMAIN_QUESTIONS, 1):
        q = item["query"]
        result = rag_instance.query_with_guardrail(q)

        success = result.get("success", False)
        best_score = result.get("best_score", 0.0)
        message = result.get("message", "")

        triggered = (not success) and ("don't have enough information" in message.lower() or result.get("reason") == "insufficient_relevance")
        if triggered:
            guardrail_triggers += 1

        status_icon = "[BLOCKED] (Guardrail Triggered)" if triggered else "[FAILED] (Hallucinated)"
        print(f"  [{i:02d}] '{q}'")
        print(f"       Status: {status_icon} | Score: {best_score:.4f}")

    guardrail_accuracy = (guardrail_triggers / len(OUT_OF_DOMAIN_QUESTIONS)) * 100

    # 4. Evaluate Existing Multi-Tool Integration
    print(f"\n[4/4] EVALUATING MULTI-TOOL INTEGRATION & PRESERVATION (4 Queries)...")
    agent = create_chatbot("default")
    tool_successes = 0

    for i, item in enumerate(TOOL_PRESERVATION_QUESTIONS, 1):
        q = item["query"]
        expected_kw = item["expected_keyword"]

        # Sleep briefly to avoid Gemini free tier RPM rate limits
        time.sleep(3.0)

        try:
            response = agent.invoke({"messages": [HumanMessage(content=q)]})
            messages = response["messages"]
            final_content = ""
            for msg in reversed(messages):
                if msg.type == "ai" and msg.content:
                    final_content = str(msg.content)
                    break

            success = expected_kw.lower() in final_content.lower()
            if success:
                tool_successes += 1

            status_icon = "[PASSED]" if success else "[FAILED]"
            print(f"  [{i:02d}] '{q}' ({item['expected_type']})")
            print(f"       Status: {status_icon} | Response Preview: {final_content[:70]}...")
        except Exception as e:
            print(f"  [{i:02d}] '{q}' -> Error: {e}")

    tool_preservation_rate = (tool_successes / len(TOOL_PRESERVATION_QUESTIONS)) * 100

    # 5. Final Evaluation Summary Report
    print("\n" + "=" * 70)
    print("                  FINAL RAG BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  1. Knowledge Retrieval Precision:      {knowledge_retrieval_rate:.1f}% ({knowledge_hits}/{len(KNOWLEDGE_QUESTIONS)} queries hit)")
    print(f"  2. Average Context Relevance Score:   {avg_retrieval_score:.4f}")
    print(f"  3. Hallucination Guardrail Accuracy:   {guardrail_accuracy:.1f}% ({guardrail_triggers}/{len(OUT_OF_DOMAIN_QUESTIONS)} blocked)")
    print(f"  4. Multi-Tool Preservation Rate:       {tool_preservation_rate:.1f}% ({tool_successes}/{len(TOOL_PRESERVATION_QUESTIONS)} tools functional)")
    print("=" * 70 + "\n")

    return {
        "knowledge_docs": rag_instance.total_docs,
        "knowledge_chunks": rag_instance.total_chunks,
        "retrieval_precision": round(knowledge_retrieval_rate, 1),
        "avg_relevance_score": round(avg_retrieval_score, 4),
        "guardrail_accuracy": round(guardrail_accuracy, 1),
        "tool_preservation_rate": round(tool_preservation_rate, 1),
    }


if __name__ == "__main__":
    run_evaluation()
