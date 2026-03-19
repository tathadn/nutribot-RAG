"""
Evaluation pipeline for NutriBot RAG system.

Measures:
  - Retrieval precision (are the right chunks coming back?)
  - Retrieval recall (are all relevant chunks found?)
  - Answer relevance (does Claude use the context correctly?)
  - Cost tracking (tokens used per query)

Usage:
    python -m app.evaluate                  # run full eval
    python -m app.evaluate --retrieval-only # skip LLM calls (free)
"""

import json
import time
import argparse
from pathlib import Path

from app.config import FAISS_INDEX_DIR, TOP_K
from app.hybrid_retriever import HybridRetriever
from app.llm import generate_answer

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
RESULTS_FILE = EVAL_DIR / "results.json"

# ---------------------------------------------------------------------------
# Test dataset: 30 question-answer pairs with expected source papers
# ---------------------------------------------------------------------------

EVAL_QUESTIONS = [
    # -- Gut Microbiome, Diet & Obesity (Patloka 2024) --
    {
        "id": 1,
        "question": "How does dietary fiber affect gut microbiota composition?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["fiber", "microbiota", "SCFA", "short-chain fatty acid"],
        "ground_truth": "Dietary fiber promotes the growth of beneficial SCFA-producing bacteria in the gut, which helps regulate metabolism and reduce inflammation."
    },
    {
        "id": 2,
        "question": "What is the relationship between gut dysbiosis and obesity?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["dysbiosis", "obesity", "microbiome", "metabolic"],
        "ground_truth": "Gut dysbiosis is associated with obesity through altered energy harvest from food, chronic low-grade inflammation, and disrupted metabolic signaling."
    },
    {
        "id": 3,
        "question": "What role does resistant starch play in gut health?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["resistant starch", "ferment", "colon", "butyrate"],
        "ground_truth": "Resistant starch reaches the colon undigested where it is fermented by gut bacteria, producing beneficial short-chain fatty acids like butyrate."
    },
    {
        "id": 4,
        "question": "How do high-fat diets impact the gut microbiome?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["high-fat", "diet", "microbiome", "diversity"],
        "ground_truth": "High-fat diets tend to reduce gut microbial diversity and promote pro-inflammatory bacterial populations."
    },
    {
        "id": 5,
        "question": "What is personalized nutrition in the context of gut microbiome?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["personalized", "nutrition", "microbiome", "individual"],
        "ground_truth": "Personalized nutrition uses individual gut microbiome composition to tailor dietary recommendations for optimal health outcomes."
    },
    {
        "id": 6,
        "question": "How do vegetable and animal proteins differently affect gut bacteria?",
        "expected_sources": ["gut_microbiome"],
        "expected_keywords": ["protein", "plant", "animal", "bacteria"],
        "ground_truth": "Plant proteins tend to promote beneficial bacterial growth and diversity, while high animal protein intake can increase potentially harmful bacterial metabolites."
    },

    # -- Omega-3 Fatty Acids (Patted 2024) --
    {
        "id": 7,
        "question": "What are the main dietary sources of omega-3 fatty acids?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["fish", "EPA", "DHA", "ALA", "flaxseed"],
        "ground_truth": "Main sources include fatty fish (salmon, mackerel) for EPA and DHA, and plant sources (flaxseed, chia, walnuts) for ALA."
    },
    {
        "id": 8,
        "question": "How do omega-3 fatty acids reduce inflammation?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["inflammation", "omega-3", "anti-inflammatory", "cytokine"],
        "ground_truth": "Omega-3s reduce inflammation by competing with omega-6 fatty acids in inflammatory pathways, decreasing pro-inflammatory cytokine production."
    },
    {
        "id": 9,
        "question": "What is the difference between EPA and DHA?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["EPA", "DHA", "eicosapentaenoic", "docosahexaenoic"],
        "ground_truth": "EPA is primarily anti-inflammatory and cardiovascular-protective, while DHA is crucial for brain and retinal structure and function."
    },
    {
        "id": 10,
        "question": "What are the cardiovascular benefits of omega-3 supplementation?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["cardiovascular", "heart", "triglyceride", "omega-3"],
        "ground_truth": "Omega-3s reduce triglyceride levels, lower blood pressure, decrease risk of arrhythmia, and may reduce overall cardiovascular event risk."
    },
    {
        "id": 11,
        "question": "How do omega-3s affect cognitive function and brain health?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["cognitive", "brain", "DHA", "neurological"],
        "ground_truth": "DHA is a major structural component of brain cell membranes and supports cognitive function, with potential protective effects against neurodegenerative diseases."
    },
    {
        "id": 12,
        "question": "What is the recommended daily intake of omega-3 fatty acids?",
        "expected_sources": ["omega3"],
        "expected_keywords": ["recommended", "intake", "dose", "daily"],
        "ground_truth": "General recommendations suggest 250-500mg of combined EPA and DHA per day for healthy adults, with higher doses for specific conditions."
    },

    # -- Mediterranean Diet (Abrignani 2024) --
    {
        "id": 13,
        "question": "What are the key components of the Mediterranean diet?",
        "expected_sources": ["mediterranean"],
        "expected_keywords": ["olive oil", "vegetables", "fish", "whole grains"],
        "ground_truth": "The Mediterranean diet emphasizes olive oil, fruits, vegetables, whole grains, legumes, nuts, fish, and moderate wine consumption while limiting red meat and processed foods."
    },
    {
        "id": 14,
        "question": "How does the Mediterranean diet affect cardiovascular health?",
        "expected_sources": ["mediterranean"],
        "expected_keywords": ["cardiovascular", "heart", "Mediterranean", "disease"],
        "ground_truth": "The Mediterranean diet reduces cardiovascular disease risk by lowering inflammation, improving lipid profiles, and reducing blood pressure."
    },
    {
        "id": 15,
        "question": "What is the connection between the Mediterranean diet and gut microbiome?",
        "expected_sources": ["mediterranean", "gut_microbiome"],
        "expected_keywords": ["Mediterranean", "microbiome", "gut", "diversity"],
        "ground_truth": "The Mediterranean diet promotes gut microbial diversity and increases beneficial bacterial populations, particularly those producing short-chain fatty acids."
    },
    {
        "id": 16,
        "question": "Does the Mediterranean diet help with weight management?",
        "expected_sources": ["mediterranean"],
        "expected_keywords": ["weight", "Mediterranean", "obesity", "body"],
        "ground_truth": "Evidence suggests the Mediterranean diet supports healthy weight management through satiety-promoting foods and balanced macronutrient composition."
    },
    {
        "id": 17,
        "question": "How does olive oil contribute to the health benefits of the Mediterranean diet?",
        "expected_sources": ["mediterranean"],
        "expected_keywords": ["olive oil", "polyphenol", "monounsaturated", "antioxidant"],
        "ground_truth": "Olive oil provides monounsaturated fatty acids and polyphenols with anti-inflammatory and antioxidant properties that protect against cardiovascular disease."
    },
    {
        "id": 18,
        "question": "What role does the Mediterranean diet play in diabetes prevention?",
        "expected_sources": ["mediterranean"],
        "expected_keywords": ["diabetes", "insulin", "glucose", "Mediterranean"],
        "ground_truth": "The Mediterranean diet improves insulin sensitivity and glucose metabolism, reducing the risk of developing type 2 diabetes."
    },

    # -- Nutrition, Gut Microbiota & Immunity (2025) --
    {
        "id": 19,
        "question": "How does nutrition influence the immune system through gut microbiota?",
        "expected_sources": ["immunity"],
        "expected_keywords": ["immune", "gut", "microbiota", "nutrition"],
        "ground_truth": "Dietary nutrients shape gut microbiota composition, which in turn modulates immune responses through metabolite production and immune cell signaling."
    },
    {
        "id": 20,
        "question": "What is the nutrition-gut microbiota-immunity axis?",
        "expected_sources": ["immunity"],
        "expected_keywords": ["axis", "nutrition", "microbiota", "immunity"],
        "ground_truth": "The nutrition-gut microbiota-immunity axis describes the three-way regulatory relationship where diet shapes the microbiome, which modulates immune function, affecting overall health."
    },
    {
        "id": 21,
        "question": "How do autoimmune disorders relate to gut microbiota disruption?",
        "expected_sources": ["immunity"],
        "expected_keywords": ["autoimmune", "gut", "dysbiosis", "immune"],
        "ground_truth": "Disruption of gut microbiota balance can trigger inappropriate immune responses, contributing to autoimmune disorders through impaired immune tolerance."
    },
    {
        "id": 22,
        "question": "Can dietary interventions modulate allergic responses through the gut?",
        "expected_sources": ["immunity"],
        "expected_keywords": ["allergy", "diet", "gut", "immune"],
        "ground_truth": "Dietary interventions that promote healthy gut microbiota can influence immune tolerance and potentially reduce allergic responses."
    },
    {
        "id": 23,
        "question": "What nutrients are most important for immune function?",
        "expected_sources": ["immunity", "omega3"],
        "expected_keywords": ["vitamin", "zinc", "immune", "nutrient"],
        "ground_truth": "Key nutrients for immune function include vitamins A, C, D, E, zinc, selenium, iron, and omega-3 fatty acids."
    },
    {
        "id": 24,
        "question": "How does gut microbiota affect mental health through the immune system?",
        "expected_sources": ["immunity"],
        "expected_keywords": ["mental", "gut-brain", "inflammation", "immune"],
        "ground_truth": "Gut microbiota influences mental health through the gut-brain axis, where immune-mediated inflammation and microbial metabolites affect neurological function."
    },

    # -- Probiotics, Prebiotics & Gut Health (2024) --
    {
        "id": 25,
        "question": "What is the difference between probiotics and prebiotics?",
        "expected_sources": ["probiotics"],
        "expected_keywords": ["probiotic", "prebiotic", "live", "fiber"],
        "ground_truth": "Probiotics are live beneficial microorganisms, while prebiotics are non-digestible food components that feed and support the growth of beneficial gut bacteria."
    },
    {
        "id": 26,
        "question": "What are synbiotics and how do they work?",
        "expected_sources": ["probiotics"],
        "expected_keywords": ["synbiotic", "probiotic", "prebiotic", "combination"],
        "ground_truth": "Synbiotics are combinations of probiotics and prebiotics designed to work together, where the prebiotic supports the survival and activity of the probiotic."
    },
    {
        "id": 27,
        "question": "What are postbiotics and their health benefits?",
        "expected_sources": ["probiotics"],
        "expected_keywords": ["postbiotic", "metabolite", "benefit", "health"],
        "ground_truth": "Postbiotics are bioactive compounds produced by probiotic bacteria during fermentation, including short-chain fatty acids, enzymes, and peptides that provide health benefits."
    },
    {
        "id": 28,
        "question": "How do probiotics help with digestive health?",
        "expected_sources": ["probiotics"],
        "expected_keywords": ["probiotic", "digestive", "gut", "barrier"],
        "ground_truth": "Probiotics improve digestive health by strengthening the gut barrier, competing with pathogens, modulating immune responses, and aiding nutrient absorption."
    },
    {
        "id": 29,
        "question": "What factors affect gut microbiome development from birth?",
        "expected_sources": ["probiotics"],
        "expected_keywords": ["birth", "development", "infant", "microbiome"],
        "ground_truth": "Gut microbiome development is influenced by delivery mode, breastfeeding, diet, antibiotic exposure, and environmental factors from birth onwards."
    },
    {
        "id": 30,
        "question": "Can probiotics help manage inflammatory bowel disease?",
        "expected_sources": ["probiotics", "immunity"],
        "expected_keywords": ["IBD", "probiotic", "inflammatory", "bowel"],
        "ground_truth": "Certain probiotic strains show promise in managing IBD symptoms by reducing inflammation and restoring microbial balance, though results vary by strain and condition."
    },
]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def keyword_hit_rate(retrieved_chunks: list[dict], expected_keywords: list[str]) -> float:
    """What fraction of expected keywords appear in retrieved chunk texts?"""
    if not expected_keywords:
        return 1.0
    combined_text = " ".join(c["text"].lower() for c in retrieved_chunks)
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    return hits / len(expected_keywords)


def mean_reciprocal_rank(retrieved_chunks: list[dict], expected_keywords: list[str]) -> float:
    """MRR: inverse rank of the first chunk containing any expected keyword."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        text_lower = chunk["text"].lower()
        if any(kw.lower() in text_lower for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


def answer_keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """What fraction of expected keywords appear in the generated answer?"""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(retrieval_only: bool = False):
    """Run the full evaluation suite."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading hybrid retriever...")
    retriever = HybridRetriever()
    if not retriever.is_ready():
        print("ERROR: No index found. Run `python -m app.ingest` first.")
        return

    results = []
    total_retrieval_precision = 0.0
    total_mrr = 0.0
    total_answer_coverage = 0.0
    api_calls = 0

    print(f"\nRunning evaluation on {len(EVAL_QUESTIONS)} questions...")
    print("=" * 70)

    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"\n[{i+1}/{len(EVAL_QUESTIONS)}] {q['question'][:60]}...")

        # -- Retrieval --
        chunks = retriever.search(q["question"], top_k=TOP_K)
        kw_hit = keyword_hit_rate(chunks, q["expected_keywords"])
        mrr = mean_reciprocal_rank(chunks, q["expected_keywords"])
        total_retrieval_precision += kw_hit
        total_mrr += mrr

        result = {
            "id": q["id"],
            "question": q["question"],
            "retrieval_keyword_hit_rate": round(kw_hit, 3),
            "mrr": round(mrr, 3),
            "num_chunks_retrieved": len(chunks),
            "top_chunk_score": round(chunks[0]["score"], 4) if chunks else 0,
            "chunk_sources": [c["source"] for c in chunks],
        }

        # -- Generation (skip if retrieval-only) --
        if not retrieval_only:
            try:
                gen_result = generate_answer(q["question"], chunks)
                answer = gen_result["answer"]
                ans_coverage = answer_keyword_coverage(answer, q["expected_keywords"])
                total_answer_coverage += ans_coverage
                api_calls += 1

                result["answer"] = answer
                result["answer_keyword_coverage"] = round(ans_coverage, 3)
                print(f"   Retrieval: {kw_hit:.0%} | MRR: {mrr:.2f} | Answer: {ans_coverage:.0%}")
            except Exception as e:
                result["answer"] = f"ERROR: {e}"
                result["answer_keyword_coverage"] = 0.0
                print(f"   Retrieval: {kw_hit:.0%} | MRR: {mrr:.2f} | Answer: ERROR")
        else:
            print(f"   Retrieval: {kw_hit:.0%} | MRR: {mrr:.2f}")

        results.append(result)

    # -- Summary --
    n = len(EVAL_QUESTIONS)
    summary = {
        "total_questions": n,
        "avg_retrieval_keyword_hit_rate": round(total_retrieval_precision / n, 3),
        "avg_mrr": round(total_mrr / n, 3),
        "api_calls_made": api_calls,
    }
    if not retrieval_only:
        summary["avg_answer_keyword_coverage"] = round(total_answer_coverage / n, 3)
        summary["estimated_cost_usd"] = round(api_calls * 0.008, 3)

    output = {"summary": summary, "results": results}

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Questions evaluated:           {n}")
    print(f"  Avg retrieval keyword hit rate: {summary['avg_retrieval_keyword_hit_rate']:.1%}")
    print(f"  Avg MRR:                        {summary['avg_mrr']:.3f}")
    if not retrieval_only:
        print(f"  Avg answer keyword coverage:   {summary['avg_answer_keyword_coverage']:.1%}")
        print(f"  API calls made:                {api_calls}")
        print(f"  Estimated cost:                ${summary['estimated_cost_usd']:.3f}")
    print(f"\n  Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NutriBot RAG system")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only evaluate retrieval (no LLM calls, zero cost)")
    args = parser.parse_args()
    run_evaluation(retrieval_only=args.retrieval_only)
