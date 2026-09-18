import os
import json
import time
import math
from pathlib import Path

from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = Path("evaluation/results")

BASELINE_PATH = RESULTS_DIR / "baseline_results.json"
ADVANCED_PATH = RESULTS_DIR / "advanced_results.json"

BASELINE_RAGAS_PATH = RESULTS_DIR / "baseline_ragas_results.json"
ADVANCED_RAGAS_PATH = RESULTS_DIR / "advanced_ragas_results.json"

START_FROM_ID = "Q009"

# Root-cause fix #1: gpt-oss-20b burns tokens on hidden reasoning before
# emitting the verdict JSON that RAGAS needs. "low" is enough for
# judging tasks like this and drastically cuts token usage.
REASONING_EFFORT = "low"

# Safety net on top of the reasoning_effort fix -- your contexts run
# 2000-3000+ words for some questions (adversarial/multi-hop ones
# especially), so give the judge headroom even at low reasoning.
MAX_TOKENS = 4096

# ============================================================
# 4 GROQ KEYS
# ============================================================

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
]

GROQ_KEYS = [key for key in GROQ_KEYS if key]

if not GROQ_KEYS:
    raise ValueError("No GROQ API keys found.")

print(f"Loaded {len(GROQ_KEYS)} Groq API keys.")


# ============================================================
# RAGAS METRICS
# ============================================================

METRICS = [
    Faithfulness(),
    AnswerRelevancy(strictness=1),  # GPT-OSS Groq setup does not support n > 1
    ContextPrecision(),
    ContextRecall(),
]


# ============================================================
# BUILD CLIENTS ONCE (root-cause fix #2)
# ============================================================
# Previously a fresh ChatGroq + MistralAIEmbeddings + implicit event
# loop was created per question. That's what produced the
# "Event loop is closed" / "coroutine was never awaited" noise --
# async clients were being torn down before their connections
# finished closing. Build everything once, up front, and reuse.

def build_llm(key):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=key,
        temperature=0,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
    )
    return LangchainLLMWrapper(llm)


def build_embeddings():
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY"),
    )
    return LangchainEmbeddingsWrapper(embeddings)


LLM_WRAPPERS = [build_llm(key) for key in GROQ_KEYS]
EMBEDDINGS = build_embeddings()  # embeddings don't need key rotation


# ============================================================
# HELPERS
# ============================================================

def is_valid(value):
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_dataset(row):
    return Dataset.from_dict({
        "user_input": [row["user_input"]],
        "reference": [row["reference"]],
        "response": [row["response"]],
        "retrieved_contexts": [row["retrieved_contexts"]],
    })


def has_all_scores(result):
    required = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    return all(key in result and is_valid(result[key]) for key in required)


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(row, key_index):

    dataset = make_dataset(row)

    llm = LLM_WRAPPERS[key_index]  # reused, not rebuilt

    run_config = RunConfig(
        max_workers=1,
        timeout=180,       # a bit more headroom alongside the bigger max_tokens
        max_retries=1,
        max_wait=30,
    )

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=llm,
        embeddings=EMBEDDINGS,
        run_config=run_config,
        raise_exceptions=True,
        show_progress=False,
    )

    scores = result.to_pandas().iloc[0].to_dict()

    return {
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
        "context_precision": scores.get("context_precision"),
        "context_recall": scores.get("context_recall"),
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_file(input_path, output_path, start_from_id=None):

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {input_path.name}")
    print(f"{'=' * 60}")

    source_data = load_json(input_path)

    if output_path.exists():
        existing_list = load_json(output_path)
    else:
        existing_list = []

    existing = {str(item["id"]): item for item in existing_list}

    key_index = 0
    started = start_from_id is None

    for i, row in enumerate(source_data):

        question_id = str(row["id"])

        if not started:
            if question_id != start_from_id:
                continue
            started = True

        if question_id in existing:
            old_scores = existing[question_id].get("scores", {})
            if has_all_scores(old_scores):
                print(f"[{i + 1}/{len(source_data)}] {question_id} -> already complete, skipping")
                continue

        success = False

        for attempt in range(len(GROQ_KEYS)):

            current_key = key_index % len(GROQ_KEYS)

            print(f"\n[{i + 1}/{len(source_data)}] {question_id} | Key {current_key + 1}/4")

            try:
                scores = evaluate_question(row, current_key)

                if not has_all_scores(scores):
                    raise RuntimeError(f"Invalid scores returned: {scores}")

                existing[question_id] = {
                    "id": question_id,
                    "question": row["user_input"],
                    "scores": scores,
                }

                save_json(output_path, list(existing.values()))

                print("SUCCESS:")
                print(json.dumps(scores, indent=2))

                key_index = (current_key + 1) % len(GROQ_KEYS)
                success = True
                break

            except Exception as e:
                print(f"FAILED with key {current_key + 1}: {type(e).__name__}: {e}")
                key_index = (current_key + 1) % len(GROQ_KEYS)
                time.sleep(3)

        if not success:
            print(f"\n!!! ALL 4 KEYS FAILED FOR {question_id} !!!")
            save_json(output_path, list(existing.values()))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\nStarting RAGAS evaluation...")
    print(f"Using Groq keys: 4, 5, 6, 7 | reasoning_effort={REASONING_EFFORT} | max_tokens={MAX_TOKENS}")
    print("Starting baseline from: Q009")

    evaluate_file(BASELINE_PATH, BASELINE_RAGAS_PATH, start_from_id="Q009")
    evaluate_file(ADVANCED_PATH, ADVANCED_RAGAS_PATH)

    print("\nDONE.")