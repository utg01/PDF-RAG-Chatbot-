import json
import os
import time
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from simple_rag_agent import agent as baseline_agent
from agent import agent as advanced_agent

load_dotenv()

DATASET_PATH = "evaluation/ragas_golden_dataset.json"
RESULTS_DIR = "evaluation/results"

BASELINE_PATH = f"{RESULTS_DIR}/baseline_results.json"
ADVANCED_PATH = f"{RESULTS_DIR}/advanced_results.json"
ERRORS_PATH = f"{RESULTS_DIR}/errors.json"

INDEX_NAME = "pdf-cloud-sre-handbook-generation-plan-pdf"

MAX_RETRIES = 3
RETRY_DELAY = 10
RECursion_LIMIT = 15


def load_json(path, default):
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    os.replace(temp_path, path)


def run_agent(agent, question, index_name):

    thread_id = f"eval-{uuid.uuid4()}"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "index_name": index_name
        },
        "recursion_limit": RECursion_LIMIT
    }

    retrieved_contexts = []
    final_response = ""

    for message, metadata in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages"
    ):

        if isinstance(message, ToolMessage):
            if message.content:
                retrieved_contexts.append(message.content)

        elif isinstance(message, AIMessage):
            if message.content:
                final_response += message.content

    return {
        "response": final_response,
        "retrieved_contexts": retrieved_contexts
    }


def run_with_retry(agent, question, index_name):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            return run_agent(
                agent,
                question,
                index_name
            )

        except Exception as e:

            last_error = e
            error_text = str(e).lower()

            print(f"    Attempt {attempt}/{MAX_RETRIES} failed:")
            print(f"    {e}")

            # Permanent/authentication type errors
            if "403" in error_text and (
                "invalid" in error_text
                or "authentication" in error_text
                or "unauthorized" in error_text
                or "api key" in error_text
            ):
                print("    Authentication/permission error. Skipping question.")
                break

            # Retry temporary/rate-limit errors
            retryable = (
                "429" in error_text
                or "rate limit" in error_text
                or "too many requests" in error_text
                or "timeout" in error_text
                or "timed out" in error_text
                or "500" in error_text
                or "502" in error_text
                or "503" in error_text
                or "504" in error_text
                or "temporarily unavailable" in error_text
            )

            if not retryable:
                print("    Non-retryable error. Skipping question.")
                break

            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (2 ** (attempt - 1))

                print(f"    Retrying after {delay} seconds...")
                time.sleep(delay)

    raise last_error


def evaluate_agent(
    agent,
    dataset,
    results,
    errors,
    agent_name,
    output_path
):

    completed_ids = {
        item["id"]
        for item in results
        if item.get("status") == "success"
    }

    print(f"\n========== {agent_name.upper()} ==========\n")

    for i, item in enumerate(dataset, start=1):

        question_id = item["id"]

        # Already successfully evaluated
        if question_id in completed_ids:
            print(
                f"[{i}/{len(dataset)}] "
                f"{question_id} already completed. Skipping."
            )
            continue

        question = item["question"]
        reference = item["reference_answer"]

        print(f"[{i}/{len(dataset)}] {question_id}")
        print(f"  {question}")

        try:

            result = run_with_retry(
                agent,
                question,
                INDEX_NAME
            )

            record = {
                "id": question_id,
                "user_input": question,
                "reference": reference,
                "response": result["response"],
                "retrieved_contexts": result["retrieved_contexts"],
                "question_type": item["question_type"],
                "difficulty": item["difficulty"],
                "source_pages": item["source_pages"],
                "status": "success"
            }

            results.append(record)

            # SAVE IMMEDIATELY
            save_json(output_path, results)

            print("  ✓ Saved")

        except Exception as e:

            error_record = {
                "id": question_id,
                "agent": agent_name,
                "question": question,
                "error": str(e),
                "timestamp": time.time()
            }

            errors.append(error_record)

            # SAVE ERROR IMMEDIATELY
            save_json(ERRORS_PATH, errors)

            print("  ✗ Failed")
            print(f"  Error: {e}")

            # Continue with next question
            continue


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = load_json(DATASET_PATH, [])

    baseline_results = load_json(BASELINE_PATH, [])
    advanced_results = load_json(ADVANCED_PATH, [])
    errors = load_json(ERRORS_PATH, [])

    print(f"Loaded {len(dataset)} golden questions.")

    print(
        f"Existing baseline results: "
        f"{len(baseline_results)}"
    )

    print(
        f"Existing advanced results: "
        f"{len(advanced_results)}"
    )

    # -------------------------
    # BASELINE
    # -------------------------

    evaluate_agent(
        baseline_agent,
        dataset,
        baseline_results,
        errors,
        "baseline",
        BASELINE_PATH
    )

    # -------------------------
    # ADVANCED
    # -------------------------

    evaluate_agent(
        advanced_agent,
        dataset,
        advanced_results,
        errors,
        "advanced",
        ADVANCED_PATH
    )

    print("\n================================")
    print("Evaluation run finished.")
    print("================================")

    print(
        f"Baseline completed: "
        f"{len(baseline_results)}/{len(dataset)}"
    )

    print(
        f"Advanced completed: "
        f"{len(advanced_results)}/{len(dataset)}"
    )

    print(
        f"Errors recorded: "
        f"{len(errors)}"
    )


if __name__ == "__main__":
    main()