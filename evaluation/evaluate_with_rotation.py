import os
import importlib

from dotenv import load_dotenv

load_dotenv()

# Reuse everything from existing evaluation script
from evaluation.evaluate import (
    load_json,
    evaluate_agent,
    DATASET_PATH,
    RESULTS_DIR,
    BASELINE_PATH,
    ADVANCED_PATH,
    ERRORS_PATH,
    INDEX_NAME,
)


# ============================================================
# GROQ API KEYS
# ============================================================

GROQ_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
]

GROQ_KEYS = [key for key in GROQ_KEYS if key]

if not GROQ_KEYS:
    raise ValueError(
        "No Groq API keys found. "
        "Set GROQ_API_KEY_1 and GROQ_API_KEY_2 in .env"
    )


# ============================================================
# 429 DETECTION
# ============================================================

def is_rate_limit_error(error):
    error_text = str(error).lower()

    return (
        "429" in error_text
        or "rate limit" in error_text
        or "too many requests" in error_text
        or "rate_limit_exceeded" in error_text
    )


# ============================================================
# ROTATING AGENT
# ============================================================

class RotatingAgent:

    def __init__(self, module_name, agent_name):
        self.module_name = module_name
        self.agent_name = agent_name

        self.key_index = 0
        self.module = None
        self.agent = None

        self.load_agent()

    def load_agent(self):

        os.environ["GROQ_API_KEY"] = GROQ_KEYS[self.key_index]

        print(
            f"\n🔑 {self.agent_name}: "
            f"Using GROQ API {self.key_index + 1}"
        )

        self.module = importlib.import_module(
            self.module_name
        )

        self.module = importlib.reload(
            self.module
        )

        self.agent = self.module.agent

    def switch_key(self):

        if len(GROQ_KEYS) <= 1:
            raise RuntimeError(
                "Only one Groq API key is configured."
            )

        old_key = self.key_index

        self.key_index = (
            self.key_index + 1
        ) % len(GROQ_KEYS)

        print(
            f"\n⚠️ GROQ API {old_key + 1} hit rate limit."
        )

        print(
            f"🔄 Switching to GROQ API "
            f"{self.key_index + 1}"
        )

        self.load_agent()

    def stream(self, input_data, config, stream_mode):

        while True:

            try:

                # Buffer the complete response first.
                # This prevents partial output from being
                # duplicated if a 429 happens mid-stream.
                chunks = []

                for chunk in self.agent.stream(
                    input_data,
                    config=config,
                    stream_mode=stream_mode
                ):
                    chunks.append(chunk)

                # Only expose chunks after the entire
                # agent execution succeeds.
                for chunk in chunks:
                    yield chunk

                return

            except Exception as e:

                if is_rate_limit_error(e):

                    self.switch_key()

                    print(
                        "🔁 Retrying the same question "
                        "with the new API key..."
                    )

                    continue

                raise


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    dataset = load_json(
        DATASET_PATH,
        []
    )

    baseline_results = load_json(
        BASELINE_PATH,
        []
    )

    advanced_results = load_json(
        ADVANCED_PATH,
        []
    )

    errors = load_json(
        ERRORS_PATH,
        []
    )

    print(
        f"Loaded {len(dataset)} golden questions."
    )

    print(
        f"Existing baseline results: "
        f"{len(baseline_results)}"
    )

    print(
        f"Existing advanced results: "
        f"{len(advanced_results)}"
    )

    # ========================================================
    # BASELINE
    # ========================================================

    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    baseline_agent = RotatingAgent(
        module_name="simple_rag_agent",
        agent_name="BASELINE"
    )

    evaluate_agent(
        baseline_agent,
        dataset,
        baseline_results,
        errors,
        "baseline",
        BASELINE_PATH
    )

    # ========================================================
    # ADVANCED
    # ========================================================

    print("\n" + "=" * 60)
    print("ADVANCED EVALUATION")
    print("=" * 60)

    advanced_agent = RotatingAgent(
        module_name="agent",
        agent_name="ADVANCED"
    )

    evaluate_agent(
        advanced_agent,
        dataset,
        advanced_results,
        errors,
        "advanced",
        ADVANCED_PATH
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("EVALUATION FINISHED")
    print("=" * 60)

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