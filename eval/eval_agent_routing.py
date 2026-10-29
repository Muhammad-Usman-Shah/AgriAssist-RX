
"""
AgriAssist-RX
Phase 4, Step 7: Tool Routing Evaluation

Compares:
1. Keyword-based baseline router
2. Gemini-based intelligent router

Measures:
- Accuracy
- Per-class recall
- Confusion matrices
- Routing errors

Important:
This evaluates routing only.
It does NOT run RAG or weather API calls.
"""

import json
import sys
import time

from collections import Counter
from pathlib import Path


# ==========================================
# STEP 1: Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EVAL_DIR = PROJECT_ROOT / "eval"

sys.path.insert(0, str(SRC_DIR))

from agent_rag import choose_tool as keyword_router
from agent_intelligent import choose_tool as gemini_router


DATASET_PATH = EVAL_DIR / "routing_test_questions.json"
RESULTS_PATH = EVAL_DIR / "routing_evaluation_results.json"

LABELS = ["rag", "weather", "unsupported"]

MAX_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 5


# ==========================================
# STEP 2: Save results safely
# ==========================================

def save_results(results):

    temporary = RESULTS_PATH.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    temporary.replace(RESULTS_PATH)


# ==========================================
# STEP 3: Load benchmark
# ==========================================

def load_questions():

    questions = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    ids = [item["id"] for item in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found.")

    for item in questions:

        if item["expected_tool"] not in LABELS:
            raise ValueError(
                f"Invalid expected tool: {item['id']}"
            )

    return questions


# ==========================================
# STEP 4: Read existing progress
# ==========================================

def load_previous_results():

    if not RESULTS_PATH.exists():
        return {}

    stored = json.loads(
        RESULTS_PATH.read_text(encoding="utf-8")
    )

    return {
        item["id"]: item
        for item in stored
    }


# ==========================================
# STEP 5: Run keyword baseline
# ==========================================

def run_keyword_router(question):

    state = {
        "question": question
    }

    result = keyword_router(state)

    return result["selected_tool"]


# ==========================================
# STEP 6: Run Gemini router with retries
# ==========================================

def run_gemini_router(question):

    for attempt in range(1, MAX_ATTEMPTS + 1):

        state = {
            "question": question
        }

        result = gemini_router(state)

        selected = result.get(
            "selected_tool", "unsupported"
        )

        # The existing Gemini router uses the
        # unsupported route when its API fails.
        # Detect that failure from its error message.
        if result.get("answer", "").startswith(
            "The intelligent router is temporarily"
        ) or result.get("answer", "").startswith(
            "Gemini API key is missing"
        ):

            print(
                f"  Gemini error, attempt "
                f"{attempt}/{MAX_ATTEMPTS}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
                continue

            return None, result.get("answer", "")

        if selected not in LABELS:

            return None, (
                f"Unexpected Gemini tool: {selected}"
            )

        return selected, None

    return None, "Maximum attempts reached."


# ==========================================
# STEP 7: Score accuracy
# ==========================================

def accuracy(rows, field):

    valid = [
        row for row in rows
        if row.get(field) in LABELS
    ]

    if not valid:
        return None

    correct = sum(
        row[field] == row["expected_tool"]
        for row in valid
    )

    return correct / len(valid)


# ==========================================
# STEP 8: Calculate confusion matrix
# ==========================================

def confusion_matrix(rows, field):

    matrix = {
        expected: {
            predicted: 0
            for predicted in LABELS
        }
        for expected in LABELS
    }

    for row in rows:

        expected = row["expected_tool"]
        predicted = row.get(field)

        if predicted in LABELS:
            matrix[expected][predicted] += 1

    return matrix


# ==========================================
# STEP 9: Calculate per-class recall
# ==========================================

def per_class_recall(matrix):

    recalls = {}

    for label in LABELS:

        total = sum(matrix[label].values())

        recalls[label] = (
            matrix[label][label] / total
            if total else None
        )

    return recalls


# ==========================================
# STEP 10: Print evaluation report
# ==========================================

def print_report(rows):

    print("\n" + "=" * 60)
    print("AGRIASSIST-RX: ROUTING EVALUATION")
    print("=" * 60)

    total = len(rows)

    print(f"\nTotal benchmark questions: {total}")

    for name, field in [
        ("Keyword baseline", "keyword_prediction"),
        ("Gemini router", "gemini_prediction")
    ]:

        valid = [
            row for row in rows
            if row.get(field) in LABELS
        ]

        correct = sum(
            row[field] == row["expected_tool"]
            for row in valid
        )

        failed = total - len(valid)

        print(f"\n{name}")
        print("-" * 40)

        print(f"Evaluated: {len(valid)}/{total}")
        print(f"Request failures: {failed}")

        if valid:

            print(
                f"Accuracy: "
                f"{correct / len(valid):.2%}"
            )

            matrix = confusion_matrix(rows, field)

            print("\nConfusion matrix")
            print(
                "Expected \\ Predicted | "
                "RAG | Weather | Unsupported"
            )

            for label in LABELS:

                values = [
                    matrix[label][predicted]
                    for predicted in LABELS
                ]

                print(
                    f"{label:20} | "
                    + " | ".join(
                        str(value) for value in values
                    )
                )

            print("\nPer-class recall")

            recalls = per_class_recall(matrix)

            for label, value in recalls.items():

                if value is not None:
                    print(
                        f"  {label}: {value:.2%}"
                    )

            errors = [
                row for row in valid
                if row[field] != row["expected_tool"]
            ]

            print(
                f"\nRouting errors: {len(errors)}"
            )

            for row in errors:

                print(
                    f"  {row['id']}: "
                    f"expected={row['expected_tool']}, "
                    f"predicted={row[field]}"
                )

                print(
                    f"    {row['question']}"
                )

    print("\n" + "=" * 60)
    print(f"Results saved to:\n{RESULTS_PATH}")
    print("=" * 60)


# ==========================================
# STEP 11: Main evaluation
# ==========================================

def main():

    questions = load_questions()

    previous = load_previous_results()

    print("=" * 60)
    print("AGRIASSIST-RX: 30-QUESTION ROUTING BENCHMARK")
    print("=" * 60)

    print(f"\nQuestions loaded: {len(questions)}")
    print(
        "Expected classes:",
        dict(
            Counter(
                item["expected_tool"]
                for item in questions
            )
        )
    )

    rows = []

    for index, item in enumerate(questions, start=1):

        question_id = item["id"]
        question = item["question"]
        expected = item["expected_tool"]

        print(
            f"\n[{index}/{len(questions)}] "
            f"{question_id}: {question}",
            flush=True
        )

        old = previous.get(question_id, {})

        # Reuse only results with an identical
        # question and expected class.
        same_question = (
            old.get("question") == question
            and old.get("expected_tool") == expected
        )

        baseline = run_keyword_router(question)

        # Do not repeat a successful Gemini call.
        if (
            same_question
            and old.get("gemini_prediction") in LABELS
            and not old.get("gemini_error")
        ):

            gemini = old["gemini_prediction"]
            error = None

            print("  Using saved Gemini result.")

        else:

            gemini, error = run_gemini_router(
                question
            )

        row = {
            "id": question_id,
            "question": question,
            "expected_tool": expected,
            "keyword_prediction": baseline,
            "gemini_prediction": gemini,
            "gemini_error": error,
            "keyword_correct": (
                baseline == expected
            ),
            "gemini_correct": (
                gemini == expected
                if gemini is not None
                else None
            ),
        }

        rows.append(row)

        save_results(rows)

        print(f"  Expected: {expected}")
        print(f"  Keyword:  {baseline}")
        print(f"  Gemini:   {gemini or 'FAILED'}")

    print_report(rows)


if __name__ == "__main__":
    main()
