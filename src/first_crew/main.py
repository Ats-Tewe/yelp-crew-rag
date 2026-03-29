import json
from first_crew.crew import FirstCrew


def run():
    # Read the first test case from the exam paper
    with open("data/test_review_subset.json", "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        test_entry = json.loads(first_line)

    user_id = test_entry["user_id"]
    item_id = test_entry["item_id"]

    print(f"\n>>> Running prediction for:")
    print(f"    user_id: {user_id}")
    print(f"    item_id: {item_id}\n")

    inputs = {
        "user_id": user_id,
        "item_id": item_id
    }

    result = FirstCrew().crew().kickoff(inputs=inputs)

    print("\n=== FINAL RESULT ===")
    print(result)


if __name__ == "__main__":
    run()