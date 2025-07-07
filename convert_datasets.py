import argparse
import os
import random
import json

import google.generativeai as genai


def generate_qa(model, label):
    prompt = (
        "You are an assistant that writes synthetic veterinary consultation data.\n"
        f"Condition label: {label}.\n"
        "1. Create a short question from a pet owner describing typical symptoms for this condition.\n"
        "2. Provide a concise helpful answer that mentions the condition name naturally.\n"
        "Respond with JSON containing 'question' and 'answer'."
    )
    response = model.generate_content(prompt)
    text = getattr(response, 'text', None)
    if text is None and hasattr(response, 'candidates'):
        text = response.candidates[0].text
    try:
        data = json.loads(text)
        return data.get('question', '').strip(), data.get('answer', '').strip()
    except Exception:
        lines = text.strip().splitlines()
        question = lines[0] if lines else ''
        answer = " ".join(lines[1:]) if len(lines) > 1 else ''
        return question.strip(), answer.strip()


def collect_records(model, dataset_path):
    records = []
    parent = os.path.dirname(dataset_path)
    for root, _, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = os.path.join(root, name)
                rel = os.path.relpath(img_path, parent).replace(os.sep, "/")
                question, answer = generate_qa(model, label)
                records.append({
                    "image": rel,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ],
                })
    return records


def main():
    parser = argparse.ArgumentParser(description="Convert image datasets for fine-tuning")
    parser.add_argument("--dogs1", required=True, help="Path to first dog dataset")
    parser.add_argument("--dogs2", required=True, help="Path to second dog dataset")
    parser.add_argument("--cats", required=True, help="Path to cat dataset")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-latest")

    random.seed(args.seed)

    datasets = {
        "dogs1": collect_records(model, args.dogs1),
        "dogs2": collect_records(model, args.dogs2),
        "cats": collect_records(model, args.cats),
    }

    limit = min(len(v) for v in datasets.values())
    train, val = [], []
    for recs in datasets.values():
        random.shuffle(recs)
        recs = recs[:limit]
        n_val = int(limit * args.val_ratio)
        val.extend(recs[:n_val])
        train.extend(recs[n_val:])

    random.shuffle(train)
    random.shuffle(val)

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "train.jsonl"), "w") as f:
        for r in train:
            json.dump(r, f)
            f.write("\n")
    with open(os.path.join(args.output, "val.jsonl"), "w") as f:
        for r in val:
            json.dump(r, f)
            f.write("\n")


if __name__ == "__main__":
    main()
