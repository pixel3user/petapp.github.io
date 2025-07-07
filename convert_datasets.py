import argparse
import os
import random
import json

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.0-flash"


def generate_qa(client, image_path, label):
    with open(image_path, "rb") as f:
        data = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_data(mime_type="image/jpeg", data=data)],
        ),
    ]

    system_instruction = (
        "You are a synthetic dataset generation model which output two text results, question and answer.\n"
        "For generating the question follow these steps :- Analyse the image and write about the activity of the animal, and query about if there are anything wrong with the animal here. remember to act like the owner of the animal, and detect the type of the animal. (you have to act as a synthetic prompt generation model for animals activity and diseases)\n\n"
        "For generating the answer follow these steps:- You are an assistant that writes synthetic veterinary consultation data.\n"
        f"Condition label is {label}.\n"
        "You should tell the user about the condition that is given above and in a well written response included the below details about the condition too. Also make sure to state that its not 100% correct and its according to and AI.\n"
        "Also add information about\nSymptoms\nHome remedies\nPrevention\nEmergency Relief\nVet required or not\n"
        "Respond with JSON containing 'question' and 'answer'."
    )

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            properties={
                "question": genai.types.Schema(type=genai.types.Type.STRING),
                "answer": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
        system_instruction=[types.Part.from_text(text=system_instruction)],
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=gen_config,
    )

    text = getattr(response, "text", None)
    if text is None and hasattr(response, "candidates"):
        text = response.candidates[0].text

    try:
        data = json.loads(text)
        return data.get("question", "").strip(), data.get("answer", "").strip()
    except Exception:
        lines = text.strip().splitlines()
        question = lines[0] if lines else ""
        answer = " ".join(lines[1:]) if len(lines) > 1 else ""
        return question.strip(), answer.strip()


def collect_records(client, dataset_path):
    records = []
    parent = os.path.dirname(dataset_path)
    for root, _, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = os.path.join(root, name)
                rel = os.path.relpath(img_path, parent).replace(os.sep, "/")
                question, answer = generate_qa(client, img_path, label)
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
    parser.add_argument(
        "--out",
        "--output",
        dest="output",
        default=".",
        help="Output directory",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    random.seed(args.seed)

    datasets = {
        "dogs1": collect_records(client, args.dogs1),
        "dogs2": collect_records(client, args.dogs2),
        "cats": collect_records(client, args.cats),
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
