import argparse
import os
import random
import json
import time

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.0-flash"

RATE_LIMIT_PER_MIN = 12
_CALL_INTERVAL = 60.0 / RATE_LIMIT_PER_MIN
_last_call_time = 0.0
MAX_RETRIES = 3


def generate_qa(client, image_path, label):
    global _last_call_time

    with open(image_path, "rb") as f:
        data = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data=genai.types.Blob(
                        mime_type="image/jpeg",
                        data=data,
                    )
                )
            ],
        ),
    ]

    system_instruction = (
        "You are a synthetic dataset generation model which output two text results, question and answer in the json structured format.\n"
        "For generating the question follow these steps :- Analyse the image and write about the activity of the animal, and query about if there are anything wrong with the animal here. remember to act like the owner of the animal, and detect the type of the animal. (you have to act as a synthetic prompt generation model for animals activity and diseases) Remember to write a short question as people does in a chatbot max 100 words.\n\n"
        "For generating the answer follow these steps:- You are an assistant that writes synthetic veterinary consultation data.\n"
        f"        Condition label is {label}.\n"
        "       You should tell the user about the condition that is given above and in a well written response included the below details about the condition too. Also make sure to state that its not 100% correct and its according to and AI.\n"
        "        Also add information about \n       Symptoms\n       Home remedies \n       Prevention\n       Emergency Relief\n       Vet required or not\n\n"
        "now return the answers in the below format:-\n{\n\"question\": <question>\n\"answer\": <answer>\n}\n\n"
        "Please avoid adding blank lines or \\n\\n at the end of your response."
    )

    gen_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["question", "answer"],
            properties={
                "question": genai.types.Schema(type=genai.types.Type.STRING),
                "answer": genai.types.Schema(type=genai.types.Type.STRING),
            },
        ),
        system_instruction=[types.Part.from_text(text=system_instruction)],
    )

    for attempt in range(MAX_RETRIES):
        wait = _last_call_time + _CALL_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        start = time.time()
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=gen_config,
            )
        except Exception:
            _last_call_time = start
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
            continue
        _last_call_time = start
        text = getattr(response, "text", None)
        if text is None and hasattr(response, "candidates"):
            text = response.candidates[0].text
        if text is None:
            text = ""
        text = text.strip()

        try:
            data = json.loads(text)
            q = data.get("question", "").strip()
            a = data.get("answer", "").strip()
            if not q or not a:
                raise ValueError("empty response")
            return q, a
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
            continue

    raise RuntimeError("Failed to generate QA after retries")


def collect_records(client, dataset_path, checkpoint_file, max_records=None):
    records = []
    processed = set()
    parent = os.path.dirname(dataset_path)
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    img = rec.get("image")
                    if img:
                        if not os.path.isabs(img):
                            img = os.path.abspath(os.path.join(parent, img))
                        img = img.replace(os.sep, "/")
                        rec["image"] = img
                        processed.add(img)
                    records.append(rec)
                    if max_records and len(records) >= max_records:
                        return records

    for root, _, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                img_path = os.path.join(root, name)
                abs_path = os.path.abspath(img_path).replace(os.sep, "/")
                if abs_path in processed:
                    continue
                if max_records and len(records) >= max_records:
                    return records
                question, answer = generate_qa(client, img_path, label)
                rec = {
                    "image": abs_path,
                    "conversations": [
                        {"from": "human", "value": question},
                        {"from": "gpt", "value": answer},
                    ],
                }
                records.append(rec)
                with open(checkpoint_file, "a") as cp:
                    json.dump(rec, cp)
                    cp.write("\n")
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
    parser.add_argument(
        "--checkpoint-dir",
        default=".checkpoints",
        help="Directory to store intermediate records",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum total images to process. 0 means no limit",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    client = genai.Client(api_key=api_key)

    random.seed(args.seed)

    cp_dir = os.path.join(args.output, args.checkpoint_dir)
    os.makedirs(cp_dir, exist_ok=True)

    paths = {
        "dogs1": args.dogs1,
        "dogs2": args.dogs2,
        "cats": args.cats,
    }

    per_limit = None
    if args.max_images:
        per_limit = max(1, args.max_images // len(paths))

    datasets = {}
    for name, path in paths.items():
        datasets[name] = collect_records(
            client,
            path,
            os.path.join(cp_dir, f"{name}.jsonl"),
            max_records=per_limit,
        )

    limit = min(len(v) for v in datasets.values())
    if args.max_images:
        limit = min(limit, max(1, args.max_images // len(datasets)))
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
