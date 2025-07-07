# Dataset Conversion

`convert_datasets.py` generates conversation data for image datasets using the Gemini Flash API.
The script respects the API rate limit by issuing at most 12 requests per minute
and stores progress in checkpoint files so interrupted runs can be resumed.

## Requirements

- Python 3
- [`google-genai`](https://pypi.org/project/google-genai/)
- Environment variable `GEMINI_API_KEY` with a valid API key.

Install the dependency:

```bash
pip install google-genai
```

## Usage

Each dataset directory should contain subfolders named after their labels. Images
inside a subfolder are assumed to have that label.

Run the converter:

```bash
python convert_datasets.py --dogs1 /path/to/dogs1 \
                           --dogs2 /path/to/dogs2 \
                           --cats  /path/to/cats \
                           --out data/qwen25-dataset
```

Checkpoint files are stored in `--out/.checkpoints`. Rerun the command to
resume generation if it stops.

The script mixes and shuffles the examples from all datasets, then writes
`train.jsonl` and `val.jsonl` to the output directory.
