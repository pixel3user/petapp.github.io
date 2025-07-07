# Dataset Conversion

`convert_datasets.py` generates conversation data for image datasets using the Gemini Flash API.

## Requirements

- Python 3
- [`google-generativeai`](https://pypi.org/project/google-generativeai/)
- Environment variable `GEMINI_API_KEY` with a valid API key.

Install the dependency:

```bash
pip install google-generativeai
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

The script mixes and shuffles the examples from all datasets, then writes
`train.jsonl` and `val.jsonl` to the output directory.
