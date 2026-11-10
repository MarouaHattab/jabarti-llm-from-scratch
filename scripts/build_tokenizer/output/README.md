# Generated tokenizer output

This directory is created and populated by `scripts/build_tokenizer/clean_corpus.py`.
Generated Parquet files are intentionally ignored by Git because they can be
large and are reproducible from the Hugging Face dataset.

Run the following command from the repository root:

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe .\scripts\build_tokenizer\clean_corpus.py --max-shards 7 --tokenizer-docs 400000
```

The command downloads the selected pretraining shards and writes:

- `tokenizer_corpus.parquet`: cleaned corpus sample used to train the tokenizer.
- `ft_train_filtered.parquet`: downloaded fine-tuning training split.
- `ft_eval_filtered.parquet`: downloaded fine-tuning evaluation split.

These files are local build outputs and should not be committed.
