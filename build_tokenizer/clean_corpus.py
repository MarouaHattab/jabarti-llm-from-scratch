# this file will download and filter the corpus 

# step 1 : download and filter

# step 2 : Type A text cleaning 

from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd
from cleaning import clean_dataframe,normalize_text,prepare_document

from data_exploration.config import OUTPUT_DIR
REPO = "bakrianoo/jabarti-llm-dataset"
MAX_CHUNKS=20 # to make balance between articles and article 
HF_FILES = {
    "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
    "phase1_eval": "pretrain/phase1_eval-00000-of-00001.parquet",

    "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
    "phase2_eval": "pretrain/phase2_eval-00000-of-00001.parquet",

    "ft_train": "finetune/train-00000-of-00001.parquet",
    "ft_eval": "finetune/eval-00000-of-00001.parquet",
}
TRAIN_SPLITS=  ["phase1_train", "phase2_train"]
EVAL_SPLITS=  ["phase1_eval", "phase2_eval"]

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def demo_normalizer():
    print("TYPE B NORMALIZER (embeddable, runs at inference too)")

    samples = [
        "جُمْهُورِيَّةُ مِصْرَ الْعَرَبِيَّة",      # heavy diacritics
        "أحمد إبراهيم آدم علىّ",                   # alef + ya variants
        "The   United    States",                 # extra spaces
    ]

    for s in samples:
        print(f"  raw       : {s}")
        print(f"  normalized: {normalize_text(s)}")
        print("="*30)


def download_and_filter():
# Download raw hf splits , apply hard cap to train splits , write *_filtered.parquet

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    for name,filename in HF_FILES.items():
        print(f"loading {name} from {filename}")
        local=hf_hub_download(
            repo_id=REPO,
            repo_type="dataset", 
            filename=filename,
            )
        df=pd.read_parquet(local)
        if name in TRAIN_SPLITS:
            before_filter=len(df)

            df=(
                df.sort_values(by=["article_id","chunk_index"])
                .groupby("article_id",sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
            )
            after_filter=len(df)
            print(f"df : {name} => {before_filter} filtered to be  {after_filter} rows")
        output_path=OUT_DIR/ f"{name}_filtered.parquet"
        df.to_parquet(output_path, index=False)

def clean_phase(phase,input_path,output_path):
    df = pd.read_parquet(input_path)
    print(f" Souce : {input_path}")
    print(f" Rows : {len(df) : ,}")

    cleaned_df = clean_dataframe(df)
    removed_rows = len(df) - len(cleaned_df)

    print(f" Rows kept : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)



def main():
    download_and_filter()

    for name in TRAIN_SPLITS:
        input_path=OUT_DIR/ f"{name}_filtered.parquet"
        output_path=OUT_DIR/ f"{name}_cleaned.parquet"
        clean_phase(phase=name,input_path=input_path,output_path=output_path)


    for name in EVAL_SPLITS:
        input_path=OUT_DIR/ f"{name}_filtered.parquet"
        output_path=OUT_DIR/ f"{name}_cleaned.parquet"
        clean_phase(phase=name,input_path=input_path,output_path=output_path)

 

if __name__ == "__main__":
    main()
   
