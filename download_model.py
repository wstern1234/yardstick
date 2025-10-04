import argparse
import os
from huggingface_hub import hf_hub_download

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_bert(models_dir: str):
    print("Downloading BERT ONNX model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id="google-bert/bert-base-uncased",
        filename="model.onnx"
    )
    target_path = os.path.join(models_dir, "bert.onnx")

    if not os.path.exists(target_path):
        os.rename(model_path, target_path)
    print(f"BERT model available at: {target_path}")

def main():
    parser = argparse.ArgumentParser(description="Download large models for yardstick benchmarks.")
    parser.add_argument("--model", type=str, choices=["bert", "all"], default="all",
                        help="Which model to download (bert, etc.)")
    parser.add_argument("--outdir", type=str, default="models",
                        help="Directory to store downloaded models")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # all flag for future proofing in case more models are supported
    if args.model in ("bert", "all"):
        download_bert(args.outdir)

    print("✅ Done.")

if __name__ == "__main__":
    main()
