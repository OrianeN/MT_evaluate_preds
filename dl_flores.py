"""Simple script to download Flores-200 dataset for a particular language pair

requires HuggingFace's dataset library: `pip install datasets`

Usage example: `python dl_flores.py fra_Latn oci_Latn flores_fr_oc.tsv`

Output format can be TSV or TXT (2 files)
"""
from datasets import load_dataset


def norm_whitespaces(text):
    text = " ".join(text.splitlines())
    text = " ".join(text.split())
    return text


def hf_dataset_to_tsv(hf_dataset, tsv_path, src_lang, tgt_lang):
    with open(tsv_path, "w", encoding="utf-8") as f:
        for sample in hf_dataset:
            src_sample = norm_whitespaces(sample["sentence_"+src_lang])
            tgt_sample = norm_whitespaces(sample["sentence_"+tgt_lang])
            f.write(f"{src_sample}\t{tgt_sample}\n")


def hf_dataset_to_txt(hf_dataset, out_src_path, out_tgt_path, src_lang, tgt_lang):
    with open(out_src_path, "w", encoding="utf-8") as f_src, open(out_tgt_path, "w", encoding="utf-8") as f_tgt:
        for sample in hf_dataset:
            src_sample = norm_whitespaces(sample["sentence_"+src_lang])
            tgt_sample = norm_whitespaces(sample["sentence_"+tgt_lang])
            f_src.write(f"{src_sample}\n")
            f_tgt.write(f"{tgt_sample}\n")


if __name__ == "__main__":
    import argparse

    TSV_EXT = "tsv"
    TXT_EXT = "txt"

    parser = argparse.ArgumentParser()
    parser.add_argument("src_lang", help="Source language code")
    parser.add_argument("tgt_lang", help="Target language code")
    parser.add_argument("output_path", help="Path to the output TSV file")
    parser.add_argument("--output_format", choices=[TSV_EXT, TXT_EXT], default=TSV_EXT)
    parser.add_argument("--split", default="devtest")

    args = parser.parse_args()

    # Load dataset
    hf_dataset_params = {
        "path": "facebook/flores",
        "name": f"{args.src_lang}-{args.tgt_lang}",
        "split": args.split,
    }
    hf_dataset = load_dataset(**hf_dataset_params, trust_remote_code=True)

    # Export dataset to file(s)
    if args.output_format == TSV_EXT:
        if not args.output_path.endswith("." + args.output_format):
            args.output_path += "." + args.output_format
        hf_dataset_to_tsv(hf_dataset, args.output_path, args.src_lang, args.tgt_lang)
    else:  # txt
        out_source = args.output_path + args.src_lang + "." + args.output_format
        out_target = args.output_path + args.tgt_lang + "." + args.output_format
        hf_dataset_to_txt(hf_dataset, out_source, out_target, args.src_lang, args.tgt_lang)
