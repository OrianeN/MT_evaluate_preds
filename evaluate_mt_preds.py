"""Script to run evaluations of translations
in cases where you don't have access to the model (or even for human translation),
but only the predictions (+ references/sources)

Usage: `python3 evaluate_mt_preds.py -s source_file.txt -p pred_file.txt -r ref_file.txt`

Metrics implemented: BLEU, COMET
"""

import argparse

from comet import download_model as download_comet_model
from comet import load_from_checkpoint as load_comet_from_checkpoint
from sacrebleu.metrics import BLEU


def compute_metrics(predictions, references, sources, comet_model):
    metrics = {}

    # Compute BLEU score
    metrics["BLEU"] = BLEU().corpus_score(predictions, [references]).score

    # Compute COMET score
    if comet_model:
        comet_score = comet_model.predict([
            {"src": src_seg, "mt": mt_seg, "ref": ref_seg}
            for src_seg, mt_seg, ref_seg
            in zip(sources, predictions, references)
        ]).system_score
        metrics["COMET"] = 100 * comet_score

    return metrics


def load_segments_from_file(path):
    with open(path) as f:
        return [l.strip() for l in f]


def load_comet_model(model_name):
    comet_model_path = download_comet_model(model_name)
    return load_comet_from_checkpoint(comet_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_file",
                        help="Path to the file with source segments",
                        required=True)
    parser.add_argument("-p", "--pred_file",
                        help="Path to the file with predicted translations",
                        required=True)
    parser.add_argument("-r", "--ref_file",
                        help="Path to the file with reference translations",
                        required=True)
    parser.add_argument("-cm", "--comet_model_name",
                        help="Name of the COMET model to run evaluation with",
                        default="Unbabel/wmt22-comet-da")
    parser.add_argument("--no_comet", action="store_true",
                        help="Disabling scoring with COMET")

    args = parser.parse_args()

    print(f"Source file: {args.source_file}")
    print(f"Predictions file: {args.pred_file}")
    print(f"References file: {args.ref_file}")

    comet_model = load_comet_model(args.comet_model_name) if not args.no_comet else None
    source_segments = load_segments_from_file(args.source_file)
    preds_segments = load_segments_from_file(args.pred_file)
    ref_segments = load_segments_from_file(args.ref_file)

    results = compute_metrics(preds_segments, ref_segments, source_segments, comet_model)

    print("\nResults:")
    for metric, score in results.items():
        print(f"\t- {metric}: {score:.2f}")
