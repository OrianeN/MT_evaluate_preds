"""Script to run evaluations of translations
in cases where you don't have access to the model (or even for human translation),
but only the predictions (+ references/sources)

requires numpy package: `pip install numpy`

Usage: `python3 evaluate_mt_preds.py -s source_file.txt -p pred_file.txt -r ref_file.txt`

Metrics implemented: BLEU, COMET
"""

import argparse
from collections import defaultdict

import numpy as np
from comet import download_model as download_comet_model
from comet import load_from_checkpoint as load_comet_from_checkpoint
from sacrebleu.metrics import BLEU


def compute_metrics(predictions, references, sources, comet_model):
    """
    Compute scores given a triplets with predictions, references and sources.

    Metrics used are BLEU and COMET (only if a model is passed).

    Computes scores at the sentence level, and returns both the global and sentence-specific scores.
    For BLEU, this means computing the scores twice, with `.corpus_score` and `sentence_score`.

    :param predictions: list of predicted translations
    :param references: list of reference translations
    :param sources: list of source segments
    :param comet_model: loaded COMET model
    :return:
    - dict with global scores for each metric
    - dict with scores for each sentence and for each metric,
        in the format {"BLEU": [score1, score2...], "COMET": [score1, score2...]}
    """
    metrics_global = {}
    metrics_sentences = defaultdict(list)

    # Compute BLEU score at corpus-level
    metrics_global["BLEU"] = BLEU().corpus_score(predictions, [references]).score

    # Compute BLEU score for each sentence
    bleu_metric = BLEU(effective_order=True)
    for pred, ref in zip(predictions, references):
        metrics_sentences["BLEU"].append(bleu_metric.sentence_score(pred, [ref]).score)

    # Compute COMET score
    if comet_model:
        comet_result = comet_model.predict([
            {"src": src_seg, "mt": mt_seg, "ref": ref_seg}
            for src_seg, mt_seg, ref_seg
            in zip(sources, predictions, references)
        ])
        metrics_global["COMET"] = 100 * comet_result.system_score
        metrics_sentences["COMET"] = [100 * s for s in comet_result.scores]

    return metrics_global, metrics_sentences


def print_sentence_scores_stats(sent_scores):
    """
    Prints detailed stats for the sentence-specific scores of each metric
    :param sent_scores: dict of format {"BLEU": [score1, score2, ...], "COMET": [score1, score2, ...]}
    """
    for metric_name, metric_scores in sent_scores.items():
        print(f"\nStats for {metric_name}:")
        np_scores = np.asarray(metric_scores)
        print(f"\t-Mean: {np_scores.mean():.4f}")
        print(f"\t-Median: {np.median(np_scores):.4f}")
        print(f"\t-Min: {np_scores.min():.4f}")
        print(f"\t-Max: {np_scores.max():.4f}")
        print(f"\t-Standard deviation: {np.std(np_scores):.4f}")


def load_segments_from_file(path):
    """
    Loads segments from a TXT file, where each line contains one segment
    :return: list of segments
    """
    with open(path) as f:
        return [l.strip() for l in f]


def load_comet_model(model_name):
    """
    (Downloads and) loads COMET model
    """
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
    parser.add_argument("-o", "--output_triplet_scores",
                        help="Path to the output TSV file where to write sentence triplets with their scores",
                        default="output_triplet_scores.tsv")
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

    results_global, results_sentences = compute_metrics(preds_segments, ref_segments, source_segments, comet_model)

    # Write results to stdout
    print("\nResults:")
    for metric, score in results_global.items():
        print(f"\t- {metric}: {score:.2f}")
    print_sentence_scores_stats(results_sentences)

    # Export scores and triplets into a TSV file
    with open(args.output_triplet_scores, "w", encoding="utf-8") as f:
        # Write header
        for metric_name in results_sentences.keys():
            f.write(metric_name+"\t")
        f.write("Source segment\tPredicted translation\tReference translation\n")
        # Write all triplets along with their scores
        for i, (src_seg, pred_seg, ref_seg) in enumerate(zip(source_segments, preds_segments, ref_segments)):
            # Write scores
            for metric_scores in results_sentences.values():
                f.write(f"{metric_scores[i]:.4f}\t")
            # Write triplet
            f.write(f"{src_seg}\t{pred_seg}\t{ref_seg}\n")

    print(f"\nSentence-level scores stored in {args.output_triplet_scores}")
