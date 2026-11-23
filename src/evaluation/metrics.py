import jiwer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util


class EvaluationSuite:
    """Calculates CER, WER, BLEU, Exact Match, and Cosine Semantic Similarity."""

    def __init__(self, sem_model_name: str = "all-MiniLM-L6-v2"):
        self.sem_model = None
        self.smooth_fn = SmoothingFunction().method1
        try:
            self.sem_model = SentenceTransformer(sem_model_name)
        except Exception:
            self.sem_model = None

    def calculate_cer(self, references: list[str], predictions: list[str]) -> float:
        return float(jiwer.cer(references, predictions))

    def calculate_wer(self, references: list[str], predictions: list[str]) -> float:
        return float(jiwer.wer(references, predictions))

    def calculate_bleu(self, references: list[str], predictions: list[str]) -> float:
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = [ref.lower().split()]
            pred_tokens = pred.lower().split()
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=self.smooth_fn,
            )
            scores.append(score)
        return float(sum(scores) / len(scores)) if scores else 0.0

    def calculate_exact_match(self, references: list[str], predictions: list[str]) -> float:
        correct = sum(
            1
            for r, p in zip(references, predictions)
            if r.strip().lower() == p.strip().lower()
        )
        return (correct / len(references)) * 100.0 if references else 0.0

    @staticmethod
    def _token_overlap_similarity(reference: str, prediction: str) -> float:
        ref_tokens = set(reference.lower().split())
        pred_tokens = set(prediction.lower().split())
        if not ref_tokens and not pred_tokens:
            return 1.0
        if not ref_tokens or not pred_tokens:
            return 0.0
        intersection = len(ref_tokens & pred_tokens)
        union = len(ref_tokens | pred_tokens)
        return intersection / union if union else 0.0

    def calculate_semantic_similarity(self, references: list[str], predictions: list[str]) -> float:
        if self.sem_model is not None:
            try:
                ref_embs = self.sem_model.encode(references, convert_to_tensor=True)
                pred_embs = self.sem_model.encode(predictions, convert_to_tensor=True)
                cosine_scores = util.cos_sim(ref_embs, pred_embs)
                sim_scores = [cosine_scores[i][i].item() for i in range(len(references))]
                return float(sum(sim_scores) / len(sim_scores)) if sim_scores else 0.0
            except Exception:
                pass

        scores = [
            self._token_overlap_similarity(ref, pred)
            for ref, pred in zip(references, predictions)
        ]
        return float(sum(scores) / len(scores)) if scores else 0.0

    @staticmethod
    def calculate_detection_scores(
        references: list[str], inputs: list[str], predictions: list[str]
    ) -> dict[str, float]:
        """Token-level precision, recall and F1 of error detection *and* correction.

        Following the paper, for every word (reference ``r``, noisy input ``n``,
        system output ``p``):

        * TP: the input was wrong (``n != r``) and the output is right (``p == r``);
        * FP: the system changed the word (``p != n``) but got it wrong (``p != r``),
          which includes changing a valid word;
        * FN: the input was wrong and the output is still wrong (``p != r``).

        A system that changes nothing has precision 0 by convention (no changes made).
        """
        true_pos = false_pos = false_neg = 0
        for ref_sentence, noisy_sentence, pred_sentence in zip(references, inputs, predictions):
            ref_words = ref_sentence.lower().split()
            noisy_words = noisy_sentence.lower().split()
            pred_words = pred_sentence.lower().split()
            for index, ref in enumerate(ref_words):
                noisy = noisy_words[index] if index < len(noisy_words) else ""
                pred = pred_words[index] if index < len(pred_words) else ""
                was_wrong = noisy != ref
                if was_wrong and pred == ref:
                    true_pos += 1
                elif was_wrong:
                    false_neg += 1
                if pred != noisy and pred != ref:
                    false_pos += 1
        precision = true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
        recall = true_pos / (true_pos + false_neg) if true_pos + false_neg else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"Precision": precision, "Recall": recall, "F1": f1}

    def evaluate_all(
        self,
        references: list[str],
        predictions: list[str],
        inputs: list[str] | None = None,
    ) -> dict:
        """Compute every metric; pass the noisy ``inputs`` to add precision/recall/F1."""
        extra = (
            {k: round(v, 4) for k, v in self.calculate_detection_scores(references, inputs, predictions).items()}
            if inputs is not None
            else {}
        )
        return {
            **extra,
            "CER": round(self.calculate_cer(references, predictions), 4),
            "WER": round(self.calculate_wer(references, predictions), 4),
            "BLEU": round(self.calculate_bleu(references, predictions), 4),
            "Exact_Match (%)": round(self.calculate_exact_match(references, predictions), 2),
            "Semantic_Sim": round(self.calculate_semantic_similarity(references, predictions), 4),
        }
