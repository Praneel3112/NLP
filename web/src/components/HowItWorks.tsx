import { HaloProgress } from "@/components/ui/halo-progress"
import type { EmotionMetrics } from "@/lib/api"
import { percent } from "@/lib/emotions"
import { cn } from "@/lib/utils"

const STAGES = [
  { step: "00", title: "Slang normalizer", text: "Expands short forms (bht → bahut, plz → please) and separates emoji." },
  {
    step: "01",
    title: "Edit distance + G2P",
    text: "English words are skipped by language ID. Indic words are matched to a 279-word lexicon by edit distance and by sound.",
  },
  {
    step: "02",
    title: "Bi-GRU + attention",
    text: "A 3-layer bidirectional GRU with Bahdanau attention, trained on 1,117 noisy-to-clean pairs, fixes words letter by letter.",
  },
  {
    step: "03",
    title: "XLM-RoBERTa ranker",
    text: "A pretrained multilingual transformer picks the candidate that fits the whole sentence best.",
  },
]

const COMPARISON = [
  { key: "rules_only", label: "Rule engine only", note: "emoji + slang + word dictionaries" },
  { key: "model_only", label: "ML model only", note: "TF-IDF + Logistic Regression" },
  { key: "hybrid", label: "Hybrid (ours)", note: "50% rules + 50% model" },
] as const

export function HowItWorks({ metrics }: { metrics: EmotionMetrics | null }) {
  return (
    <section id="how-it-works" aria-labelledby="how-title" className="mx-auto max-w-6xl px-4 py-20">
      <p className="font-mono text-xs tracking-[0.2em] text-cyan-300/80 uppercase">How it works</p>
      <h2 id="how-title" className="mt-2 text-3xl font-semibold text-neutral-100 sm:text-4xl">
        Two branches, one message
      </h2>
      <p className="mt-3 max-w-2xl text-neutral-400">
        Every message goes down two paths at once: a correction pipeline for the words, and an emotion detector for the
        feeling behind them.
      </p>

      <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {STAGES.map((stage) => (
          <article key={stage.step} className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
            <span className="font-mono text-xs text-cyan-300/80">PHASE {stage.step}</span>
            <h3 className="mt-2 font-semibold text-neutral-100">{stage.title}</h3>
            <p className="mt-1 text-sm text-neutral-400">{stage.text}</p>
          </article>
        ))}
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-[1fr_1.2fr]">
        <article className="rounded-2xl border border-pink-400/20 bg-pink-400/[0.04] p-5">
          <span className="font-mono text-xs text-pink-300/80">EMOTION BRANCH</span>
          <h3 className="mt-2 font-semibold text-neutral-100">Hybrid emotion detector</h3>
          <ul className="mt-3 space-y-2 text-sm text-neutral-400">
            <li>
              <strong className="text-neutral-200">Rule engine:</strong> about 150 emoji, chat slang and emotion words in
              English, Hindi, Kannada, Tamil and Bengali. Handles negation (“not happy”, “khush nahi”) and intensity
              (CAPS, !!!, sooo, bahut, tumba, romba).
            </li>
            <li>
              <strong className="text-neutral-200">ML model:</strong> word and character TF-IDF features with Logistic
              Regression, trained on our own labelled code-mixed dataset.
            </li>
            <li>
              <strong className="text-neutral-200">Combined:</strong> 50/50 average. If no dictionary clue is found, the
              model decides alone.
            </li>
          </ul>
        </article>

        <article className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
          <span className="font-mono text-xs text-neutral-500">EVALUATION</span>
          <h3 className="mt-2 font-semibold text-neutral-100">Accuracy on unseen sentences</h3>
          {metrics ? (
            <>
              <p className="mt-1 text-sm text-neutral-400">
                Trained on {metrics.train_size} sentences, tested on {metrics.test_size} it never saw (
                {metrics.dataset_size} labelled in total).
              </p>
              <div className="mt-4 space-y-4">
                {COMPARISON.map(({ key, label, note }) => (
                  <div key={key}>
                    <div className="mb-1.5 flex items-baseline justify-between gap-3 text-sm">
                      <span className={cn(key === "hybrid" ? "font-semibold text-cyan-200" : "text-neutral-300")}>
                        {label} <span className="text-xs text-neutral-500">· {note}</span>
                      </span>
                      <span className="text-neutral-300 tabular-nums">
                        {percent(metrics.comparison[key].accuracy, 1)}
                        <span className="ml-2 text-xs text-neutral-500">
                          F1 {metrics.comparison[key].macro_f1.toFixed(2)}
                        </span>
                      </span>
                    </div>
                    <HaloProgress
                      value={Math.round(metrics.comparison[key].accuracy * 100)}
                      aria-label={`${label} accuracy`}
                      indicatorClassName={
                        key === "hybrid" ? "bg-gradient-to-r from-cyan-300 to-cyan-500" : "bg-gradient-to-r from-neutral-400 to-neutral-500"
                      }
                    />
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs text-neutral-500">
                ML model 5-fold cross-validation accuracy: {percent(metrics.cv_accuracy_mean, 1)} ±{" "}
                {percent(metrics.cv_accuracy_std, 1)}.
              </p>
            </>
          ) : (
            <p className="mt-2 text-sm text-neutral-500">Metrics appear once the backend has trained the model.</p>
          )}
        </article>
      </div>
    </section>
  )
}
