import { ChevronDown } from "lucide-react"

import { CopyButton } from "@/components/ui/copy-button"
import type { AnalysisResult } from "@/lib/api"
import { cn } from "@/lib/utils"

function PhaseCard({
  phase,
  title,
  text,
  note,
  primary = false,
}: {
  phase: string
  title: string
  text: string
  note?: string
  primary?: boolean
}) {
  return (
    <section
      className={cn(
        "rounded-2xl border p-4",
        primary ? "border-cyan-400/30 bg-cyan-400/5" : "border-white/10 bg-black/30"
      )}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="font-mono text-[11px] tracking-[0.2em] text-neutral-500 uppercase">{phase}</span>
        <span className="text-xs text-neutral-400">{title}</span>
        {primary ? <CopyButton value={text} className="ml-auto text-neutral-300" aria-label="Copy Phase 3 result" /> : null}
      </div>
      <p className={cn("text-lg break-words", primary ? "font-semibold text-white" : "text-neutral-200")}>
        {text || <em className="text-neutral-500">(no words to correct)</em>}
      </p>
      {note ? <p className="mt-2 text-xs text-neutral-500">{note}</p> : null}
    </section>
  )
}

export function CorrectionView({ result }: { result: AnalysisResult }) {
  const usedSafeguard = result.phase2_steps.some((step) => step.decision.includes("safeguard"))

  return (
    <div className="space-y-3 p-1 text-left">
      {result.phase0.changed ? (
        <p className="text-xs text-neutral-500">
          Correction ran on the Phase 0 output: <span className="text-neutral-300">“{result.phase0.normalized}”</span>
        </p>
      ) : null}
      <PhaseCard phase="Phase 1" title="Edit distance + phonetic G2P" text={result.phase1} />
      <PhaseCard
        phase="Phase 2"
        title="3-layer Bi-GRU + attention"
        text={result.phase2}
        note={
          usedSafeguard
            ? "Lexicon safeguard applied where the GRU output was not trustworthy."
            : "Output passed the sample-lexicon checks."
        }
      />
      <PhaseCard
        phase="Phase 3"
        title="XLM-RoBERTa context ranker"
        text={result.phase3}
        note={result.phase3_trace.decision}
        primary
      />

      {result.phase3_trace.rankings?.length ? (
        <details className="group rounded-2xl border border-white/10 bg-white/[0.03]">
          <summary className="flex cursor-pointer list-none items-center justify-between p-4 text-sm font-medium text-neutral-200">
            How XLM-RoBERTa chose ({result.phase3_trace.rankings.length})
            <ChevronDown className="size-4 transition group-open:rotate-180" aria-hidden />
          </summary>
          <ul className="divide-y divide-white/5 border-t border-white/10">
            {result.phase3_trace.rankings.map((ranking) => (
              <li key={ranking.position} className="px-4 py-2.5 text-sm">
                <p className="text-neutral-300">
                  <span className="font-mono text-neutral-500">{ranking.token}</span>
                  {" → "}
                  <strong className="text-cyan-300">{ranking.chosen}</strong>
                </p>
                <p className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-neutral-500">
                  {ranking.candidates.map((candidate) => (
                    <span key={candidate.word} className={candidate.word === ranking.chosen ? "text-cyan-200" : undefined}>
                      {candidate.word} <span className="tabular-nums">({candidate.score.toFixed(1)})</span>
                    </span>
                  ))}
                </p>
              </li>
            ))}
          </ul>
          <p className="border-t border-white/10 px-4 py-2 text-xs text-neutral-500">
            Score = how well the word fits the sentence (XLM-RoBERTa) + closeness to what you typed + lexicon bonus.
          </p>
        </details>
      ) : null}

      {result.timings_ms ? (
        <p className="text-xs text-neutral-500">
          Time: Phase 1 {result.timings_ms.phase1.toFixed(0)} ms · Phase 2 {result.timings_ms.phase2.toFixed(0)} ms · Phase 3{" "}
          {result.timings_ms.phase3.toFixed(0)} ms
        </p>
      ) : null}

      {result.phase1_steps.length ? (
        <details className="group rounded-2xl border border-white/10 bg-white/[0.03]">
          <summary className="flex cursor-pointer list-none items-center justify-between p-4 text-sm font-medium text-neutral-200">
            Word-by-word decisions ({result.phase1_steps.length})
            <ChevronDown className="size-4 transition group-open:rotate-180" aria-hidden />
          </summary>
          <ul className="divide-y divide-white/5 border-t border-white/10">
            {result.phase1_steps.map((step, index) => {
              const changed = step.original !== step.corrected
              return (
                <li key={index} className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 text-sm">
                  <span className="text-neutral-300">
                    {step.original}
                    {changed ? (
                      <>
                        {" → "}
                        <strong className="text-cyan-300">{step.corrected}</strong>
                      </>
                    ) : null}
                  </span>
                  <span className="text-xs text-neutral-500" title={step.candidates.join(", ")}>
                    {step.distance === 0 ? "in lexicon" : `${step.distance} edit${step.distance === 1 ? "" : "s"}`} ·{" "}
                    {step.decision}
                  </span>
                </li>
              )
            })}
          </ul>
        </details>
      ) : null}
    </div>
  )
}
