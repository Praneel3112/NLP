import { ArrowRight } from "lucide-react"

import type { Phase0Result } from "@/lib/api"
import { cn } from "@/lib/utils"

export function SlangView({ phase0 }: { phase0: Phase0Result }) {
  const replaced = new Map(phase0.replacements.map((item) => [item.position, item]))
  const emojiPositions = new Set(phase0.emoji.map((item) => item.position))
  const tokens = phase0.input.split(/\s+/).filter(Boolean)

  return (
    <div className="space-y-6 p-1 text-left">
      <p className="text-sm text-neutral-400">
        <strong className="text-neutral-200">Phase 0</strong> expands chat slang and short forms, and takes emoji out
        of the text, before spelling correction. Emoji go to the emotion detector instead.
      </p>

      <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr] md:items-center">
        <div className="rounded-2xl border border-white/10 bg-black/30 p-4">
          <p className="mb-2 font-mono text-[11px] tracking-[0.2em] text-neutral-500 uppercase">You typed</p>
          <p className="flex flex-wrap gap-x-1.5 gap-y-1 text-lg text-neutral-200">
            {tokens.map((token, index) => (
              <span
                key={index}
                className={cn(
                  replaced.has(index) && "rounded bg-amber-400/15 px-1 text-amber-200",
                  emojiPositions.has(index) && !replaced.has(index) && "rounded bg-pink-400/10 px-1"
                )}
              >
                {token}
              </span>
            ))}
          </p>
        </div>
        <ArrowRight className="mx-auto size-5 rotate-90 text-neutral-500 md:rotate-0" aria-hidden />
        <div className="rounded-2xl border border-cyan-400/20 bg-cyan-400/5 p-4">
          <p className="mb-2 font-mono text-[11px] tracking-[0.2em] text-cyan-300/70 uppercase">Sent to correction</p>
          <p className="text-lg text-neutral-100">{phase0.normalized || <em className="text-neutral-500">(nothing: only emoji)</em>}</p>
        </div>
      </div>

      {!phase0.changed ? (
        <p className="rounded-xl border border-white/10 bg-white/5 p-3 text-sm text-neutral-400">
          No slang or emoji found. The text goes to Phase 1 exactly as typed.
        </p>
      ) : null}

      {phase0.replacements.length ? (
        <section className="space-y-2">
          <h4 className="text-sm font-semibold text-neutral-200">Slang expanded</h4>
          <ul className="grid gap-2 sm:grid-cols-2">
            {phase0.replacements.map((item, index) => (
              <li key={index} className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm">
                <span className="font-mono text-amber-200">{item.original}</span>
                <ArrowRight className="size-3.5 text-neutral-500" aria-hidden />
                <span className="font-medium text-neutral-100">{item.replacement}</span>
                <span className="ml-auto rounded-full border border-white/10 px-2 py-0.5 text-[10px] text-neutral-400 uppercase">
                  {item.kind === "indic" ? "Indic short form" : "chat slang"}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {phase0.emoji.length ? (
        <section className="space-y-2">
          <h4 className="text-sm font-semibold text-neutral-200">Emoji passed to emotion detection</h4>
          <ul className="flex flex-wrap gap-2">
            {phase0.emoji.map((item, index) => (
              <li key={index} className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-sm text-neutral-300">
                <span className="text-xl">{item.emoji}</span>
                {item.name}
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  )
}
