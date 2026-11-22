import { HaloBadge } from "@/components/ui/halo-badge"
import { HaloProgress } from "@/components/ui/halo-progress"
import { RollingNumber } from "@/components/ui/rolling-number"
import { TextAnimate } from "@/components/ui/text-animate"
import type { EmotionResult } from "@/lib/api"
import { EMOTION_ORDER, EMOTION_STYLE, percent } from "@/lib/emotions"
import { cn } from "@/lib/utils"

const INTENSITY_LABEL = { none: "No intensity cues", low: "Low intensity", medium: "Medium intensity", high: "High intensity" }
const TYPE_LABEL = { emoji: "emoji", slang: "slang", word: "word" }

export function EmotionView({ emotion, input }: { emotion: EmotionResult | null; input: string }) {
  if (!emotion) {
    return <p className="p-6 text-sm text-neutral-400">The emotion model is still training. Try again in a moment.</p>
  }

  const style = EMOTION_STYLE[emotion.label]
  const ordered = [...EMOTION_ORDER].sort((a, b) => emotion.scores[b] - emotion.scores[a])

  return (
    <div className="space-y-6 p-1 text-left">
      {/* Headline result */}
      <div className="flex flex-col items-center gap-4 rounded-2xl border border-white/10 bg-black/30 p-5 sm:flex-row sm:items-center">
        <span className="text-6xl leading-none" aria-hidden>
          {style.emoji}
        </span>
        <div className="min-w-0 flex-1 text-center sm:text-left">
          <p className="font-mono text-[11px] tracking-[0.2em] text-neutral-500 uppercase">Detected emotion</p>
          <TextAnimate
            key={`${input}-${emotion.label}`}
            text={style.label}
            type="popIn"
            className={cn("m-0 justify-center p-0 text-4xl font-bold sm:justify-start", style.text)}
          />
          <div className="mt-2 flex flex-wrap justify-center gap-2 sm:justify-start">
            <HaloBadge variant="secondary">
              {emotion.method === "hybrid" ? "Rules + ML model" : "ML model only (no dictionary cues)"}
            </HaloBadge>
            <HaloBadge variant={emotion.intensity === "high" ? "destructive" : "secondary"}>
              {INTENSITY_LABEL[emotion.intensity]}
            </HaloBadge>
          </div>
        </div>
        <div className="text-center sm:text-right">
          <p className={cn("text-4xl font-bold tabular-nums", style.text)}>
            <RollingNumber value={Math.round(emotion.confidence * 100)} format={(n) => `${n}%`} />
          </p>
          <p className="text-xs text-neutral-500">confidence</p>
        </div>
      </div>

      {/* Score bars */}
      <section aria-label="Score for every emotion" className="space-y-3">
        <h4 className="text-sm font-semibold text-neutral-200">How sure is it about each emotion?</h4>
        {ordered.map((name) => (
          <div key={name} className="grid grid-cols-[6.5rem_1fr_3rem] items-center gap-3">
            <span className="text-sm text-neutral-300">
              {EMOTION_STYLE[name].emoji} {EMOTION_STYLE[name].label}
            </span>
            <HaloProgress
              value={Math.round(emotion.scores[name] * 100)}
              aria-label={`${EMOTION_STYLE[name].label} score`}
              indicatorClassName={cn("bg-gradient-to-r", EMOTION_STYLE[name].bar)}
              className="min-w-0"
            />
            <span className="text-right text-xs text-neutral-400 tabular-nums">{percent(emotion.scores[name])}</span>
          </div>
        ))}
        <p className="text-xs text-neutral-500">
          {emotion.method === "hybrid"
            ? "Final score = 50% rule engine + 50% trained model."
            : "No emoji, slang or emotion word was found, so the trained model decided alone."}{" "}
          Rules said <strong className="text-neutral-300">{percent(emotion.rule_scores[emotion.label])}</strong>, model
          said <strong className="text-neutral-300">{percent(emotion.model_scores[emotion.label])}</strong> for{" "}
          {EMOTION_STYLE[emotion.label].label.toLowerCase()}.
        </p>
      </section>

      {/* Evidence */}
      <section aria-label="Why this emotion" className="space-y-3">
        <h4 className="text-sm font-semibold text-neutral-200">Why? The clues it found</h4>
        {emotion.evidence.length === 0 ? (
          <p className="text-sm text-neutral-400">No dictionary clues. Decided by the trained model.</p>
        ) : (
          <ul className="flex flex-wrap gap-2">
            {emotion.evidence.map((item, index) => (
              <li
                key={`${item.token}-${index}`}
                className={cn("rounded-xl border px-3 py-2 text-sm", EMOTION_STYLE[item.emotion].chip)}
              >
                <span className={cn("font-semibold", item.negated && "line-through decoration-2")}>{item.token}</span>
                <span className="mx-1.5 opacity-60">→</span>
                {EMOTION_STYLE[item.emotion].label}
                <span className="ml-2 text-[11px] opacity-60">
                  {TYPE_LABEL[item.type]} · +{item.weight}
                </span>
                {item.negated && item.original_emotion ? (
                  <span className="mt-0.5 block text-[11px] opacity-75">
                    negated: was {EMOTION_STYLE[item.original_emotion].label.toLowerCase()}
                  </span>
                ) : null}
                {item.boosts.length ? (
                  <span className="mt-0.5 block text-[11px] opacity-75">boosted by {item.boosts.join(", ")}</span>
                ) : null}
              </li>
            ))}
          </ul>
        )}
        {emotion.signals.length ? (
          <p className="text-xs text-neutral-500">Intensity signals: {emotion.signals.join(" · ")}</p>
        ) : null}
      </section>
    </div>
  )
}
