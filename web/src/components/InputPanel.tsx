import { Loader2, ShieldCheck, Sparkles } from "lucide-react"

import { BgAnimateButton } from "@/components/ui/bg-animate-button"
import {
  TextureCard,
  TextureCardContent,
  TextureCardDescription,
  TextureCardFooter,
  TextureCardHeader,
  TextureCardTitle,
  TextureSeparator,
} from "@/components/ui/texture-card"
import { MAX_TEXT_LENGTH } from "@/lib/api"

const EXAMPLES = [
  "namast aap kaise ho 😂",
  "bht gussa aa raha h 😡😡",
  "not happy yaar 😔",
  "omg plz tell me ur coming 😲",
  "tumba bhaya aagtide 😨",
  "love u maa ❤️",
]

interface InputPanelProps {
  text: string
  onTextChange: (value: string) => void
  onAnalyze: () => void
  canAnalyze: boolean
  isLoading: boolean
  inputError: string | null
}

export function InputPanel({ text, onTextChange, onAnalyze, canAnalyze, isLoading, inputError }: InputPanelProps) {
  return (
    <TextureCard className="h-fit">
      <TextureCardHeader className="px-6">
        <p className="pl-2 font-mono text-[11px] tracking-[0.2em] text-cyan-300/80 uppercase">01 · Your text</p>
        <TextureCardTitle className="mt-1">Type a message</TextureCardTitle>
        <TextureCardDescription>
          Romanized Indic or English, with emoji and slang welcome.
        </TextureCardDescription>
      </TextureCardHeader>
      <TextureSeparator />
      <TextureCardContent className="space-y-4">
        <label htmlFor="source-text" className="sr-only">
          Message to analyse
        </label>
        <textarea
          id="source-text"
          rows={6}
          maxLength={MAX_TEXT_LENGTH}
          spellCheck={false}
          value={text}
          onChange={(event) => onTextChange(event.target.value)}
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && canAnalyze) onAnalyze()
          }}
          placeholder="e.g. bht gussa aa raha h 😡😡"
          aria-describedby="input-help"
          className="w-full resize-y rounded-xl border border-white/10 bg-black/30 p-4 text-base text-neutral-100 placeholder:text-neutral-500 focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20 focus:outline-none"
        />
        <div id="input-help" className="flex items-center justify-between text-xs text-neutral-500">
          <span>Ctrl + Enter to analyse</span>
          <span className="tabular-nums">
            {text.length.toLocaleString()} / {MAX_TEXT_LENGTH.toLocaleString()}
          </span>
        </div>
        {inputError ? (
          <p role="alert" className="text-sm text-red-400">
            {inputError}
          </p>
        ) : null}

        <div>
          <p className="mb-2 text-xs font-medium text-neutral-400">Try an example</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => onTextChange(example)}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-neutral-300 transition hover:border-cyan-400/40 hover:bg-cyan-400/10 hover:text-white"
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      </TextureCardContent>
      <TextureSeparator />
      <TextureCardFooter className="flex-col items-stretch gap-3 sm:flex-row sm:items-center">
        <BgAnimateButton
          type="button"
          gradient="ocean"
          animation="spin-slow"
          rounded="full"
          onClick={onAnalyze}
          disabled={!canAnalyze || isLoading}
          className="disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="flex items-center justify-center gap-2 px-2 whitespace-nowrap">
            {isLoading ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
            {isLoading ? "Analysing…" : "Analyse message"}
          </span>
        </BgAnimateButton>
        <button
          type="button"
          onClick={() => onTextChange("")}
          className="rounded-full px-4 py-2 text-sm text-neutral-400 transition hover:text-white"
        >
          Clear
        </button>
        <p className="flex items-center gap-1.5 text-xs text-neutral-500 sm:ml-auto">
          <ShieldCheck className="size-3.5" /> Processed locally, never stored
        </p>
      </TextureCardFooter>
    </TextureCard>
  )
}
