import { AlertTriangle, Loader2, MessageSquareText } from "lucide-react"

import { CorrectionView } from "@/components/CorrectionView"
import { EmotionView } from "@/components/EmotionView"
import { SlangView } from "@/components/SlangView"
import { DirectionAwareTabs } from "@/components/ui/direction-aware-tabs"
import {
  TextureCard,
  TextureCardContent,
  TextureCardHeader,
  TextureCardTitle,
  TextureSeparator,
} from "@/components/ui/texture-card"
import type { AnalysisResult } from "@/lib/api"

interface ResultsPanelProps {
  result: AnalysisResult | null
  isLoading: boolean
  error: string | null
  initialTab?: number
}

export function ResultsPanel({ result, isLoading, error, initialTab = 0 }: ResultsPanelProps) {
  let body
  if (isLoading) {
    body = (
      <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-neutral-400" aria-busy="true">
        <Loader2 className="size-6 animate-spin text-cyan-300" />
        Running Phase 0 → 3 and emotion detection…
      </div>
    )
  } else if (error) {
    body = (
      <div role="alert" className="flex min-h-64 items-start gap-3 rounded-2xl border border-red-400/30 bg-red-400/5 p-5">
        <AlertTriangle className="size-5 shrink-0 text-red-400" />
        <div>
          <p className="font-medium text-red-200">Analysis unavailable</p>
          <p className="mt-1 text-sm text-red-200/70">{error}</p>
        </div>
      </div>
    )
  } else if (!result) {
    body = (
      <div className="flex min-h-64 flex-col items-center justify-center gap-3 text-center">
        <MessageSquareText className="size-8 text-neutral-600" />
        <p className="font-medium text-neutral-300">Your results will appear here</p>
        <p className="max-w-xs text-sm text-neutral-500">
          Type a message, then press Analyse.
        </p>
      </div>
    )
  } else {
    body = (
      <DirectionAwareTabs
        defaultTab={initialTab}
        className="mb-4 bg-neutral-800"
        tabs={[
          { id: 0, label: "Emotion", content: <EmotionView emotion={result.emotion} input={result.input} /> },
          { id: 1, label: "Slang · Phase 0", content: <SlangView phase0={result.phase0} /> },
          { id: 2, label: "Correction · 1–3", content: <CorrectionView result={result} /> },
        ]}
      />
    )
  }

  return (
    <TextureCard className="h-fit min-w-0" aria-live="polite">
      <TextureCardHeader className="px-6">
        <p className="pl-2 font-mono text-[11px] tracking-[0.2em] text-cyan-300/80 uppercase">02 · Results</p>
        <TextureCardTitle className="mt-1">What the pipeline found</TextureCardTitle>
      </TextureCardHeader>
      <TextureSeparator />
      <TextureCardContent>{body}</TextureCardContent>
    </TextureCard>
  )
}
