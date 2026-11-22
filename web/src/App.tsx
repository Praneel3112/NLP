import { useCallback, useEffect, useState } from "react"

import { Hero } from "@/components/Hero"
import { HowItWorks } from "@/components/HowItWorks"
import { InputPanel } from "@/components/InputPanel"
import { ResultsPanel } from "@/components/ResultsPanel"
import { StatusIsland, type ConnectionState } from "@/components/StatusIsland"
import {
  analyzeText,
  fetchEmotionMetrics,
  fetchStatus,
  MAX_TEXT_LENGTH,
  type AnalysisResult,
  type EmotionMetrics,
} from "@/lib/api"

function useBackendStatus() {
  const [state, setState] = useState<ConnectionState>("checking")
  const [stage, setStage] = useState("")

  useEffect(() => {
    let timer: number | undefined
    let cancelled = false

    const poll = async () => {
      try {
        const status = await fetchStatus()
        if (cancelled) return
        setState(status.status)
        setStage(status.stage)
        if (status.status === "loading") timer = window.setTimeout(poll, 2500)
      } catch {
        if (cancelled) return
        setState("offline")
        timer = window.setTimeout(poll, 4000)
      }
    }
    poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [])

  return { state, stage }
}

export default function App() {
  const { state, stage } = useBackendStatus()
  // A link like /?text=bht%20gussa%20😡 pre-fills the box and runs once the models are ready.
  const [sharedText] = useState(() => new URLSearchParams(window.location.search).get("text") ?? "")
  const [text, setText] = useState(sharedText)
  const [autoRunPending, setAutoRunPending] = useState(Boolean(sharedText.trim()))
  const [inputError, setInputError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [metrics, setMetrics] = useState<EmotionMetrics | null>(null)

  useEffect(() => {
    if (state === "ready" && !metrics) {
      fetchEmotionMetrics().then(setMetrics).catch(() => setMetrics(null))
    }
  }, [state, metrics])

  const handleTextChange = (value: string) => {
    setText(value)
    setInputError(null)
  }

  const analyze = useCallback(async () => {
    if (!text.trim()) {
      setInputError("Type a message or pick an example first.")
      return
    }
    if (text.length > MAX_TEXT_LENGTH) {
      setInputError(`Text must be ${MAX_TEXT_LENGTH.toLocaleString()} characters or fewer.`)
      return
    }
    setIsLoading(true)
    setError(null)
    try {
      setResult(await analyzeText(text))
    } catch (caught) {
      setResult(null)
      setError(caught instanceof Error ? caught.message : "The analysis failed.")
    } finally {
      setIsLoading(false)
    }
  }, [text])

  useEffect(() => {
    if (autoRunPending && state === "ready") {
      setAutoRunPending(false)
      analyze()
    }
  }, [autoRunPending, state, analyze])

  return (
    <div className="min-h-svh bg-background text-foreground">
      <StatusIsland state={state} stage={stage} />
      <Hero />

      <main id="workspace" className="mx-auto grid max-w-6xl scroll-mt-20 gap-6 px-4 pt-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <InputPanel
          text={text}
          onTextChange={handleTextChange}
          onAnalyze={analyze}
          canAnalyze={state === "ready"}
          isLoading={isLoading}
          inputError={inputError}
        />
        <ResultsPanel result={result} isLoading={isLoading} error={error} />
      </main>

      <HowItWorks metrics={metrics} />

      <footer className="border-t border-white/10 px-4 py-8 text-center text-xs text-neutral-500">
        Translit Lab · Romanized Indic error correction and emoji/slang emotion detection · Results are experimental and
        trained on small sample datasets.
      </footer>
    </div>
  )
}
