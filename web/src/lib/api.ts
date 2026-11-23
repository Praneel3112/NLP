// Types and fetch helpers for the Python backend (server.py).

export type Emotion = "joy" | "sadness" | "anger" | "fear" | "surprise" | "love" | "neutral"

export interface BackendStatus {
  status: "loading" | "ready" | "error"
  stage: string
  error: string | null
  training_examples: number
  phase2_loss: number | null
  phase3_model: string | null
  emotion_accuracy: number | null
  emotion_macro_f1: number | null
}

export interface SlangReplacement {
  original: string
  replacement: string
  kind: "chat" | "indic"
  position: number
}

export interface FoundEmoji {
  emoji: string
  name: string
  position: number
}

export interface Phase0Result {
  input: string
  normalized: string
  replacements: SlangReplacement[]
  emoji: FoundEmoji[]
  changed: boolean
}

export interface EmotionEvidence {
  token: string
  type: "emoji" | "slang" | "word"
  emotion: Emotion
  original_emotion?: Emotion
  weight: number
  negated: boolean
  boosts: string[]
}

export interface EmotionResult {
  label: Emotion
  emoji: string
  confidence: number
  method: "hybrid" | "model_only"
  rule_weight: number
  scores: Record<Emotion, number>
  rule_scores: Record<Emotion, number>
  model_scores: Record<Emotion, number>
  intensity: "none" | "low" | "medium" | "high"
  intensity_score: number
  signals: string[]
  evidence: EmotionEvidence[]
}

export interface Phase1Step {
  original: string
  corrected: string
  distance: number
  candidates: string[]
  decision: string
}

export interface Phase2Step {
  original: string
  model_output: string
  corrected: string
  decision: string
}

export interface RankedCandidate {
  word: string
  score: number
  pieces: number
}

export interface Phase3Ranking {
  position: number
  token: string
  chosen: string
  candidates: RankedCandidate[]
}

export interface Phase3Trace {
  model_output: string | null
  decision: string
  output: string
  model?: string | null
  rankings?: Phase3Ranking[]
}

export interface PhaseTimings {
  phase1: number
  phase2: number
  phase3: number
}

export interface AnalysisResult {
  input: string
  phase0: Phase0Result
  emotion: EmotionResult | null
  phase1: string
  phase1_guided: string
  phase1_steps: Phase1Step[]
  phase2: string
  phase2_steps: Phase2Step[]
  phase3: string
  phase3_trace: Phase3Trace
  timings_ms?: PhaseTimings
}

export interface ScorePair {
  accuracy: number
  macro_f1: number
}

export interface EmotionMetrics {
  dataset_size: number
  train_size: number
  test_size: number
  class_counts: Record<Emotion, number>
  accuracy: number
  macro_f1: number
  cv_accuracy_mean: number
  cv_accuracy_std: number
  labels: Emotion[]
  confusion_matrix: number[][]
  comparison: { rules_only: ScorePair; model_only: ScorePair; hybrid: ScorePair }
}

export const MAX_TEXT_LENGTH = 2000

export async function fetchStatus(): Promise<BackendStatus> {
  const response = await fetch("/api/status", { cache: "no-store" })
  if (!response.ok) throw new Error("The Python backend did not return a valid status.")
  return response.json()
}

export async function fetchEmotionMetrics(): Promise<EmotionMetrics> {
  const response = await fetch("/api/emotion/metrics", { cache: "no-store" })
  if (!response.ok) throw new Error("Emotion metrics are not available yet.")
  return response.json()
}

export async function analyzeText(text: string): Promise<AnalysisResult> {
  let response: Response
  try {
    response = await fetch("/api/correct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    })
  } catch {
    throw new Error("Could not reach the Python backend. Start it with: python server.py")
  }
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(
      response.status === 404
        ? "The Python backend is not running. Start it with: python server.py"
        : payload.error || "The analysis request failed."
    )
  }
  return payload as AnalysisResult
}
