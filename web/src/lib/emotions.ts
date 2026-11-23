import type { Emotion } from "@/lib/api"

export const EMOTION_ORDER: Emotion[] = ["joy", "sadness", "anger", "fear", "surprise", "love", "neutral"]

// One hue per emotion; used for bars, chips and the result header.
export const EMOTION_STYLE: Record<Emotion, { emoji: string; label: string; text: string; bar: string; chip: string }> = {
  joy: { emoji: "😄", label: "Joy", text: "text-amber-300", bar: "from-amber-300 to-amber-500", chip: "border-amber-400/30 bg-amber-400/10 text-amber-200" },
  sadness: { emoji: "😢", label: "Sadness", text: "text-sky-300", bar: "from-sky-300 to-sky-500", chip: "border-sky-400/30 bg-sky-400/10 text-sky-200" },
  anger: { emoji: "😡", label: "Anger", text: "text-red-400", bar: "from-red-400 to-red-600", chip: "border-red-400/30 bg-red-400/10 text-red-200" },
  fear: { emoji: "😨", label: "Fear", text: "text-violet-300", bar: "from-violet-300 to-violet-500", chip: "border-violet-400/30 bg-violet-400/10 text-violet-200" },
  surprise: { emoji: "😲", label: "Surprise", text: "text-teal-300", bar: "from-teal-300 to-teal-500", chip: "border-teal-400/30 bg-teal-400/10 text-teal-200" },
  love: { emoji: "❤️", label: "Love", text: "text-pink-300", bar: "from-pink-300 to-pink-500", chip: "border-pink-400/30 bg-pink-400/10 text-pink-200" },
  neutral: { emoji: "😐", label: "Neutral", text: "text-neutral-300", bar: "from-neutral-300 to-neutral-500", chip: "border-neutral-400/30 bg-neutral-400/10 text-neutral-200" },
}

export const percent = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`
