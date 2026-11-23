import { useEffect } from "react"

import {
  DynamicContainer,
  DynamicIsland,
  DynamicIslandProvider,
  useDynamicIslandSize,
} from "@/components/ui/dynamic-island"
import type { BackendStatus } from "@/lib/api"
import { cn } from "@/lib/utils"

export type ConnectionState = BackendStatus["status"] | "offline" | "checking"

function shortStage(state: ConnectionState, stage: string): string {
  if (state === "checking") return "Connecting to backend…"
  if (state === "offline") return "Backend offline · run python server.py"
  if (state === "error") return "Model training failed"
  if (state === "ready") return "All models ready"
  if (/emotion/i.test(stage)) return "Training emotion model…"
  if (/xlm/i.test(stage)) return "Loading XLM-RoBERTa · Phase 3…"
  if (/loading saved/i.test(stage)) return "Loading saved GRU…"
  if (/training phase 2/i.test(stage)) return "First run: training Bi-GRU…"
  return "Preparing models…"
}

function IslandContent({ state, stage }: { state: ConnectionState; stage: string }) {
  const { setSize } = useDynamicIslandSize()
  const label = shortStage(state, stage)

  useEffect(() => {
    setSize(state === "ready" ? "compact" : "compactLong")
  }, [state, setSize])

  const dot =
    state === "ready"
      ? "bg-emerald-400"
      : state === "error" || state === "offline"
        ? "bg-red-400"
        : "bg-amber-300 animate-pulse"

  return (
    <DynamicContainer key={label} className="flex h-full w-full items-center justify-center gap-2.5 px-4">
      <span className={cn("size-2 shrink-0 rounded-full", dot)} aria-hidden />
      <span className="truncate text-xs font-medium text-white sm:text-sm">{label}</span>
    </DynamicContainer>
  )
}

export function StatusIsland({ state, stage }: { state: ConnectionState; stage: string }) {
  return (
    <div className="pointer-events-none fixed inset-x-0 top-3 z-50 flex justify-center">
      <div className="pointer-events-auto" role="status" aria-live="polite">
        <DynamicIslandProvider initialSize="compactLong">
          <DynamicIsland id="backend-status">
            <IslandContent state={state} stage={stage} />
          </DynamicIsland>
        </DynamicIslandProvider>
      </div>
    </div>
  )
}
