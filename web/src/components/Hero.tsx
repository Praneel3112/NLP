import { ArrowDown } from "lucide-react"

import {
  HeroDitheringActions,
  HeroDitheringBadges,
  HeroDitheringContainer,
  HeroDitheringContent,
  HeroDitheringDescription,
  HeroDitheringHeading,
  HeroDitheringMobileVisual,
  HeroDitheringRoot,
  HeroDitheringVisual,
} from "@/components/ui/hero-dithering"

const LANGUAGES = [{ name: "Hindi" }, { name: "Kannada" }, { name: "Tamil" }, { name: "Bengali" }, { name: "English" }]

export function Hero() {
  return (
    <HeroDitheringRoot
      className="pt-24 sm:pt-28"
      srTitle="Translit Lab"
      title={<span>Translit Lab</span>}
      subtitle={<span className="text-cyan-300">Fix the spelling. Read the mood.</span>}
      description={
        <>
          Corrects Romanized Hindi, Kannada, Tamil and Bengali, then detects the{" "}
          <strong className="text-foreground">emotion</strong> behind the message from emoji, chat slang
          and code-mixed words.
        </>
      }
      ctaProps={{
        label: (
          <span className="flex items-center gap-2">
            Try it <ArrowDown className="size-4" />
          </span>
        ),
        href: "#workspace",
        target: undefined,
        rel: undefined,
      }}
      techStack={LANGUAGES}
      desktopShaderProps={{ colorFront: "#22d3ee", shape: "swirl", speed: 0.6 }}
      mobileShaderProps={{ colorFront: "#22d3ee", speed: 0.5 }}
    >
      <HeroDitheringContainer className="mx-auto max-w-6xl px-4">
        <HeroDitheringContent>
          <p className="text-center font-mono text-xs tracking-[0.2em] text-cyan-300/80 uppercase lg:text-left">
            NLP project · Romanized Indic text
          </p>
          <HeroDitheringHeading headingClassName="font-semibold" />
          <HeroDitheringDescription />
          <HeroDitheringActions />
          <div className="flex justify-center lg:justify-start">
            <HeroDitheringBadges />
          </div>
        </HeroDitheringContent>
        <HeroDitheringVisual />
      </HeroDitheringContainer>
      <HeroDitheringMobileVisual />
    </HeroDitheringRoot>
  )
}
