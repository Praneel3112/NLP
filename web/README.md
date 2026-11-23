# Translit Lab frontend

React + TypeScript + Tailwind v4 + shadcn/Cult UI, built with Vite.

- `npm run build` writes `dist/`, which `../server.py` serves at http://localhost:8000.
- `npm run dev` starts a hot-reload dev server; keep `python server.py` running for the `/api` proxy.
- Cult UI components live in `src/components/ui/` (added with `npx shadcn@latest add @cult-ui/<name>`).
