# web

Static front end for SteerEats (Vite + D3 v7). Not scaffolded yet; see `docs/PLAN.md` §2 and §3.

Planned layout:

```
web/
  index.html
  src/
    tokens.css        design tokens (OKLCH colors, type, spacing)
    steer.js          client-side controls: score normalization, distance/popularity terms, MMR
    views/            one module per view (V1–V7)
  public/data/
    nyc_nta_2020.topojson
    cases/demo_*.json one anonymized case file per demo user (others git-ignored)
```
