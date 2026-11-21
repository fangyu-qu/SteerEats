# User study protocol

Plan section 4 ("Usefulness"). Everything a session needs is built; results come only from real sessions.

## Design

- **Participants:** 3–5 classmates or friends; no personal data is recorded (IDs P1–P5 only). Course projects
  with peers normally need no IRB review, but check the course's NYU participant rules before the first
  session and get verbal consent (script below).
- **Conditions (within subjects):** the full interface vs a list-only baseline (console and top 10 only,
  like TasteWeights' input and output columns; no rank flow, map, ledger, previews or analyst views).
- **Counterbalancing:** two task sets (A, B) on different demo users. Odd IDs start with the full interface,
  even IDs with the list; P1–P2 start with set A, P3–P4 with set B, P5 as P1.
- **Tasks per set** (answers computed by `src/28_user_study.py --tasks`, scored automatically):
  1. *Why:* a named place left the top 10 under preset targets; which control pushed it out?
  2. *Reach:* get a restaurant from a named borough into the top 10 while keeping at least 8 of the
     model's 10 (solvable with the console alone, so in both conditions).
  3. *Model:* which model leans more on a named kind of evidence for this user?
- **Measures:** success, time, number of interactions per task; after each part, three 1–7 ratings
  (understanding, control, trust); think-aloud notes by the facilitator.

## Running a session (about 25 minutes)

1. `cd web && npm run dev` (or use the deployed site) and open `http://localhost:5173/#study=P1` with the
   participant's ID. Full screen, desktop browser.
2. Consent (read aloud): "This is a class project. You will use a restaurant recommender for about 20 minutes
   and think aloud. We record task times and answers, not your name. You can stop at any time."
3. Two-minute tour of the console only (same for both conditions); do not explain the explanation views.
4. The participant works through the bar at the bottom of the screen. Do not help with answers; note
   confusions and quotes.
5. At the end click **Download results** and move the file to `data/user_study/` (git-ignored).

## Analysis

```bash
python src/28_user_study.py      # outputs/user_study/USER_STUDY.md and summary.csv
```

Descriptive only (success rate, median time and interactions per task and condition, median ratings); with at
most five people no significance test. Report the think-aloud themes next to the numbers.
