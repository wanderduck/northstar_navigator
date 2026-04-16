# NorthStar Navigator — Video Demo Script & Action Plan

**Target length:** 2:45–2:55 (leave 5s buffer under the 3:00 limit) **Evaluation:** Impact & Vision (40pts), Storytelling (30pts), Technical Depth (30pts)

---

## Action Plan: Recording Setup

### Before Recording

1.  **Pod is running** with latest code (`git pull` + `start.sh`)
2.  **Pre-warm Ollama** — send a test query so the first demo response is fast
3.  **Clear chat history** in Gradio (refresh the page)
4.  **Browser setup:**
    -   Use Chrome/Firefox in a clean window (no bookmarks bar, no other tabs)
    -   Set zoom to 110-125% so the UI fills the frame
    -   Navigate to `https://navigator.wanderduck.dev`
5.  **Screen recording:** OBS Studio or similar, 1080p, capture browser window only
6.  **Audio:** Use a decent microphone. Quiet room. Record narration live over the demo, OR record screen first and add voiceover in editing.
7.  **Have these queries pre-typed** in a text file for quick paste (avoid typos on camera):

```
Query 1 (English):
I'm a single mom with two kids, ages 3 and 7. I just got laid off from my warehouse job where I made $32,000. We're in Ramsey County and I'm worried about paying rent and feeding my kids.

Query 2 (Spanish):
Soy madre soltera con dos hijos. Perdí mi trabajo y necesito ayuda con comida y alquiler. Vivo en el condado de Dakota.

Query 3 (Hmong):
Kuv yog ib tug neeg laus nyob hauv Hennepin County. Kuv xav tau kev pab them nqi cua sov rau lub caij ntuj no.
```

### Recording Tips

-   **Do NOT speed up the video** — judges want to see real response times
-   **Pause briefly** after pasting each query to let the viewer read it
-   **Let the streaming response play out** — the token-by-token generation is visually impressive
-   **Scroll slowly** through long responses so viewers can read
-   If a response takes >20s, that's fine — it shows real inference, not a fake demo

---

## Video Script

### SEGMENT 1: The Problem (0:00 – 0:30)

**[SCREEN: Title card or banner image]** *Show `NorthStar_Navigator_banner.png` for 3-4 seconds*

**NARRATION:**

> "Minnesota has over 300,000 residents who don't speak English at home — Hmong families, Somali families, Spanish-speaking families. When they need help paying rent, feeding their kids, or heating their homes, the answers exist. But the documents explaining those answers are written at a 12th-grade reading level, in English, behind web portals that assume you already know which program to look for."

**[SCREEN: Briefly flash a screenshot of a dense DHS manual page, or the MNbenefits.mn.gov portal — to visually show the complexity]**

> "NorthStar Navigator changes that."

---

### SEGMENT 2: Live Demo — English (0:30 – 1:20)

**[SCREEN: Switch to the live demo at navigator.wanderduck.dev]**

**NARRATION:**

> "Here's how it works. A user describes their situation in plain language."

**[ACTION: Paste Query 1 into the chat and press Enter]**

> "A single mother in Ramsey County, just laid off, worried about rent and food for her two kids."

**[ACTION: Let the streaming response play. The "Analyzing your situation..." and "Finding programs..." status messages will show briefly before the response streams in.]**

> "The system runs a three-stage pipeline. First, Gemma 4 parses her situation into a structured profile — household size, income, county, dependents. Second, a rule-based engine checks Federal Poverty Level thresholds against real program data — 1,938 documents from the DHS manual, county programs, and SAM.gov. Third, Gemma 4 generates a plain-language response at her reading level."

**[ACTION: Scroll through the response as you narrate, pointing out specific programs mentioned]**

> "She gets specific programs — SNAP, MFIP, Emergency Assistance — with eligibility thresholds and where to apply. Not guesses. Verified data."

---

### SEGMENT 3: Multilingual Demo (1:20 – 1:50)

**NARRATION:**

> "But what if she speaks Spanish?"

**[ACTION: Change the Language dropdown to "Spanish", then paste Query 2 and press Enter]**

> "Same situation, described in Spanish. The model was fine-tuned on 10,000 multilingual examples across four languages."

**[ACTION: Let the Spanish response stream in. Pause to show it's actually in Spanish.]**

> "The response comes back in Spanish — not translated from English, but generated natively in Spanish. We also support Hmong and Somali, covering Minnesota's three largest limited-English communities."

**[OPTIONAL ACTION: Quick paste of the Hmong query to show it works — don't wait for full response, just show the first few tokens streaming in Hmong, then move on]**

---

### SEGMENT 4: Safety & Privacy (1:50 – 2:15)

**[ACTION: Point to the sidebar — show the System Status indicator saying "OK — model 'navigator' loaded"]**

**NARRATION:**

> "Every piece of this runs locally through Ollama. The model is on this GPU. The data is on this machine. No API calls, no cloud services, no data leaving the device. When you're asking about income, immigration status, or domestic violence, that's not a nice-to-have — it's essential."

**[ACTION: Scroll to the bottom of a response to show the disclaimer]**

> "The system never says someone qualifies — always 'may be eligible.' Every response includes a disclaimer. Every answer cites its sources. And when it doesn't know, it directs to 211 — Minnesota's human helpline."

**[ACTION: Show the reading level toggle — switch from "standard" to "simple"]**

> "Reading level adapts too. Fifth grade, eighth grade, or full detail. The same answer, three ways."

---

### SEGMENT 5: Technical Close & Impact (2:15 – 2:50)

**[SCREEN: Show the architecture diagram from the notebook, or a quick slide]**

**NARRATION:**

> "Under the hood: Gemma 4 E4B, fine-tuned with QLoRA using Unsloth for rapid prototyping and GGUF export. The model runs as a 5.3 gigabyte quantized GGUF through Ollama on a single consumer GPU. The eligibility engine is deliberately not an LLM — it uses hard-coded rules and hybrid RAG retrieval, because eligibility thresholds are facts, not opinions."

> "We built this for the Hmong grandmother whose heating bill just arrived and who has no one to translate the government website. She types her situation in Hmong. She gets an answer in Hmong. At her reading level. With programs she may be eligible for and phone numbers to call."

**[SCREEN: Return to the live demo, showing the chat history with responses in multiple languages]**

> "That's not a feature. That's dignity. And that's what happens when the right tools are accessible to everyone."

**[SCREEN: End card — project name, demo URL, GitHub link. Hold for 3-5 seconds.]**

```
NorthStar Navigator
navigator.wanderduck.dev
github.com/wanderduck/northstar_navigator

Kaggle Gemma 4 Good Hackathon
```

---

## Timing Summary

Segment

Duration

Content

1. The Problem

0:00–0:30

Title card, problem statement, DHS complexity

2. English Demo

0:30–1:20

Live query, streaming response, architecture narration

3. Multilingual

1:20–1:50

Spanish query, Hmong flash, multilingual fine-tuning

4. Safety & Privacy

1:50–2:15

Ollama local, disclaimer, reading levels

5. Technical + Close

2:15–2:50

Architecture, Unsloth/GGUF, grandmother callback, end card

**Total**

**~2:50**

**10s buffer under 3:00**

---

## Assets to Prepare

Asset

Location

Use

Banner image

`docs/Styling/NorthStar_Navigator_banner.png`

Title card

Icon

`docs/Styling/NorthStar_Navigator_icon.png`

End card

Architecture diagram

From `northstar_navigator.ipynb` Section 1

Technical segment

DHS screenshot

Capture from mn.gov/dhs

Problem segment (show complexity)

Pre-typed queries

Copy from above

Paste during demo

## Editing Notes

-   **Keep it tight.** Cut any dead air. The streaming response IS the content — don't talk over the first few seconds of it.
-   **Background music** (optional): Low, ambient, non-distracting. Fade out during demo segments.
-   **Captions** (recommended): Auto-generate with YouTube, then clean up. Judges may watch without sound.
-   **Thumbnail**: Use `NorthStar_Navigator_banner.png` as the YouTube thumbnail.