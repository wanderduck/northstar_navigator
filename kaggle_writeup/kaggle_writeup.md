# NorthStar Navigator: Plain-Language Government Benefits for Every Minnesotan

*A fine-tuned Gemma 4 E4B system that speaks four languages, respects three reading levels, and never claims you qualify for anything.*

---

Minnesota's state motto is "L'Etoile du Nord" — The Star of the North. For the 300,000 residents with limited English proficiency, that star might as well be behind clouds. Government benefits exist. People need them. The documents explaining those benefits read at a 12th-grade level, dense with acronyms like BBCE, MFIP, and CCAP, buried behind portals that assume you already know which program to look for.

Consider a Hmong grandmother in Ramsey County, raising two grandchildren on $18,000 a year. She may qualify for SNAP, Medical Assistance, Energy Assistance, WIC, and Emergency Assistance — five programs, three application portals, a dozen documents to gather. The information is scattered across 614 DHS manual sections, 267 county program listings, and 515 federal assistance entries. What does not exist is a single place where she can describe what she needs, in her own words, and get a plain-language answer.

NorthStar Navigator is that place. A Gradio chat interface backed by fine-tuned Gemma 4 E4B running through Ollama. A user describes their situation — in English, Spanish, Hmong, or Somali — and the system guides them toward programs they may be eligible for, at a reading level they choose.

## Architecture: Three Stages, One Guardrail Philosophy

The pipeline separates what language models do well from what they must never do alone.

**Stage 1 — Intake.** Gemma 4 parses freeform text into a structured profile: household size, income, county, dependents, concerns. It infers county from city names, detects language, and identifies missing information. If critical fields are absent, it asks a single warm follow-up question rather than presenting a form. The model excels here — turning messy human language into clean data.

**Stage 2 — Eligibility.** Deliberately not an LLM stage. A rule-based engine applies hard-coded Federal Poverty Level thresholds (200% for SNAP, 138% for Medicaid, 185% for WIC) and layers in county-specific programs from a curated database of 267 programs across five counties and three CAP agencies. ChromaDB holds 1,938 documents retrieved via hybrid search — 60% semantic vector similarity, 40% BM25 keyword matching. The LLM generates language around verified facts. It does not generate facts. The rules are the rules.

**Stage 3 — Response.** Gemma 4 produces plain-language guidance calibrated to the user's reading level, with mandatory disclaimers. Every response is Flesch-Kincaid scored against its target. The system never says "qualifies" — always "may be eligible." This is not a prompt instruction the model can drift from; it is behavior fine-tuned into the GGUF weights through QLoRA on consistently hedged training examples. A "Sources and Reasoning" section accompanies every answer: matched programs, FPL percentages, statutory citations. Users can independently verify each claim. When errors occur, the system directs to Minnesota's 211 helpline rather than generating a potentially incorrect answer. Seventy-five tests validate disclaimer presence, eligibility logic, and readability scoring.

## Breaking Four Walls

Government benefits catch people when they fall. Minnesota's three largest LEP communities — Hmong (66,000 speakers), Somali (57,000), and Spanish (~200,000) — are disproportionately represented among families who need these programs. They are not newcomers waiting for a handshake. They are neighbors who deserve answers in their own language.

**The Language Wall.** Navigator supports four languages — not as a translation layer bolted onto an English system, but through multilingual fine-tuning. Gemini generates grounded English Q&A pairs from actual DHS manual content, Cloud Translate NMT renders them to Spanish, Hmong, and Somali, and QLoRA adapts Gemma 4 across all four simultaneously. Approximately 10,000 multilingual examples, trained on 4x A100-80GB with DDP. The result is a model that does not merely translate — it converses.

**The Literacy Wall.** The DHS Combined Manual reads at a 12th-grade level or above. Nationally, 54% of U.S. adults read below 6th grade. The people who most need these programs are the least equipped to parse the documents describing them. Navigator offers three reading levels — 5th, 8th, and 12th grade — with Flesch-Kincaid scoring on every response. The same eligibility information, explained three different ways. For millions of Americans, that toggle is the difference between understanding and exclusion.

**The Knowledge Wall.** Traditional systems require knowing program acronyms before you search. Navigator inverts this entirely. "I lost my job and can't feed my kids" maps to SNAP, MFIP, and Emergency Assistance. Users describe problems. The system finds programs. That inversion matters because the people furthest from the system are the ones least likely to know its vocabulary.

**The Trust Wall.** Navigator asks about income, disability status, immigration history, domestic violence. For immigrant communities and survivors, disclosing this to a digital system carries real risk. This is where architecture matters most. Ollama runs every inference call on localhost:11434. No API key, no outbound network call, no third-party logging. The conversation never leaves the machine. A Somali mother describing her immigration status is describing it to her own hardware. Privacy here is not a feature toggle — it is a technical impossibility of data exfiltration. The fine-tuned GGUF (q4_k_m, 5.3 GB) loads natively via `ollama create navigator -f Modelfile`, with the system prompt baked into the Modelfile itself. Three call modes serve the pipeline: `chat_json()` for structured intake extraction, `chat_stream()` for real-time Gradio streaming, `chat()` for standard generation. Consumer hardware handles inference: RTX 4090 on RunPod for the live demo, RTX 2080 Ti for local development. No A100 required at serving time.

## Fine-Tuning with Unsloth and Raw PEFT

Gemma 4's `ClippableLinear` layers broke standard PEFT — they extend `nn.Module`, not `nn.Linear`, so the library cannot find LoRA targets. Unsloth's `FastLanguageModel` handles this transparently. We used Unsloth for rapid prototyping on Kaggle's free P100 GPUs — 4-bit NF4 quantized loading and optimized gradient checkpointing fit the full training loop within 16 GB VRAM. LoRA rank 16, alpha 32. Unsloth also provided native GGUF export with q4_k_m quantization in a single `save_pretrained_gguf` call — no separate llama.cpp conversion pipeline needed.

For production scale, we used raw PEFT with manual `ClippableLinear` patching, since Unsloth targets single-GPU workflows. Four A100-80GB GPUs, DDP with `ddp_find_unused_parameters=True` (Gemma 4 E4B's vision and audio towers do not participate in text-only loss), three epochs, approximately 45 minutes. The Unsloth prototyping loop — free GPUs to working LoRA in under an hour — made the production run possible by validating hyperparameters cheaply first.

## The Name

Minnesota is the North Star State. NorthStar Navigator guides people to help the way a pole star guides travelers — not by moving them, but by giving them a fixed point to orient from. Our Hmong grandmother's heating bill arrived and the number terrifies her. She types her situation in Hmong. She receives an answer in Hmong, at her reading level, with programs she may be eligible for and phone numbers to call next.

That is not a feature. That is dignity. And it is what happens when the right tools are accessible to everyone.

---

**Resources:** [Live Demo](https://navigator.wanderduck.dev) | [GitHub](https://github.com/wanderduck/northstar_navigator) | [GGUF Model](https://huggingface.co/wanderduck/northstar-navigator-gguf) | [Kaggle Notebook](https://www.kaggle.com/code/wanderduck/northstar-navigator)