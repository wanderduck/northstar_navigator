# Google Cloud Translate API — Research Findings

Research compiled 2026-04-13 from three parallel doc investigations:
Setup, Translate Text (v3), and Batch Translation.

## Decision: Single-Request v3 API (NOT Batch)

For our volume (~3,000 Q+A pairs x 3 languages = ~2.7M chars), single-request
`translate_text` is the clear winner over `batch_translate_text`:

| Factor | Single-Request | Batch |
|---|---|---|
| GCS required | No (inline text) | Yes (two buckets) |
| Async/sync | Synchronous | Long-running async op |
| Setup complexity | Low | High |
| Cost | $20/M chars | $20/M chars (same) |
| Our estimated cost | ~$0.054 (possibly free) | Same |
| Time for 3K pairs | ~30 seconds | Minutes (job overhead) |

## Package and Import

```
pip install google-cloud-translate
```

```python
from google.cloud import translate_v3
from google.oauth2 import service_account
```

Client class: `translate_v3.TranslationServiceClient()`

## v2 (Basic) vs v3 (Advanced)

| | v2 Basic | v3 Advanced |
|---|---|---|
| Import | `translate_v2` | `translate_v3` |
| Auth | ADC or API key | ADC only (no API key) |
| `parent` param | Not needed | Required: `projects/{id}/locations/global` |
| Batch in single request | No (one string) | Yes (up to 1024 strings) |
| Max request size | 100KB | 30,000 codepoints |
| Glossaries | No | Yes |

We use v3 for the multi-string `contents` parameter.

## Authentication on Modal

Modal secrets are env vars, not files. Cloud Translate client needs credentials.

Cleanest approach: pass service account JSON as an env var, parse it at runtime:

```python
import json, os
from google.cloud import translate_v3
from google.oauth2 import service_account

creds_json = os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
client = translate_v3.TranslationServiceClient(credentials=creds)
project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
parent = f"projects/{project_id}/locations/global"
```

Modal secret setup:
```bash
modal secret create gcloud-translate \
  GOOGLE_CLOUD_PROJECT=your-gcp-project-id \
  GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat path/to/service-account.json)"
```

Service account needs the **Cloud Translation API User** role.

## translate_text API

```python
response = client.translate_text(
    contents=["Hello", "Goodbye"],    # list of strings, max 1024
    target_language_code="es",         # single target per call
    source_language_code="en",         # optional (auto-detect if omitted)
    parent=parent,                     # projects/{id}/locations/global
    mime_type="text/plain",            # or text/html
)

# Response: 1:1 correspondence with input
for t in response.translations:
    print(t.translated_text)
```

## Limits

| Limit | Value |
|---|---|
| Max strings per `contents` | 1024 |
| Max codepoints per request | 30,000 (recommended) |
| Characters per minute | 6,000,000 |
| Requests per minute | 6,000 |
| Daily quota | Unlimited (default) |
| Pricing | $20/M chars (first 500K/month free) |

## Language Support

| Language | BCP-47 Code | Supported | Quality Notes |
|---|---|---|---|
| Spanish | `es` | Yes | High-quality NMT |
| Hmong | `hmn` | Yes | Lower quality (less training data), White Hmong |
| Somali | `so` | Yes | Moderate quality |

## Known Limitations vs Gemini Translation

Cloud Translate is NMT (neural machine translation), not instruction-following LLM.
It cannot be told to "keep program names in English" or "be warm and empathetic."

In practice:
- Acronyms (SNAP, MFIP, UI, WIC) usually pass through unchanged
- URLs and email addresses usually pass through unchanged
- County names may sometimes get translated in low-resource languages
- Tone/warmth is not controllable

Mitigation options:
1. **Glossary** (v3 feature): define terms that should not be translated
2. **Post-processing**: regex to restore known program names
3. **Accept the tradeoff**: speed and cost vs. fine-grained control

## Batch Translation (Not Used)

For reference, batch translation requires:
- Two GCS buckets (input/output)
- `text/plain` or `text/html` input only (no JSONL)
- Up to 100 files, 10 target languages per job
- Returns a long-running operation
- Same $20/M chars pricing

Only worthwhile for tens of millions of characters or fire-and-forget overnight jobs.
