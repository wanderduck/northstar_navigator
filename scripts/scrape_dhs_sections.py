"""Targeted DHS scraper: scrape specific chapter sections immediately on discovery.

Unlike scrape_dhs_manual.py (which discovers ALL sections first, then scrapes),
this script visits each chapter page and scrapes its subsections before moving
on. If the bot deterrent triggers mid-run, you keep everything scraped so far.

The bot deterrent typically kicks in after ~18 page loads. To work around this:
- Uses randomized delays (base +/- 50%) to look more human
- Pauses longer between chapters
- In --headed mode, pauses for you to solve CAPTCHA and refreshes cookies
- Skips already-downloaded files, so just rerun to pick up where you left off

Usage:
    # Scrape chapters 16-18, 22-24, 26-28
    PYTHONPATH=src uv run python scripts/scrape_dhs_sections.py --headed 15-18 22-24 26-28

    # Slower delay to stretch the ~18-request window
    PYTHONPATH=src uv run python scripts/scrape_dhs_sections.py --headed --delay 6 15-18
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page, Browser, BrowserContext

from navigator.config import RAW_DIR

OUTPUT_DIR = RAW_DIR / "dhs_combined_manual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.dhs.state.mn.us"
SECTION_LINK_RE = re.compile(r"dDocName=((?:lp_)?cm_[\w]+)", re.IGNORECASE)
SECTION_NUM_RE = re.compile(r"^\s*(\d[\d.]*)\s")

NOISE_SELECTORS = [
    "nav", "header", "footer", "script", "style", "noscript",
    "[class*='nav']", "[class*='menu']", "[class*='breadcrumb']",
    "[class*='sidebar']", "[class*='header']", "[class*='footer']",
    "[id*='nav']", "[id*='menu']", "[id*='breadcrumb']",
    "[id*='sidebar']", "[id*='header']", "[id*='footer']",
]

CONTENT_SELECTORS = [
    "#idcMainContent", "#mainContent", "#contentArea",
    "#region1", "#regionMain", "#bodyContent",
    "[class*='idcMainContent']", "[class*='mainContent']",
    "main", "article", "#content", ".content",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jitter(base_delay: float) -> float:
    """Return a randomized delay: base * (0.5 to 1.5)."""
    return base_delay * (0.5 + random.random())


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def is_captcha(text: str) -> bool:
    """Check if page content is a bot challenge."""
    lower = text.lower()
    return (
        "solve this captcha" in lower
        or "your activity and behavior" in lower
        or "verifying your browser" in lower
    )


def _save_cookies(context: BrowserContext, cookie_file: Path) -> None:
    """Save current browser cookies to disk for future runs."""
    cookies = context.cookies()
    cookie_file.write_text(json.dumps(cookies, indent=2))
    log.info("Refreshed cookies (%d cookies saved)", len(cookies))


def _page_text_preview(page: Page, chars: int = 300) -> str:
    try:
        return page.query_selector("body").inner_text()[:chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def extract_text(page: Page) -> str:
    """Extract clean text from the current page."""
    for selector in NOISE_SELECTORS:
        try:
            page.evaluate(f"document.querySelectorAll('{selector}').forEach(el => el.remove());")
        except Exception:
            pass

    for selector in CONTENT_SELECTORS:
        try:
            el = page.query_selector(selector)
            if el:
                text = el.inner_text()
                if len(text.strip()) > 100:
                    return _clean_text(text)
        except Exception:
            continue

    try:
        return _clean_text(page.query_selector("body").inner_text())
    except Exception:
        return ""


def extract_subsection_links(page: Page) -> list[dict]:
    """Extract CM subsection links from the current page."""
    links = page.evaluate("""
        () => {
            const results = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.getAttribute('href') || '';
                const text = a.innerText.trim();
                if (href && text) results.push({href, text});
            });
            return results;
        }
    """)

    seen = set()
    sections = []
    for link in links:
        href = link.get("href", "")
        text = link.get("text", "")

        m = SECTION_LINK_RE.search(href)
        if not m:
            continue

        doc_name = m.group(1)
        if doc_name.lower().startswith("lp_cm_"):
            continue
        if doc_name.lower() in seen:
            continue
        seen.add(doc_name.lower())

        num_match = SECTION_NUM_RE.match(text)
        if num_match:
            label = num_match.group(1).replace(".", "_")
        else:
            label = doc_name.lower().replace("cm_", "")

        full_url = (
            f"{BASE_URL}/main/idcplg?IdcService=GET_DYNAMIC_CONVERSION"
            f"&RevisionSelectionMethod=LatestReleased&dDocName={doc_name}"
        )
        sections.append({
            "doc_name": doc_name,
            "label": label,
            "title": text[:120],
            "url": full_url,
        })

    return sections


# ---------------------------------------------------------------------------
# CAPTCHA handling
# ---------------------------------------------------------------------------

def handle_captcha(page: Page, context: BrowserContext, cookie_file: Path,
                   url: str, headed: bool) -> bool:
    """Handle a CAPTCHA. Returns True if resolved, False if we should stop."""
    if not headed:
        log.error("Bot deterrent hit (headless mode) — cannot solve CAPTCHA")
        return False

    log.warning(">>> BOT CHALLENGE DETECTED <<<")
    log.info("Solve the CAPTCHA in the browser window, then press Enter here.")
    input("Press Enter after solving CAPTCHA> ")

    # Save refreshed cookies so future runs start clean
    _save_cookies(context, cookie_file)

    # Retry the page
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)
        if is_captcha(_page_text_preview(page)):
            log.error("Still blocked after CAPTCHA solve. Try again or wait.")
            return False
    except Exception as e:
        log.error("Navigation failed after CAPTCHA: %s", e)
        return False

    log.info("CAPTCHA resolved — continuing")
    return True


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_page(page: Page, context: BrowserContext, section: dict,
                output_dir: Path, delay: float, cookie_file: Path,
                headed: bool) -> str:
    """Navigate to a section and save its text.

    Returns: "scraped", "skipped", "captcha", or "failed"
    """
    safe_label = re.sub(r"[^\w.-]", "_", section["label"])
    out_path = output_dir / f"section_{safe_label}.txt"

    if out_path.exists() and out_path.stat().st_size > 500:
        log.info("  SKIP (exists, %d bytes) %s", out_path.stat().st_size, safe_label)
        return "skipped"

    # Delete stub files from previous failed runs
    if out_path.exists():
        out_path.unlink()

    log.info("  FETCH %s — %s", safe_label, section["title"][:60])

    try:
        page.goto(section["url"], wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
    except Exception as e:
        log.error("  Navigation failed: %s", e)
        return "failed"

    text = extract_text(page)

    if is_captcha(text):
        resolved = handle_captcha(page, context, cookie_file, section["url"], headed)
        if not resolved:
            return "captcha"
        # Re-extract after CAPTCHA resolution
        text = extract_text(page)
        if is_captcha(text):
            return "captcha"

    if len(text) < 50:
        log.warning("  Very short content (%d chars), skipping save", len(text))
        return "failed"

    header = (
        f"SOURCE: {section['url']}\n"
        f"SECTION: {section['label']}\n"
        f"TITLE: {section['title']}\n"
        f"DOC_NAME: {section['doc_name']}\n"
        f"{'=' * 72}\n\n"
    )
    out_path.write_text(header + text, encoding="utf-8")
    log.info("  Saved %d chars -> %s", len(text), out_path.name)

    time.sleep(_jitter(delay))
    return "scraped"


def parse_section_ranges(args: list[str]) -> list[int]:
    """Parse '14-18', '22', '26-28' into sorted list of ints."""
    chapters = set()
    for arg in args:
        if "-" in arg:
            parts = arg.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            chapters.update(range(start, end + 1))
        else:
            chapters.add(int(arg))
    return sorted(chapters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Targeted DHS scraper — scrape specific chapters, "
                    "fetch subsections immediately on discovery.",
    )
    parser.add_argument(
        "chapters", nargs="*",
        help="Chapter numbers or ranges (e.g., 16-18 22-24 26-28). "
             "If omitted, prompts interactively.",
    )
    parser.add_argument(
        "--delay", type=float, default=4.0,
        help="Base seconds between requests, randomized +/-50%% (default: 4.0)",
    )
    parser.add_argument("--headed", action="store_true", help="Show browser window for CAPTCHA solving")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    # Interactive prompt if no chapters specified
    if not args.chapters:
        print("Enter chapter numbers/ranges to scrape (e.g., 16-18 22-24 26-28):")
        user_input = input("> ").strip()
        if not user_input:
            print("No chapters specified. Exiting.")
            sys.exit(0)
        args.chapters = user_input.split()

    chapters = parse_section_ranges(args.chapters)
    log.info("Target chapters: %s", chapters)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cookie_file = OUTPUT_DIR / "_cookies.json"

    total_scraped = 0
    total_skipped = 0
    total_failed = 0
    pages_loaded = 0  # Track for bot deterrent budget
    stopped_at_chapter = None

    with sync_playwright() as pw:
        browser: Browser = pw.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
        )

        if cookie_file.exists():
            cookies = json.loads(cookie_file.read_text())
            context.add_cookies(cookies)
            log.info("Loaded %d cookies from %s", len(cookies), cookie_file.name)

        page = context.new_page()
        abort = False

        for ch_idx, ch_num in enumerate(chapters):
            if abort:
                break

            doc_name = f"lp_cm_{ch_num:04d}"
            ch_url = (
                f"{BASE_URL}/main/idcplg?IdcService=GET_DYNAMIC_CONVERSION"
                f"&RevisionSelectionMethod=LatestReleased&dDocName={doc_name}"
            )

            log.info("")
            log.info("========== Chapter %d (%s) [%d/%d] ==========",
                     ch_num, doc_name, ch_idx + 1, len(chapters))

            try:
                page.goto(ch_url, wait_until="domcontentloaded", timeout=30000)
                page.wait_for_timeout(2000)
                pages_loaded += 1
            except Exception as e:
                log.error("Cannot load chapter %d: %s", ch_num, e)
                total_failed += 1
                continue

            # Check for bot challenge on the chapter page
            preview = _page_text_preview(page)
            if is_captcha(preview):
                resolved = handle_captcha(page, context, cookie_file, ch_url, args.headed)
                if not resolved:
                    stopped_at_chapter = ch_num
                    abort = True
                    break
                pages_loaded = 0  # Reset counter after CAPTCHA solve

            # Extract subsection links from this chapter page
            subsections = extract_subsection_links(page)

            if not subsections:
                # Chapter page itself may be the content
                log.info("No subsections found — scraping chapter page directly")
                text = extract_text(page)
                if not is_captcha(text) and len(text) > 100:
                    out_path = out_dir / f"section_{ch_num}.txt"
                    header = (
                        f"SOURCE: {ch_url}\n"
                        f"SECTION: {ch_num}\n"
                        f"TITLE: Chapter {ch_num}\n"
                        f"DOC_NAME: {doc_name}\n"
                        f"{'=' * 72}\n\n"
                    )
                    out_path.write_text(header + text, encoding="utf-8")
                    log.info("  Saved chapter page: %d chars", len(text))
                    total_scraped += 1
                continue

            # Count how many we actually need to fetch
            needed = 0
            for sub in subsections:
                safe_label = re.sub(r"[^\w.-]", "_", sub["label"])
                p = out_dir / f"section_{safe_label}.txt"
                if not (p.exists() and p.stat().st_size > 500):
                    needed += 1

            log.info("Found %d subsections (%d already done, %d to fetch)",
                     len(subsections), len(subsections) - needed, needed)

            # Scrape each subsection immediately
            for i, sub in enumerate(subsections, 1):
                log.info("[ch.%d: %d/%d | total pages: %d]", ch_num, i, len(subsections), pages_loaded)

                result = scrape_page(page, context, sub, out_dir, args.delay,
                                     cookie_file, args.headed)

                if result == "scraped":
                    total_scraped += 1
                    pages_loaded += 1
                elif result == "skipped":
                    total_skipped += 1
                elif result == "captcha":
                    # handle_captcha already tried to resolve; if we're here it
                    # either resolved (and we should retry) or failed
                    # Try one more time after the CAPTCHA was solved
                    retry = scrape_page(page, context, sub, out_dir, args.delay,
                                        cookie_file, args.headed)
                    if retry == "scraped":
                        total_scraped += 1
                        pages_loaded = 1  # Reset after CAPTCHA
                    elif retry == "captcha":
                        log.error("Double CAPTCHA — stopping")
                        stopped_at_chapter = ch_num
                        abort = True
                        break
                    else:
                        total_failed += 1
                else:
                    total_failed += 1

            # Longer pause between chapters to stay under the radar
            if not abort and ch_idx < len(chapters) - 1:
                between = _jitter(args.delay * 2)
                log.info("Pausing %.1fs between chapters...", between)
                time.sleep(between)

        browser.close()

    # Summary
    log.info("")
    log.info("=" * 50)
    log.info("DONE. Scraped: %d | Skipped: %d | Failed: %d", total_scraped, total_skipped, total_failed)
    log.info("Total pages loaded this run: %d", pages_loaded)

    if stopped_at_chapter is not None:
        remaining = [c for c in chapters if c >= stopped_at_chapter]
        cmd = " ".join(str(c) for c in remaining)
        log.info("")
        log.info("Bot deterrent stopped the run at chapter %d.", stopped_at_chapter)
        log.info("To resume, re-export cookies and run:")
        log.info("  PYTHONPATH=src uv run python scripts/export_dhs_cookies.py")
        log.info("  PYTHONPATH=src uv run python scripts/scrape_dhs_sections.py --headed %s", cmd)


if __name__ == "__main__":
    main()
