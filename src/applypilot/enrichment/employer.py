"""Employer-name extraction.

The `site` column on a job records where the listing was scraped from
(linkedin, indeed, Workday tenants, Greenhouse boards, etc.), which is
NOT the same as the real hiring employer. For aggregator sources like
LinkedIn and Indeed, the actual company is hidden inside the job
description or the cover letter we already generated.

Two-tier extraction:
  - Tier 1 (cheap): regex against the cover letter text. The cover letter
    almost always says "at <Company>" or "<Company>'s commitment" etc.
  - Tier 2 (fallback): one short LLM call against the job description if
    Tier 1 found nothing confident.

Use `extract_employer_name(job)` as the single entry point; it returns
the best-effort employer name or None.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from applypilot.config import load_profile
from applypilot.llm import get_client

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier 1: regex against cover letter
# ---------------------------------------------------------------------------

# Word continuation uses [ \t] only (not \s) so we don't capture across
# newlines / paragraph breaks. Each pattern's group(1) is the candidate.
# Patterns ordered roughly by precision (most specific first).
_NAME = r"[A-Z][\w&.\-]+(?:[ \t]+[A-Z][\w&.\-]+){0,3}"
_END = r"(?:[\.,'’]|[ \t]+as\b|[ \t]+and\b|[ \t]+to\b|[ \t]+I\b|[ \t]+is\b)"

_COVER_PATTERNS = [
    # "Company's commitment / mission / vision / team / culture / etc."
    re.compile(
        rf"\b({_NAME})['’]s[ \t]+"
        r"(?:commitment|mission|vision|team|culture|approach|focus|"
        r"reputation|platform|values|brand|customers|users)"
    ),
    # "join (the team at) Company"
    re.compile(rf"\bjoin[ \t]+(?:the[ \t]+team[ \t]+at[ \t]+)?({_NAME}){_END}"),
    # "excited about/to join Company"
    re.compile(
        rf"\bexcited[ \t]+(?:about|to[ \t]+join|to[ \t]+contribute[ \t]+to)"
        rf"[ \t]+({_NAME}){_END}"
    ),
    # "your goals/mission/vision/team/company at Company"
    re.compile(
        rf"\byour[ \t]+(?:goals?|mission|vision|team|company|"
        rf"organization|culture)[ \t]+at[ \t]+({_NAME})"
    ),
    # bare "at Company" (lowest precision -- last resort)
    re.compile(rf"\bat[ \t]+({_NAME}){_END}"),
]

# Generic / role / boilerplate words that the regex might catch as a "name"
# at sentence start, in a salutation, or trailing. Lowercased for matching.
_GENERIC_STOPWORDS = {
    "i", "the", "your", "their", "our", "this", "that", "my", "we",
    "us", "you", "he", "she", "they", "it", "dear", "hiring", "manager",
    "sincerely", "best", "regards", "thanks", "thank", "tyler", "shambora",
    "senior", "software", "engineer", "company", "team", "role", "position",
    "about", "join", "join us", "join our team", "what", "who", "why",
}


def _clean_extracted_name(name: str) -> str:
    """Trim common prefixes regex tiers accidentally include.

    `_NAME` allows up to 4 capitalized words, which means "About Stylitics"
    can be captured as a 2-word name when the JD says "About Stylitics is...".
    """
    if not name:
        return name
    # Strip leading "About " / "The " / "At " — only the literal English prefix.
    for prefix in ("About The ", "About the ", "About ", "The ", "At "):
        if name.startswith(prefix):
            return name[len(prefix):].strip()
    return name.strip()


def _profile_stopwords() -> set[str]:
    """Tyler's previous employers from profile.resume_facts.preserved_companies.

    These appear in cover letters as "At Codal, I led..." narrative
    references, which the bare "at <Name>" pattern would mis-capture as the
    target employer. Read fresh each call so profile edits take effect
    without restart.
    """
    try:
        profile = load_profile()
    except Exception:
        return set()
    preserved = profile.get("resume_facts", {}).get("preserved_companies", []) or []
    words: set[str] = set()
    for entry in preserved:
        if not isinstance(entry, str):
            continue
        # Add the full name AND the first word (e.g. "Pack Digital" and "Pack")
        words.add(entry.strip().lower())
        first = entry.strip().split()[0].lower() if entry.strip() else ""
        if first:
            words.add(first)
    return words


# Common JD section headers that look like proper nouns to the regex.
_JD_SECTION_HEADERS = {
    "about us", "about the role", "about the company", "about the team",
    "about the opportunity", "about the position", "about the job",
    "the role", "the company", "the team", "the opportunity", "the position",
    "requirements", "responsibilities", "benefits", "qualifications",
    "responsabilidades",  # spanish for responsibilities
    "summary", "overview", "description", "location", "key responsibilities",
    "what you", "what we", "what is", "who we", "who you", "who is",
    "why join", "why us", "why this role", "why we", "your role",
    "your responsibilities", "your team", "key skills", "preferred",
    "nice to have", "must have", "compensation", "salary range",
    "perks", "equal opportunity", "join us", "join our team",
    "our story", "our mission", "our team", "our culture", "our values",
    "company overview", "role overview", "team overview", "position overview",
    "english", "spanish",  # language column headers
}

_JD_PATTERNS = [
    # "At <Company>, we/our" (first-person narrative — very strong signal)
    re.compile(rf"\bAt[ \t]+({_NAME}),[ \t]+(?:we|our)\b"),
    # "<Company> is hiring / is looking for / is seeking / seeks"
    re.compile(rf"\b({_NAME})[ \t]+(?:is\s+hiring|is\s+looking\s+for|is\s+seeking|seeks)\b"),
    # "Join <Company> as/in/to/where/and"
    re.compile(rf"\bJoin[ \t]+(?:the[ \t]+team[ \t]+at[ \t]+)?({_NAME})[ \t]+(?:as|in|to|where|and)\b"),
    # "<Company>'s mission/vision/team/etc."
    re.compile(rf"\b({_NAME})['’]s[ \t]+(?:mission|vision|team|culture|values|"
               r"customers|users|platform|product|approach|reputation)"),
    # "About <Company>" — only count if followed by a sentence about the company,
    # not a section header followed by a newline.
    re.compile(rf"^[ \t]*About[ \t]+({_NAME})[ \t]+(?:is|we|has|provides|offers|operates)",
               re.MULTILINE),
    # Markdown bold at the very TOP of the JD (first 200 chars only).
    # Captures `**Company**` or `***Company***` style LinkedIn intro lines.
    re.compile(r"\A[\s\*]{0,40}\*{1,3}\(?\w{0,3}\)?[ \t]*\*{0,2}"
               r"([A-Z][\w&.\-]+(?:[ \t]+[A-Z][\w&.\-]+){0,3})\s*\*{1,3}"),
]


def extract_employer_from_jd(job_description: str) -> str | None:
    """Extract employer name from a job description via regex.

    JDs name the employer many times, almost always in the first 500 chars.
    Position-weighted; earliest mention wins. Returns None on no match.
    """
    if not job_description:
        return None

    profile_stops = _profile_stopwords()
    candidates: list[tuple[str, int]] = []  # (name, position)

    for pat in _JD_PATTERNS:
        for m in pat.finditer(job_description):
            name = m.group(1).strip().rstrip(".,'’\n\r\t ")
            if not name or "\n" in name or "\r" in name:
                continue
            if name.endswith("'s") or name.endswith("’s"):
                name = name[:-2].rstrip()
            if not name or len(name) < 3 or len(name) > 60:
                continue
            lower = name.lower()
            # Filter generic JD section headers ("Requirements", "About Us", etc.)
            if lower in _JD_SECTION_HEADERS:
                continue
            # Filter "<Header word> X" composites like "About Us", "Our Team"
            first_word = name.split()[0].lower()
            if first_word in _GENERIC_STOPWORDS:
                continue
            if lower in profile_stops or first_word in profile_stops:
                continue
            candidates.append((name, m.start()))

    if not candidates:
        return None

    # Weight by position (early = much more likely to be the real employer).
    text_len = max(len(job_description), 1)
    score: Counter = Counter()
    for name, pos in candidates:
        weight = 5 if pos < 200 else (3 if pos < text_len * 0.15 else 1)
        score[name] += weight

    winner, _w = score.most_common(1)[0]
    cleaned = _clean_extracted_name(winner)
    if not cleaned or len(cleaned) < 3:
        return None
    log.debug("JD extraction: %s", cleaned)
    return cleaned


def extract_employer_from_cover(letter_text: str) -> str | None:
    """Extract the most-mentioned employer name from a cover letter.

    Returns None when no candidate passes the filters. Filters out the
    candidate's own previous employers (read from profile.resume_facts)
    so that narrative references like "At Pack Digital, I led..." don't
    masquerade as the target employer.
    """
    if not letter_text:
        return None

    profile_stops = _profile_stopwords()
    candidates: list[str] = []
    for pat in _COVER_PATTERNS:
        for m in pat.finditer(letter_text):
            name = m.group(1).strip().rstrip(".,'’\n\r\t ")
            if not name or "\n" in name or "\r" in name:
                continue
            # Strip trailing possessive if it survived
            if name.endswith("'s") or name.endswith("’s"):
                name = name[:-2].rstrip()
            if not name:
                continue
            # Length sanity
            if len(name) < 3 or len(name) > 60:
                continue
            # Filter generic words and the candidate's own past employers
            first_word = name.split()[0].lower()
            if first_word in _GENERIC_STOPWORDS:
                continue
            if name.lower() in profile_stops or first_word in profile_stops:
                continue
            candidates.append(name)

    if not candidates:
        return None

    # Position-weighted scoring: mentions closer to the top of the letter
    # are far more likely to be the addressed employer (target) vs. narrative
    # mentions later about past roles.
    text_len = max(len(letter_text), 1)
    score: Counter = Counter()
    for c in candidates:
        # Find earliest occurrence to weight position
        first_idx = letter_text.find(c)
        if first_idx < 0:
            first_idx = text_len
        # Earlier mention -> bigger weight. Cap at 3 for first 25% of letter.
        weight = 3 if first_idx < text_len * 0.25 else (
            2 if first_idx < text_len * 0.5 else 1
        )
        score[c] += weight

    winner, weight = score.most_common(1)[0]
    cleaned = _clean_extracted_name(winner)
    if not cleaned or len(cleaned) < 3:
        return None
    log.debug("Cover-letter extraction: %s (weighted score %d)", cleaned, weight)
    return cleaned


# ---------------------------------------------------------------------------
# Tier 2: LLM call against job description
# ---------------------------------------------------------------------------

_LLM_PROMPT = (
    "You extract the hiring company name from job descriptions. "
    "Output ONLY the company name on a single line, with no quotes, "
    "no labels, no explanation. If the description does not name a "
    "specific company, output exactly: UNKNOWN"
)


def extract_employer_from_jd_llm(job_description: str) -> str | None:
    """LLM fallback: extract employer name when regex tiers gave up."""
    if not job_description or len(job_description) < 50:
        return None

    try:
        client = get_client()
        messages = [
            {"role": "system", "content": _LLM_PROMPT},
            {"role": "user", "content": f"JOB DESCRIPTION:\n\n{job_description[:4000]}\n\nCOMPANY NAME:"},
        ]
        raw = client.chat(messages, max_tokens=40, temperature=0.0)
    except Exception as e:
        log.debug("LLM employer extraction failed: %s", e)
        return None

    name = raw.strip().splitlines()[0].strip().strip('"“”\'')
    if not name or name.upper() == "UNKNOWN":
        return None
    if len(name) > 80:
        return None
    return name


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

# Aggregator sources where `site` is the scrape origin, not the employer.
_AGGREGATORS = {
    "linkedin", "indeed", "glassdoor", "google", "ziprecruiter",
    "simplyhired", "wellfound", "dice", "powertofly", "talent.com",
    "hacker news jobs", "hackernews",
}


def site_is_aggregator(site: str | None) -> bool:
    """True when `site` is a job board (not the actual employer)."""
    if not site:
        return True
    return site.strip().lower() in _AGGREGATORS


def extract_employer_name(
    job: dict,
    cover_letter_text: str | None = None,
    allow_llm: bool = True,
) -> str | None:
    """Best-effort employer name for a job, used by the apply prompt builder.

    Resolution order (cheapest / most reliable first):
      0. If job['site'] is NOT an aggregator, use site as-is. Workday tenants
         and Greenhouse boards store the real employer in `site` already.
      1. Regex against `full_description` (highest signal — JDs name the
         employer many times, usually in the first 500 chars).
      2. Regex against the cover letter text (less reliable -- the LLM that
         wrote the letter may have hallucinated the employer for aggregator
         jobs where the JD was ambiguous).
      3. LLM call against the JD. Skipped when allow_llm=False so backfill
         loops don't burn 1000+ API calls without explicit opt-in.
    """
    # Tier 0: site IS the employer
    site = (job.get("site") or "").strip()
    if site and not site_is_aggregator(site):
        return site

    # Tier 1: regex on job description
    jd = job.get("full_description") or ""
    name = extract_employer_from_jd(jd)
    if name:
        return name

    # Tier 2: regex on cover letter
    text = cover_letter_text
    if not text:
        cl_path = job.get("cover_letter_path")
        if cl_path:
            cl_txt = Path(cl_path).with_suffix(".txt")
            if cl_txt.exists():
                try:
                    text = cl_txt.read_text(encoding="utf-8")
                except OSError:
                    text = None
    if text:
        name = extract_employer_from_cover(text)
        if name:
            # Guard against the LLM defaulting to the aggregator's name when it
            # didn't know the real employer ("excited to join LinkedIn", etc.).
            # If the cover letter just echoes the scrape source, distrust it.
            if name.lower() == site.lower() and site_is_aggregator(site):
                pass  # fall through to LLM tier
            else:
                return name

    # Tier 3: LLM fallback
    if allow_llm:
        return extract_employer_from_jd_llm(jd)

    return None
