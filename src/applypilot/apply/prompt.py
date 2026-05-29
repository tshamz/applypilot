"""Prompt builder for the autonomous job application agent.

Constructs the full instruction prompt that tells Claude Code / the AI agent
how to fill out a job application form using Playwright MCP tools. All
personal data is loaded from the user's profile -- nothing is hardcoded.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from applypilot import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yes_no(value) -> str:
    """Normalize a boolean / "Yes" / "No" / "true" / "false" to "Yes" or "No".

    Profile fields are a mix of booleans (JSON-native) and Yes/No strings.
    Forms expect "Yes" / "No" so we normalize at render time.
    """
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("yes", "true", "y", "t", "1"):
            return "Yes"
        if v in ("no", "false", "n", "f", "0"):
            return "No"
        return value  # already a meaningful string; pass through
    return "See profile"


def _resolve_display_name(personal: dict) -> str:
    """Return the candidate's preferred public name.

    Handles all three reasonable shapes:
      full_name="Tyler Shambora", preferred_name=""           -> "Tyler Shambora"
      full_name="Peter Shambora", preferred_name="Tyler Shambora" -> "Tyler Shambora"
      full_name="Peter Shambora", preferred_name="Tyler"      -> "Tyler Shambora"
    Never doubles the last name (the previous code produced "Tyler Shambora Shambora").
    """
    full_name = personal.get("full_name", "").strip()
    preferred = (personal.get("preferred_name") or "").strip()
    if not preferred:
        return full_name
    if " " in preferred:
        return preferred  # already a full preferred name; trust it
    if " " in full_name:
        last = full_name.split()[-1]
        return f"{preferred} {last}".strip()
    return preferred


def _clean_accept_patterns(patterns: list[str]) -> list[str]:
    """Filter searches.yaml accept_patterns down to specific city names.

    The patterns serve two purposes the prompt should NOT mix:
      - Country-level patterns ("United States", "USA", " US") match remote-US
        roles during discovery. They should NOT appear in the prompt's "hybrid
        or onsite in X" list -- the candidate doesn't want "hybrid in the US"
        anywhere, only in specific cities.
      - Substring artifacts (", CA", "US ") are matching hacks, not real names.

    Keep only proper-noun city/region names for the prompt's onsite list.
    """
    country_level = {"united states", "usa", "us", "remote", "anywhere"}
    cleaned: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        if not isinstance(p, str):
            continue
        s = p.strip().lstrip(",").strip()
        if not s or len(s) < 3:
            continue
        if s.lower() in country_level:
            continue
        # Skip bare 2-letter state codes (only valid in ", XX" form for matching).
        if len(s) == 2 and s.isupper():
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        cleaned.append(s)
    return cleaned


def _build_profile_summary(profile: dict) -> str:
    """Format the applicant profile section of the prompt.

    Reads all relevant fields from the profile dict and returns a
    human-readable multi-line summary for the agent.
    """
    p = profile
    personal = p["personal"]
    work_auth = p["work_authorization"]
    comp = p["compensation"]
    exp = p.get("experience", {})
    avail = p.get("availability", {})
    eeo = p.get("eeo_voluntary", {})

    lines = [
        f"Name: {personal['full_name']}",
        f"Email: {personal['email']}",
        f"Phone: {personal['phone']}",
    ]

    # Address -- handle optional fields gracefully
    addr_parts = [
        personal.get("address", ""),
        personal.get("city", ""),
        personal.get("province_state", ""),
        personal.get("country", ""),
        personal.get("postal_code", ""),
    ]
    lines.append(f"Address: {', '.join(p for p in addr_parts if p)}")

    if personal.get("linkedin_url"):
        lines.append(f"LinkedIn: {personal['linkedin_url']}")
    if personal.get("github_url"):
        lines.append(f"GitHub: {personal['github_url']}")
    if personal.get("portfolio_url"):
        lines.append(f"Portfolio: {personal['portfolio_url']}")
    if personal.get("website_url"):
        lines.append(f"Website: {personal['website_url']}")

    # Work authorization
    lines.append(f"Work Auth: {_yes_no(work_auth.get('legally_authorized_to_work'))}")
    lines.append(f"Sponsorship Needed: {_yes_no(work_auth.get('require_sponsorship'))}")
    if work_auth.get("work_permit_type"):
        lines.append(f"Work Permit: {work_auth['work_permit_type']}")

    # Compensation
    currency = comp.get("salary_currency", "USD")
    lines.append(f"Salary Expectation: ${comp['salary_expectation']} {currency}")

    # Experience
    if exp.get("years_of_experience_total"):
        lines.append(f"Years Experience: {exp['years_of_experience_total']}")
    if exp.get("education_level"):
        lines.append(f"Education: {exp['education_level']}")

    # Employment status — explicit so the agent doesn't auto-fill "Current
    # Company" with the most recent employer when the candidate is between roles.
    is_employed = exp.get("currently_employed")
    if is_employed is False:
        lines.append("Currently Employed: No (between roles)")
        most_recent_company = exp.get("most_recent_company", "")
        most_recent_title = exp.get("most_recent_title", "")
        if most_recent_company:
            lines.append(f"Most Recent Company: {most_recent_company}")
        if most_recent_title:
            lines.append(f"Most Recent Title: {most_recent_title}")
    elif is_employed is True:
        lines.append("Currently Employed: Yes")
        # When employed, current_company and current_title come from the most
        # recent role -- form labels are usually fine either way.
        cur_company = exp.get("most_recent_company") or exp.get("current_company", "")
        cur_title = exp.get("most_recent_title") or exp.get("current_title", "")
        if cur_company:
            lines.append(f"Current Company: {cur_company}")
        if cur_title:
            lines.append(f"Current Title: {cur_title}")

    # Availability
    lines.append(f"Available: {avail.get('earliest_start_date', 'Immediately')}")
    lines.append(f"Available for Full-Time: {_yes_no(avail.get('available_for_full_time', True))}")
    lines.append(f"Available for Contract: {_yes_no(avail.get('available_for_contract', False))}")
    lines.append(f"Open to Relocation: {_yes_no(avail.get('open_to_relocation', False))}")

    # Standard responses
    lines.extend([
        "Age 18+: Yes",
        "Background Check: Yes",
        "Felony: No",
        "How Heard: Online Job Board",
    ])

    # Past-employer truth set. "Previously Worked Here" / "Are you a current
    # employee" / "Are you a former employee" / "Have you ever applied/worked"
    # questions get answered against THIS list. Anything not in it -> No.
    resume_facts = p.get("resume_facts", {})
    past_companies = resume_facts.get("preserved_companies", [])
    if past_companies:
        lines.append(
            "Previously Worked At (real, verifiable employment history -- "
            "for 'have you worked here?' / current+former employee questions, "
            "answer YES only if the asking company is in this list, otherwise NO): "
            + ", ".join(past_companies)
        )
    else:
        lines.append("Previously Worked Here: No")

    # Company-type descriptors for "experience at a <type> company?" questions
    # (SaaS, B2B, agency, FAANG, etc.) so the agent can match honestly instead
    # of defaulting to No when the resume doesn't literally use the category word.
    desc_map = resume_facts.get("company_descriptions", {})
    if desc_map:
        lines.append("Company Types (for 'experience at X type of company?' questions):")
        for co, desc in desc_map.items():
            lines.append(f"  - {co}: {desc}")

    # EEO
    lines.append(f"Gender: {eeo.get('gender', 'Decline to self-identify')}")
    lines.append(f"Race: {eeo.get('race_ethnicity', 'Decline to self-identify')}")
    lines.append(f"Veteran: {eeo.get('veteran_status', 'I am not a protected veteran')}")
    lines.append(f"Disability: {eeo.get('disability_status', 'I do not wish to answer')}")

    return "\n".join(lines)


def _build_location_check(profile: dict, search_config: dict) -> str:
    """Build the eligibility check section of the prompt.

    Covers work-type (full-time vs contract), location, and "nearshore"
    employer-location traps. All three are quick-reject signals worth
    checking before any form-fill work.
    """
    personal = profile["personal"]
    location_cfg = search_config.get("location", {})
    accept_patterns = location_cfg.get("accept_patterns", [])
    primary_city = personal.get("city", location_cfg.get("primary", "your city"))

    cleaned = _clean_accept_patterns(accept_patterns)
    city_list = ", ".join(cleaned) if cleaned else primary_city

    return f"""== ELIGIBILITY CHECK (do this FIRST -- before any form-fill) ==
Read the job description carefully. ALL three checks below must pass. If any one fails, output the corresponding RESULT and stop. Do NOT submit applications for jobs you're not eligible for.

--- 1. WORK TYPE ---
- "Full-time" / "Permanent" / no employment type mentioned -> OK.
- "Contract" / "Contract-to-hire" / "Freelance" / "1099" / "C2C" / "Temp" -> NOT ELIGIBLE. Output RESULT:FAILED:contract_role
- "Part-time only" (no full-time option) -> NOT ELIGIBLE. Output RESULT:FAILED:part_time_only
- Title contains "$XX/hr", "/hour", "Hourly", "(Contract)", "(Freelance)" -> NOT ELIGIBLE.
- Visible "Contract" pill / badge on the job listing page (LinkedIn, Indeed, etc.) -> NOT ELIGIBLE.

--- 2. LOCATION ---
- "Remote" / "work from anywhere" -> OK *only if* the posting does not restrict
  remote to a specific non-US country. Read carefully -- "Remote" alone is OK,
  but "Remote - Canada" / "Remote (Canada)" / "Remote in Canada" / "Canada
  Remote" / "Remote, Toronto" / "Remote within EMEA" / "Remote UK only" etc.
  are country-bound remotes the candidate is NOT eligible for.
- "Hybrid" or "onsite" in {city_list} -> OK.
- "Hybrid" or "onsite" in another city BUT the posting also says "remote OK" -> OK.
- "Onsite only" / "hybrid only" outside the cities above, no remote option -> NOT ELIGIBLE. Output RESULT:FAILED:not_eligible_location
- Listed city is overseas (India, Philippines, Europe, LATAM, Canada, etc.)
  with no US-remote option -> NOT ELIGIBLE.

--- 2b. CURRENCY / COUNTRY-RESTRICTED PAY SIGNAL ---
A salary posted only in a non-USD currency (CAD, GBP, EUR, AUD, INR, MXN,
BRL, etc.) without a USD equivalent is a strong signal the role is for that
country's residents only. Examples:
- "$180,000-$230,000 CAD" with no USD equivalent -> Canada-only role -> NOT ELIGIBLE
- "£70K-£90K" -> UK-only -> NOT ELIGIBLE
- "€60K-€80K" -> EU country -> NOT ELIGIBLE
USD-only or dual-currency postings (e.g. "$180K USD / CAD $230K") are fine.
When in doubt, treat non-USD-only postings as country-restricted: Output
RESULT:FAILED:not_eligible_location

--- 3. EMPLOYER / "NEARSHORE" TRAP ---
Some jobs say "Remote" but the EMPLOYER is hiring nearshore talent for North American clients -- they only want non-US workers. Red flags in the description:
- "nearshore" / "near-shore" / "offshore"
- "Latin America" / "LATAM" / "for North American clients" / "for US-based clients"
- A "Latin America city" / "country of residence" field asking for a non-US location.
- Company explicitly headquartered in LATAM/India/PH and the role description targets their region.
Any of the above -> NOT ELIGIBLE. Output RESULT:FAILED:nearshore_role

Cannot determine after a careful read? -> Continue, but answer screening questions honestly. Let the form reject you if needed."""


def _build_salary_section(profile: dict) -> str:
    """Build the salary negotiation instructions.

    Reads target / floor / ceiling from the profile's compensation section so
    that "floor" actually means salary_range_min (the lowest acceptable
    number), not salary_expectation (the desired number). Earlier behavior
    conflated the two and rejected anything below the target.
    """
    comp = profile["compensation"]
    currency = comp.get("salary_currency", "USD")
    target = comp["salary_expectation"]
    range_min = comp.get("salary_range_min", target)
    range_max = comp.get("salary_range_max",
                         str(int(target) + 50000) if target.isdigit() else target)
    conversion_note = comp.get("currency_conversion_note", "")

    # Compute example hourly rates at floor / target / ceiling.
    try:
        examples = [
            (f"${int(range_min) // 1000}K", int(range_min) // 2080),
            (f"${int(target)    // 1000}K", int(target)    // 2080),
            (f"${int(range_max) // 1000}K", int(range_max) // 2080),
        ]
        hourly_line = ", ".join(f"{sal} = ${hr}/hr" for sal, hr in examples)
    except (ValueError, TypeError):
        hourly_line = "Divide annual salary by 2080"

    if conversion_note:
        convert_line = f"Posting is in a different currency? -> {conversion_note}"
    else:
        convert_line = "Posting is in a different currency? -> Convert their midpoint to USD; answer with their midpoint if it lands inside your range."

    return f"""== SALARY (think, don't just copy) ==
Target: ${target} {currency}. Acceptable floor: ${range_min} {currency} (never below). Ceiling for ranges: ${range_max} {currency}.

Decision tree:
1. Asked for a single number with no posted range? -> ${target} {currency}.
2. Asked for a range with no posted range? -> "${range_min}-${range_max} {currency}".
3. Job posting shows their range?
   a. Their range overlaps yours -> answer with the midpoint of their range (assuming above ${range_min}).
   b. Their max is below ${range_min} -> answer with ${range_min}. The system may reject; that's fine.
   c. Their entire range is above ${range_max} -> answer with their midpoint.
4. Title says Senior / Staff / Lead / Principal / Architect / level II+ -> never answer below ${target} {currency} unless explicitly capped by their posted max.
5. {convert_line}
6. Hourly rate? -> Divide your annual answer by 2080. ({hourly_line})"""


def _build_screening_section(profile: dict) -> str:
    """Build the screening questions guidance section.

    Honors the profile for relocation and EEO -- previous behavior hardcoded
    "cannot relocate" and "Decline to self-identify" regardless of profile.
    """
    personal = profile["personal"]
    exp = profile.get("experience", {})
    avail = profile.get("availability", {})
    eeo = profile.get("eeo_voluntary", {})
    city = personal.get("city", "their city")
    years = exp.get("years_of_experience_total", "multiple")
    target_role = exp.get("target_role", personal.get("current_job_title", "software engineer"))
    work_auth = profile["work_authorization"]

    # Relocation
    if avail.get("open_to_relocation"):
        relocation_line = f"Location/relocation: lives in {city}, open to relocation for the right role."
    else:
        relocation_line = f"Location/relocation: lives in {city}, not actively looking to relocate."

    # EEO -- write out exactly what the profile says. If a field is "Decline to
    # self-identify" or similar, the agent will paste that verbatim. If the
    # candidate disclosed, the agent uses that.
    eeo_lines = [
        f'  - Gender: "{eeo.get("gender", "Decline to self-identify")}"',
        f'  - Race/Ethnicity: "{eeo.get("race_ethnicity", "Decline to self-identify")}"',
        f'  - Veteran Status: "{eeo.get("veteran_status", "Decline to self-identify")}"',
        f'  - Disability Status: "{eeo.get("disability_status", "Decline to self-identify")}"',
    ]
    eeo_block = "\n".join(eeo_lines)

    return f"""== SCREENING QUESTIONS (be strategic) ==
Hard facts -> answer truthfully from the profile. No guessing. This includes:
  - {relocation_line}
  - Work authorization: {_yes_no(work_auth.get('legally_authorized_to_work'))}
  - Citizenship, clearance, licenses, certifications: answer from profile only
  - Criminal/background: answer from profile only

Employment-history questions (CRITICAL -- NEVER fabricate prior employment) ->
  - "Have you ever worked at [Company]?" / "Are you a current employee?" /
    "Are you a former employee?" / "Have you applied here before?" ->
    Read the "Previously Worked At" line in the APPLICANT PROFILE. The asking
    company is the company you're applying TO right now (see JOB block).
    Answer YES only if that company is in the list. Otherwise: NO.
    The cover letter and tailored resume MENTION the target company
    extensively -- that is NOT evidence of past employment, it's the role
    pitch. Do NOT mistake repeated mentions for employment history.
  - "Years of experience" / "Years at most recent role" -> use what's on the
    resume; never inflate.

Company-type questions ("Do you have experience at a SaaS company?", "B2B
experience?", "Startup experience?", "FAANG?", "Agency?", etc.) ->
  Read the "Company Types" map in the APPLICANT PROFILE. If ANY past company
  matches the asked-about type, answer YES. Software the candidate has built
  for paying business customers IS B2B experience even if the resume doesn't
  literally say "B2B". A cloud platform serving multiple recurring customers
  IS SaaS even if the resume doesn't literally say "SaaS". Don't over-narrow
  on category labels.

Skills and tools -> be confident. This candidate is a {target_role} with {years} years experience. If the question asks "Do you have experience with [tool]?" and it's in the same domain (DevOps, backend, ML, cloud, automation), answer YES. Software engineers learn tools fast. Don't sell short.

Open-ended questions ("Why do you want this role?", "Tell us about yourself", "What interests you?") -> Write 2-3 sentences. Be specific to THIS job. Reference something from the job description. Connect it to a real achievement from the resume. No generic fluff. No "I am passionate about..." -- sound like a real person.

EEO / voluntary self-identification -> use the candidate's stated answers verbatim:
{eeo_block}"""


def _build_hard_rules(profile: dict) -> str:
    """Build the hard rules section with work auth and name from profile."""
    personal = profile["personal"]
    work_auth = profile["work_authorization"]

    full_name = personal["full_name"]
    display_name = _resolve_display_name(personal)
    preferred_name = (personal.get("preferred_name") or "").strip()

    # Build work auth rule dynamically. Render booleans as Yes/No so the agent
    # never has to guess what "True"/"False" mean on a form.
    sponsorship = _yes_no(work_auth.get("require_sponsorship"))
    permit_type = work_auth.get("work_permit_type", "")

    work_auth_rule = "Work auth: Answer truthfully from profile."
    if permit_type:
        work_auth_rule = f"Work auth: {permit_type}. Sponsorship needed: {sponsorship}."

    name_rule = f'Name: Legal name = {full_name}.'
    if preferred_name and preferred_name != full_name:
        name_rule += f' Preferred name = {preferred_name}. Use "{display_name}" unless a field specifically says "legal name".'

    return f"""== HARD RULES (never break these) ==
1. Never lie about: citizenship, work authorization, criminal history, education credentials, security clearance, licenses, OR employment history. Specifically:
   - NEVER claim past or current employment at a company not in the candidate's
     APPLICANT PROFILE "Previously Worked At" list. That list is the complete,
     verifiable truth set for the candidate's job history.
   - For "Have you ever worked at [Company]?" / "Are you a current or former
     employee?" / "Have you applied before?" -> if the asking company is NOT
     in "Previously Worked At" the answer is NO, period. The fact that the
     cover letter / tailored resume mentions the target company many times
     is NOT evidence of past employment; it is the pitch for the new role.
2. {work_auth_rule}
3. {name_rule}"""


def _build_captcha_section() -> str:
    """Build the CAPTCHA detection and solving instructions.

    Reads the CapSolver API key from environment. The CAPTCHA section
    contains no personal data -- it's the same for every user.
    """
    config.load_env()
    capsolver_key = os.environ.get("CAPSOLVER_API_KEY", "")

    return f"""== CAPTCHA ==
You solve CAPTCHAs via the CapSolver REST API. No browser extension. You control the entire flow.
API key: {capsolver_key or 'NOT CONFIGURED — skip to MANUAL FALLBACK for all CAPTCHAs'}
API base: https://api.capsolver.com

CRITICAL RULE: When ANY CAPTCHA appears (hCaptcha, reCAPTCHA, Turnstile -- regardless of what it looks like visually), you MUST:
1. Run CAPTCHA DETECT to get the type and sitekey
2. Run CAPTCHA SOLVE (createTask -> poll -> inject) with the CapSolver API
3. ONLY go to MANUAL FALLBACK if CapSolver returns errorId > 0
Do NOT skip the API call based on what the CAPTCHA looks like. CapSolver solves CAPTCHAs server-side -- it does NOT need to see or interact with images, puzzles, or games. Even "drag the pipe" or "click all traffic lights" hCaptchas are solved via API token, not visually. ALWAYS try the API first.

--- CAPTCHA DETECT ---
Run this browser_evaluate after every navigation, Apply/Submit/Login click, or when a page feels stuck.
IMPORTANT: Detection order matters. hCaptcha elements also have data-sitekey, so check hCaptcha BEFORE reCAPTCHA.

browser_evaluate function: () => {{{{
  const r = {{}};
  const url = window.location.href;
  // 1. hCaptcha (check FIRST -- hCaptcha uses data-sitekey too)
  const hc = document.querySelector('.h-captcha, [data-hcaptcha-sitekey]');
  if (hc) {{{{
    r.type = 'hcaptcha'; r.sitekey = hc.dataset.sitekey || hc.dataset.hcaptchaSitekey;
  }}}}
  if (!r.type && document.querySelector('script[src*="hcaptcha.com"], iframe[src*="hcaptcha.com"]')) {{{{
    const el = document.querySelector('[data-sitekey]');
    if (el) {{{{ r.type = 'hcaptcha'; r.sitekey = el.dataset.sitekey; }}}}
  }}}}
  // 2. Cloudflare Turnstile
  if (!r.type) {{{{
    const cf = document.querySelector('.cf-turnstile, [data-turnstile-sitekey]');
    if (cf) {{{{
      r.type = 'turnstile'; r.sitekey = cf.dataset.sitekey || cf.dataset.turnstileSitekey;
      if (cf.dataset.action) r.action = cf.dataset.action;
      if (cf.dataset.cdata) r.cdata = cf.dataset.cdata;
    }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="challenges.cloudflare.com"]')) {{{{
    r.type = 'turnstile_script_only'; r.note = 'Wait 3s and re-detect.';
  }}}}
  // 3. reCAPTCHA v3 (invisible, loaded via render= param)
  if (!r.type) {{{{
    const s = document.querySelector('script[src*="recaptcha"][src*="render="]');
    if (s) {{{{
      const m = s.src.match(/render=([^&]+)/);
      if (m && m[1] !== 'explicit') {{{{ r.type = 'recaptchav3'; r.sitekey = m[1]; }}}}
    }}}}
  }}}}
  // 4. reCAPTCHA v2 (checkbox or invisible)
  if (!r.type) {{{{
    const rc = document.querySelector('.g-recaptcha');
    if (rc) {{{{ r.type = 'recaptchav2'; r.sitekey = rc.dataset.sitekey; }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="recaptcha"]')) {{{{
    const el = document.querySelector('[data-sitekey]');
    if (el) {{{{ r.type = 'recaptchav2'; r.sitekey = el.dataset.sitekey; }}}}
  }}}}
  // 5. FunCaptcha (Arkose Labs)
  if (!r.type) {{{{
    const fc = document.querySelector('#FunCaptcha, [data-pkey], .funcaptcha');
    if (fc) {{{{ r.type = 'funcaptcha'; r.sitekey = fc.dataset.pkey; }}}}
  }}}}
  if (!r.type && document.querySelector('script[src*="arkoselabs"], script[src*="funcaptcha"]')) {{{{
    const el = document.querySelector('[data-pkey]');
    if (el) {{{{ r.type = 'funcaptcha'; r.sitekey = el.dataset.pkey; }}}}
  }}}}
  if (r.type) {{{{ r.url = url; return r; }}}}
  return null;
}}}}

Result actions:
- null -> no CAPTCHA. Continue normally.
- "turnstile_script_only" -> browser_wait_for time: 3, re-run detect.
- Any other type -> proceed to CAPTCHA SOLVE below.

--- CAPTCHA SOLVE ---
Three steps: createTask -> poll -> inject. Do each as a separate browser_evaluate call.

STEP 1 -- CREATE TASK (copy this exactly, fill in the 3 placeholders):
browser_evaluate function: async () => {{{{
  const r = await fetch('https://api.capsolver.com/createTask', {{{{
    method: 'POST',
    headers: {{{{'Content-Type': 'application/json'}}}},
    body: JSON.stringify({{{{
      clientKey: '{capsolver_key}',
      task: {{{{
        type: 'TASK_TYPE',
        websiteURL: 'PAGE_URL',
        websiteKey: 'SITE_KEY'
      }}}}
    }}}})
  }}}});
  return await r.json();
}}}}

TASK_TYPE values (use EXACTLY these strings):
  hcaptcha     -> HCaptchaTaskProxyLess
  recaptchav2  -> ReCaptchaV2TaskProxyLess
  recaptchav3  -> ReCaptchaV3TaskProxyLess
  turnstile    -> AntiTurnstileTaskProxyLess
  funcaptcha   -> FunCaptchaTaskProxyLess

PAGE_URL = the url from detect result. SITE_KEY = the sitekey from detect result.
For recaptchav3: add "pageAction": "submit" to the task object (or the actual action found in page scripts).
For turnstile: add "metadata": {{"action": "...", "cdata": "..."}} if those were in detect result.

Response: {{"errorId": 0, "taskId": "abc123"}} on success.
If errorId > 0 -> CAPTCHA SOLVE failed. Go to MANUAL FALLBACK.

STEP 2 -- POLL (replace TASK_ID with the taskId from step 1):
Loop: browser_wait_for time: 3, then run:
browser_evaluate function: async () => {{{{
  const r = await fetch('https://api.capsolver.com/getTaskResult', {{{{
    method: 'POST',
    headers: {{{{'Content-Type': 'application/json'}}}},
    body: JSON.stringify({{{{
      clientKey: '{capsolver_key}',
      taskId: 'TASK_ID'
    }}}})
  }}}});
  return await r.json();
}}}}

- status "processing" -> wait 3s, poll again. Max 10 polls (30s).
- status "ready" -> extract token:
    reCAPTCHA: solution.gRecaptchaResponse
    hCaptcha:  solution.gRecaptchaResponse
    Turnstile: solution.token
- errorId > 0 or 30s timeout -> MANUAL FALLBACK.

STEP 3 -- INJECT TOKEN (replace THE_TOKEN with actual token string):

For reCAPTCHA v2/v3:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  document.querySelectorAll('[name="g-recaptcha-response"]').forEach(el => {{{{ el.value = token; el.style.display = 'block'; }}}});
  if (window.___grecaptcha_cfg) {{{{
    const clients = window.___grecaptcha_cfg.clients;
    for (const key in clients) {{{{
      const walk = (obj, d) => {{{{
        if (d > 4 || !obj) return;
        for (const k in obj) {{{{
          if (typeof obj[k] === 'function' && k.length < 3) try {{{{ obj[k](token); }}}} catch(e) {{{{}}}}
          else if (typeof obj[k] === 'object') walk(obj[k], d+1);
        }}}}
      }}}};
      walk(clients[key], 0);
    }}}}
  }}}}
  return 'injected';
}}}}

For hCaptcha:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const ta = document.querySelector('[name="h-captcha-response"], textarea[name*="hcaptcha"]');
  if (ta) ta.value = token;
  document.querySelectorAll('iframe[data-hcaptcha-response]').forEach(f => f.setAttribute('data-hcaptcha-response', token));
  const cb = document.querySelector('[data-hcaptcha-widget-id]');
  if (cb && window.hcaptcha) try {{{{ window.hcaptcha.getResponse(cb.dataset.hcaptchaWidgetId); }}}} catch(e) {{{{}}}}
  return 'injected';
}}}}

For Turnstile:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const inp = document.querySelector('[name="cf-turnstile-response"], input[name*="turnstile"]');
  if (inp) inp.value = token;
  if (window.turnstile) try {{{{ const w = document.querySelector('.cf-turnstile'); if (w) window.turnstile.getResponse(w); }}}} catch(e) {{{{}}}}
  return 'injected';
}}}}

For FunCaptcha:
browser_evaluate function: () => {{{{
  const token = 'THE_TOKEN';
  const inp = document.querySelector('#FunCaptcha-Token, input[name="fc-token"]');
  if (inp) inp.value = token;
  if (window.ArkoseEnforcement) try {{{{ window.ArkoseEnforcement.setConfig({{{{data: {{{{blob: token}}}}}}}}) }}}} catch(e) {{{{}}}}
  return 'injected';
}}}}

After injecting: browser_wait_for time: 2, then snapshot.
- Widget gone or green check -> success. Click Submit if needed.
- No change -> click Submit/Verify/Continue button (some sites need it).
- Still stuck -> token may have expired (~2 min lifetime). Re-run from STEP 1.

--- MANUAL FALLBACK ---
You should ONLY be here if CapSolver createTask returned errorId > 0. If you haven't tried CapSolver yet, GO BACK and try it first.
If CapSolver genuinely failed (errorId > 0):
1. Audio challenge: Look for "audio" or "accessibility" button -> click it for an easier challenge.
2. Text/logic puzzles: Solve them yourself. Think step by step. Common tricks: "All but 9 die" = 9 left. "3 sisters and 4 brothers, how many siblings?" = 7.
3. Simple text captchas ("What is 3+7?", "Type the word") -> solve them.
4. All else fails -> Output RESULT:CAPTCHA."""


def build_prompt(job: dict, tailored_resume: str,
                 cover_letter: str | None = None,
                 dry_run: bool = False) -> str:
    """Build the full instruction prompt for the apply agent.

    Loads the user profile and search config internally. All personal data
    comes from the profile -- nothing is hardcoded.

    Args:
        job: Job dict from the database (must have url, title, site,
             application_url, fit_score, tailored_resume_path).
        tailored_resume: Plain-text content of the tailored resume.
        cover_letter: Optional plain-text cover letter content.
        dry_run: If True, tell the agent not to click Submit.

    Returns:
        Complete prompt string for the AI agent.
    """
    profile = config.load_profile()
    search_config = config.load_search_config()
    personal = profile["personal"]

    # --- Resolve resume PDF path ---
    resume_path = job.get("tailored_resume_path")
    if not resume_path:
        raise ValueError(f"No tailored resume for job: {job.get('title', 'unknown')}")

    src_pdf = Path(resume_path).with_suffix(".pdf").resolve()
    if not src_pdf.exists():
        raise ValueError(f"Resume PDF not found: {src_pdf}")

    # Copy to a clean filename for upload (recruiters see the filename)
    full_name = personal["full_name"]
    name_slug = full_name.replace(" ", "_")
    dest_dir = config.APPLY_WORKER_DIR / "current"
    dest_dir.mkdir(parents=True, exist_ok=True)
    upload_pdf = dest_dir / f"{name_slug}_Resume.pdf"
    shutil.copy(str(src_pdf), str(upload_pdf))
    pdf_path = str(upload_pdf)

    # --- Cover letter handling ---
    cover_letter_text = cover_letter or ""
    cl_upload_path = ""
    cl_path = job.get("cover_letter_path")
    if cl_path and Path(cl_path).exists():
        cl_src = Path(cl_path)
        # Read text from .txt sibling (PDF is binary)
        cl_txt = cl_src.with_suffix(".txt")
        if cl_txt.exists():
            cover_letter_text = cl_txt.read_text(encoding="utf-8")
        elif cl_src.suffix == ".txt":
            cover_letter_text = cl_src.read_text(encoding="utf-8")
        # Upload must be PDF
        cl_pdf_src = cl_src.with_suffix(".pdf")
        if cl_pdf_src.exists():
            cl_upload = dest_dir / f"{name_slug}_Cover_Letter.pdf"
            shutil.copy(str(cl_pdf_src), str(cl_upload))
            cl_upload_path = str(cl_upload)

    # --- Build all prompt sections ---
    profile_summary = _build_profile_summary(profile)
    location_check = _build_location_check(profile, search_config)
    salary_section = _build_salary_section(profile)
    screening_section = _build_screening_section(profile)
    hard_rules = _build_hard_rules(profile)
    captcha_section = _build_captcha_section()

    # Cover letter fallback text
    city = personal.get("city", "the area")
    if not cover_letter_text:
        cl_display = (
            f"None available. Skip if optional. If required, write 2 factual "
            f"sentences: (1) relevant experience from the resume that matches "
            f"this role, (2) available immediately and based in {city}."
        )
    else:
        cl_display = cover_letter_text

    # Phone digits only (for fields with country prefix)
    phone_digits = "".join(c for c in personal.get("phone", "") if c.isdigit())

    # SSO domains the agent cannot sign into (loaded from config/sites.yaml)
    from applypilot.config import load_blocked_sso
    blocked_sso = load_blocked_sso()

    # Preferred display name (same logic as _build_hard_rules)
    display_name = _resolve_display_name(personal)

    # Dry-run overrides. The single weak "don't click submit" instruction
    # buried in step 10 lost to a dozen "Submit the application" mentions
    # elsewhere -- agents followed the dominant signal and submitted anyway.
    # Now: a giant banner at the very top, a hard rule near the top of the
    # rules list, and a softened step 10 that no longer says "click Submit".
    if dry_run:
        opener = (
            "You are running in DRY-RUN mode for testing. Your mission is to "
            "navigate the application, fill every field correctly, and STOP "
            "at the review screen WITHOUT submitting. Submitting in dry-run "
            "mode is a hard failure."
        )
        dry_run_banner = (
            "\n\n"
            "============================================================\n"
            "  ⚠️  DRY RUN MODE  ⚠️\n"
            "============================================================\n"
            "  DO NOT submit this application. DO NOT click Submit / Apply /\n"
            "  Send / Finish / Complete / any final action button.\n"
            "\n"
            "  Your job is to: navigate to the application form, fill every\n"
            "  field, then STOP at the final review screen. Output\n"
            "  RESULT:APPLIED with the note (dry-run: stopped before submit).\n"
            "\n"
            "  If you find yourself about to click a final-action button --\n"
            "  STOP. Take a snapshot of the review state and output the\n"
            "  RESULT instead.\n"
            "============================================================\n"
        )
        submit_instruction = (
            "DRY-RUN MODE -- DO NOT click Submit/Apply/Send/Finish. Take a "
            "snapshot of the fully-filled review screen, then output "
            "RESULT:APPLIED with the note (dry-run: stopped before submit)."
        )
    else:
        opener = (
            "You are an autonomous job application agent. Your ONE mission: "
            "get this candidate an interview. You have all the information "
            "and tools. Think strategically. Act decisively. Submit the "
            "application."
        )
        dry_run_banner = ""
        submit_instruction = (
            "BEFORE clicking Submit/Apply, take a snapshot and review EVERY "
            "field on the page. Verify all data matches the APPLICANT "
            "PROFILE and TAILORED RESUME -- name, email, phone, location, "
            "work auth, resume uploaded, cover letter if applicable. If "
            "anything is wrong or missing, fix it FIRST. Only click Submit "
            "after confirming everything is correct."
        )

    # Prefer the real hiring employer (resolved during cover-letter generation
    # for aggregator-sourced jobs) over the scrape source. Falls back to site
    # for jobs from Workday/Greenhouse/direct sources where site IS the employer.
    company = (job.get("employer_name") or job.get("site") or "Unknown").strip()

    # When dry-run is on, also soften the "submit the application" mission
    # statements that compete with the dry-run banner.
    if dry_run:
        mission_block = (
            "== YOUR MISSION ==\n"
            "Navigate to the application form. Fill every field accurately "
            "using the profile, resume, and cover letter. Then STOP at the "
            "review screen without submitting. Output RESULT:APPLIED with "
            "the note (dry-run: stopped before submit).\n\n"
            "If something unexpected happens, figure it out yourself -- "
            "navigate, read content, try buttons. The goal is to reach the "
            "review screen with a complete form, NOT to submit."
        )
    else:
        mission_block = (
            "== YOUR MISSION ==\n"
            "Submit a complete, accurate application. Use the profile and "
            "resume as source data -- adapt to fit each form's format.\n\n"
            "If something unexpected happens and these instructions don't "
            "cover it, figure it out yourself. You are autonomous. Navigate "
            "pages, read content, try buttons, explore the site. The goal is "
            "always the same: submit the application. Do whatever it takes "
            "to reach that goal."
        )

    prompt = f"""{opener}{dry_run_banner}

== JOB ==
URL: {job.get('application_url') or job['url']}
Title: {job['title']}
Company: {company}
Source: {job.get('site', 'Unknown')}
Fit Score: {job.get('fit_score', 'N/A')}/10

== FILES ==
Resume PDF (upload this): {pdf_path}
Cover Letter PDF (upload if asked): {cl_upload_path or "N/A"}

== RESUME TEXT (use when filling text fields) ==
{tailored_resume}

== COVER LETTER TEXT (paste if text field, upload PDF if file field) ==
{cl_display}

== APPLICANT PROFILE ==
{profile_summary}

{mission_block}

{hard_rules}

== NEVER DO THESE (immediate RESULT:FAILED if encountered) ==
- NEVER grant camera, microphone, screen sharing, or location permissions. If a site requests them -> RESULT:FAILED:unsafe_permissions
- NEVER do video/audio verification, selfie capture, ID photo upload, or biometric anything -> RESULT:FAILED:unsafe_verification
- NEVER set up a freelancing profile (Mercor, Toptal, Upwork, Fiverr, Turing, etc.). These are contractor marketplaces, not job applications -> RESULT:FAILED:not_a_job_application
- NEVER agree to hourly/contract rates, availability calendars, or "set your rate" flows. You are applying for FULL-TIME salaried positions only.
- NEVER install browser extensions, download executables, or run assessment software.
- NEVER enter payment info, bank details, or SSN/SIN.
- NEVER click "Allow" on any browser permission popup. Always deny/block.
- If the site is NOT a job application form (it's a profile builder, skills marketplace, talent network signup, coding assessment platform) -> RESULT:FAILED:not_a_job_application

{location_check}

{salary_section}

{screening_section}

== STEP-BY-STEP ==
1. browser_navigate to the job URL.
2. browser_snapshot to read the page. Then run CAPTCHA DETECT (see CAPTCHA section). If a CAPTCHA is found, solve it before continuing.
3. LOCATION CHECK. Read the page for location info. If not eligible, output RESULT and stop.
4. Find and click the Apply button. If email-only (page says "email resume to X"):
   - send_email with subject "Application for {job['title']} -- {display_name}", body = 2-3 sentence pitch + contact info, attach resume PDF: ["{pdf_path}"]
   - Output RESULT:APPLIED. Done.
   After clicking Apply: browser_snapshot. Run CAPTCHA DETECT -- many sites trigger CAPTCHAs right after the Apply click. If found, solve before continuing.
5. Login wall?
   5a. FIRST: check the URL. If you landed on {', '.join(blocked_sso)}, or any SSO/OAuth page -> STOP. Output RESULT:FAILED:sso_required. Do NOT try to sign in to Google/Microsoft/SSO.
   5b. Check for popups. Run browser_tabs action "list". If a new tab/window appeared (login popup), switch to it with browser_tabs action "select". Check the URL there too -- if it's SSO -> RESULT:FAILED:sso_required.
   5c. Regular login form (employer's own site)? Try sign in: {personal['email']} / {personal.get('password', '')}
   5d. After clicking Login/Sign-in: run CAPTCHA DETECT. Login pages frequently have invisible CAPTCHAs that silently block form submissions. If found, solve it then retry login.
   5e. Sign in failed? Try sign up with same email and password.
   5f. Need email verification? Use search_emails + read_email to get the code.
   5g. After login, run browser_tabs action "list" again. Switch back to the application tab if needed.
   5h. All failed? Output RESULT:FAILED:login_issue. Do not loop.
6. Upload resume. ALWAYS upload fresh -- delete any existing resume first, then browser_file_upload with the PDF path above. This is the tailored resume for THIS job. Non-negotiable.
7. Upload cover letter if there's a field for it. Text field -> paste the cover letter text. File upload -> use the cover letter PDF path.
8. Check ALL pre-filled fields. ATS systems parse your resume and auto-fill -- it's often WRONG.
   - "Current Job Title" or "Most Recent Title" -> use the title from the TAILORED RESUME summary, NOT whatever the parser guessed.
   - "Current Company" / "Currently work at" / "Where do you currently work?" (present tense) -> read the APPLICANT PROFILE's "Currently Employed" line. If "No (between roles)", leave blank, write "N/A", or use "Between roles" (whichever the field accepts). DO NOT put the most-recent company there -- that would contradict the resume's end date and read as sloppy.
   - "Most Recent Company" / "Last Employer" / "Previous Company" / "Current OR most recent" -> use the "Most Recent Company" from the APPLICANT PROFILE.
   - "Are you currently employed?" Yes/No -> use the profile's "Currently Employed" answer verbatim.
   - Compare every other field to the APPLICANT PROFILE. Fix mismatches. Fill empty fields.
9. Answer screening questions using the rules above.
10. {submit_instruction}
11. After submit: browser_snapshot. Run CAPTCHA DETECT -- submit buttons often trigger invisible CAPTCHAs. If found, solve it (the form will auto-submit once the token clears, or you may need to click Submit again). Then check for new tabs (browser_tabs action: "list"). Switch to newest, close old. Snapshot to confirm submission. Look for "thank you" or "application received".
12. Output your result.

== RESULT CODES (output EXACTLY one) ==
RESULT:APPLIED -- submitted successfully
RESULT:EXPIRED -- job closed or no longer accepting applications
RESULT:CAPTCHA -- blocked by unsolvable captcha
RESULT:LOGIN_ISSUE -- could not sign in or create account
RESULT:FAILED:not_eligible_location -- onsite outside acceptable area, no remote option
RESULT:FAILED:not_eligible_work_auth -- requires unauthorized work location
RESULT:FAILED:contract_role -- job is contract / freelance / 1099 / hourly, not full-time
RESULT:FAILED:part_time_only -- job is part-time with no full-time option
RESULT:FAILED:nearshore_role -- employer is hiring nearshore/LATAM talent for US clients
RESULT:FAILED:reason -- any other failure (brief reason)

== BROWSER EFFICIENCY ==
- browser_snapshot ONCE per page to understand it. Then use browser_take_screenshot to check results (10x less memory).
- Only snapshot again when you need element refs to click/fill.
- Multi-page forms (Workday, Taleo, iCIMS): snapshot each new page, fill all fields, click Next/Continue. Repeat until final review page.
- Fill ALL fields in ONE browser_fill_form call. Not one at a time.
- Keep your thinking SHORT. Don't repeat page structure back.
- CAPTCHA AWARENESS: After any navigation, Apply/Submit/Login click, or when a page feels stuck -- run CAPTCHA DETECT (see CAPTCHA section). Invisible CAPTCHAs (Turnstile, reCAPTCHA v3) show NO visual widget but block form submissions silently. The detect script finds them even when invisible.

== FORM TRICKS ==
- Popup/new window opened? browser_tabs action "list" to see all tabs. browser_tabs action "select" with the tab index to switch. ALWAYS check for new tabs after clicking login/apply/sign-in buttons.
- "Upload your resume" pre-fill page (Workday, Lever, etc.): This is NOT the application form yet. Click "Select file" or the upload area, then browser_file_upload with the resume PDF path. Wait for parsing to finish. Then click Next/Continue to reach the actual form.
- File upload not working? Try: (1) browser_click the upload button/area, (2) browser_file_upload with the path. If still failing, look for a hidden file input or a "Select file" link and click that first.
- Dropdown won't fill? browser_click to open it, then browser_click the option.
- Checkbox won't check via fill_form? Use browser_click on it instead. Snapshot to verify.
- Phone field with country prefix: just type digits {phone_digits}
- Date fields: {datetime.now().strftime('%m/%d/%Y')}
- Validation errors after submit? Take BOTH snapshot AND screenshot. Snapshot shows text errors, screenshot shows red-highlighted fields. Fix all, retry.
- Honeypot fields (hidden, "leave blank"): skip them.
- Format-sensitive fields: read the placeholder text, match it exactly.

{captcha_section}

== WHEN TO GIVE UP ==
- Same page after 3 attempts with no progress -> RESULT:FAILED:stuck
- Job is closed/expired/page says "no longer accepting" -> RESULT:EXPIRED
- Page is broken/500 error/blank -> RESULT:FAILED:page_error
Stop immediately. Output your RESULT code. Do not loop."""

    return prompt
