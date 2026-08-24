#!/usr/bin/env python3
import difflib
import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from openai import OpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = (
    OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    if DEEPSEEK_API_KEY
    else None
)

SOURCES = {
    "sndsh404": "https://raw.githubusercontent.com/sndsh404/summer-2027-internships/main/README.md",
    "simplify": "https://raw.githubusercontent.com/SimplifyJobs/Summer2027-Internships/dev/README.md",
    "vanshb03": "https://raw.githubusercontent.com/vanshb03/Summer2027-Internships/dev/README.md",
}

RAW_HISTORY_PATH = "data/raw_history.json"
ARCHIVE_PATH = "data/deduplicated_archive.json"
OUTPUT_README = "README.md"
SLIDING_WINDOW_DAYS = 30

MIN_EXPECTED_ITEMS = {"sndsh404": 15, "simplify": 20, "vanshb03": 15}

LOCATION_ALIASES = {
    "nyc": "New York, NY",
    "new york city": "New York, NY",
    "new york": "New York, NY",
    "sf": "San Francisco, CA",
    "south sf": "South San Francisco, CA",
    "la": "Los Angeles, CA",
}

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
    "PR",
}

US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
    "puerto rico",
}

US_KEYWORDS = {
    "usa",
    "us",
    "united states",
    "remote",
    "remote (us)",
    "remote - us",
    "nyc",
    "sf",
    "la",
    "bay area",
    "silicon valley",
    "multiple us",
    "multiple locations",
}

NON_US_TERMS = {
    "canada",
    "ontario",
    "quebec",
    "british columbia",
    "alberta",
    "manitoba",
    "saskatchewan",
    "nova scotia",
    "toronto",
    "vancouver",
    "montreal",
    "montréal",
    "waterloo",
    "ottawa",
    "calgary",
    "edmonton",
    "richmond hill",
    "mississauga",
    "uk",
    "united kingdom",
    "london",
    "england",
    "scotland",
    "ireland",
    "dublin",
    "manchester",
    "germany",
    "berlin",
    "munich",
    "netherlands",
    "amsterdam",
    "france",
    "paris",
    "switzerland",
    "zurich",
    "sweden",
    "poland",
    "spain",
    "singapore",
    "india",
    "japan",
    "tokyo",
    "china",
    "uae",
    "united arab emirates",
    "dubai",
    "australia",
    "sydney",
    "melbourne",
    "cayman",
}

CANADIAN_PROV_CODES = {"ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE"}


def clean_url(url: str | None) -> str:
    """Normalize ATS URLs by stripping query tracking and trailing slashes."""
    if not url or url.startswith("#") or url == "🔒":
        return ""
    url = re.sub(
        r"(\?|&)(utm_[^&=]+|ref|gh_src|iis|iisn|icims|jr_id|ats|mobile|needsRedirect)=[^&=]+",
        "",
        url,
    )
    return re.sub(r"\?&", "?", url).rstrip("?&").rstrip("/")


def parse_relative_or_text_date(raw_date: str | None, fallback_date: datetime) -> str:
    """Parse relative age strings, ISO dates, or month/day text into YYYY-MM-DD."""
    if not raw_date or raw_date.strip() in ["-", "", "None"]:
        return fallback_date.strftime("%Y-%m-%d")

    raw = raw_date.strip().lower()

    if d_match := re.match(r"^(\d+)\s*d$", raw):
        return (fallback_date - timedelta(days=int(d_match.group(1)))).strftime(
            "%Y-%m-%d"
        )

    if w_match := re.match(r"^(\d+)\s*w$", raw):
        return (fallback_date - timedelta(days=int(w_match.group(1)) * 7)).strftime(
            "%Y-%m-%d"
        )

    if mo_match := re.match(r"^(\d+)\s*mo$", raw):
        return (fallback_date - timedelta(days=int(mo_match.group(1)) * 30)).strftime(
            "%Y-%m-%d"
        )

    try:
        parsed = date_parser.parse(raw_date, default=fallback_date)
        # Offset parsing forward-rollover when current date is early in the year
        if parsed.date() > fallback_date.date():
            parsed = parsed.replace(year=parsed.year - 1)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, date_parser.ParserError):
        return fallback_date.strftime("%Y-%m-%d")


def parse_emojis(text: str) -> dict[str, Any]:
    """Extract listing state flags and strip formatting metadata emojis."""
    return {
        "is_closed": "🔒" in text,
        "clean_text": re.sub(r"[🛂🇺🇸🔒🎓🔥⏳*`]", "", text).strip(),
    }


def normalize_title_for_comparison(title: str) -> str:
    """Extract core title tokens by stripping noise, terms, and degree levels."""
    t = title.lower()
    t = re.sub(
        r"\b(summer|fall|spring|winter|2026|2027|intern|internship|co-op|coop|program|undergraduate|bs|ms|phd)\b",
        "",
        t,
    )
    t = re.sub(r"[^a-z0-9]", " ", t)
    return " ".join(t.split())


def normalize_location_name(loc: str) -> str:
    cleaned = loc.strip()
    return LOCATION_ALIASES.get(cleaned.lower(), cleaned)


def generate_item_hash(item: dict[str, Any]) -> str:
    """Generate a stable deduplication fingerprint for incoming raw records."""
    link = item.get("link", "").strip()
    if link:
        return hashlib.sha256(link.encode("utf-8")).hexdigest()

    comp = item.get("company", "").strip().lower()
    role = normalize_title_for_comparison(item.get("role", ""))
    src = item.get("source", "")
    key = f"{src}::{comp}::{role}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def is_us_location(loc_str: str) -> bool:
    """Check if a location string is domestic; defaults to True on ambiguous inputs."""
    if not loc_str or not loc_str.strip():
        return True

    text = loc_str.strip()
    lower_text = text.lower()

    prov_match = re.search(r"(?:,\s*|\s+)([A-Z]{2})(?:\s+Canada|\b)", text)
    if prov_match and prov_match.group(1) in CANADIAN_PROV_CODES:
        return False

    has_non_us_term = any(
        re.search(rf"\b{re.escape(term)}\b", lower_text) for term in NON_US_TERMS
    )
    state_code_match = re.search(r"(?:,\s*|\s+)([A-Z]{2})\b", text)
    has_us_state_code = bool(
        state_code_match and state_code_match.group(1) in US_STATE_CODES
    )
    has_us_state_name = any(
        re.search(rf"\b{re.escape(st)}\b", lower_text) for st in US_STATE_NAMES
    )
    has_us_keyword = any(
        re.search(rf"\b{re.escape(kw)}\b", lower_text) for kw in US_KEYWORDS
    )

    is_explicit_us = has_us_state_code or has_us_state_name or has_us_keyword
    return not (has_non_us_term and not is_explicit_us)


def filter_us_listing(item: dict[str, Any]) -> dict[str, Any] | None:
    """Retain only domestic locations; drops item if all locations are foreign."""
    raw_locs = item.get("locations", [])
    if not raw_locs:
        return item

    valid_us_locs = [
        normalize_location_name(loc) for loc in raw_locs if is_us_location(loc)
    ]

    if not valid_us_locs:
        return None

    item["locations"] = valid_us_locs
    return item


def parse_sndsh404(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
    """Parse standard markdown table from sndsh404."""
    results = []
    match = re.search(
        r"##\s+the list.*?\n(\|.+?)(?=\n##\s+programs|\Z)",
        raw_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return results

    for line in match.group(1).strip().splitlines():
        if not line.startswith("|") or "---" in line or "Company | Role" in line:
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) < 4:
            continue

        company_raw, role_raw, loc_raw, apply_raw = cols[0], cols[1], cols[2], cols[3]
        date_raw = cols[4] if len(cols) > 4 else None

        link_match = re.search(r"\((https?://[^\)]+)\)", apply_raw)
        link = clean_url(link_match.group(1)) if link_match else ""
        flags = parse_emojis(f"{company_raw} {role_raw}")
        locations = [
            normalize_location_name(l.strip())
            for l in re.split(r"[/,]\s*(?![^()]*\))", loc_raw)
            if l.strip()
        ]

        results.append(
            {
                "company": parse_emojis(company_raw)["clean_text"],
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "is_closed": flags["is_closed"] or "🔒" in apply_raw,
                "date_posted": parse_relative_or_text_date(date_raw, current_time),
                "source": "sndsh404",
            }
        )
    return results


def parse_simplify(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
    """Parse HTML table elements from SimplifyJobs, resolving row hierarchy."""
    results = []
    soup = BeautifulSoup(raw_text, "html.parser")
    current_company = ""

    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        comp_cell, role_cell, loc_cell, app_cell, age_cell = (
            cols[0],
            cols[1],
            cols[2],
            cols[3],
            cols[4],
        )
        comp_text = comp_cell.get_text(strip=True)

        if "↳" in comp_text:
            company = current_company
        else:
            anchor = comp_cell.find("a")
            company = anchor.get_text(strip=True) if anchor else comp_text
            company = parse_emojis(company)["clean_text"]
            current_company = company

        role_raw = role_cell.get_text(" ", strip=True)
        flags = parse_emojis(role_raw)

        for tag in loc_cell.find_all(["details", "summary"]):
            tag.unwrap()
        locations = [
            normalize_location_name(l.strip())
            for l in loc_cell.stripped_strings
            if not l.lower().endswith("locations")
        ]

        link = ""
        for a in app_cell.find_all("a", href=True):
            href_attr = a.get("href")
            href = href_attr if isinstance(href_attr, str) else ""
            if not href:
                continue

            img = a.find("img")
            if img:
                alt_attr = img.get("alt")
                alt = alt_attr if isinstance(alt_attr, str) else ""
                if alt.lower() == "apply":
                    link = clean_url(href)
                    break

            if not link and "simplify.jobs/p/" not in href:
                link = clean_url(href)

        results.append(
            {
                "company": company,
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "is_closed": flags["is_closed"],
                "date_posted": parse_relative_or_text_date(
                    age_cell.get_text(strip=True), current_time
                ),
                "source": "simplify",
            }
        )
    return results


def parse_vanshb03(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
    """Parse hybrid HTML/Markdown table from vanshb03."""
    results = []
    match = re.search(r"TABLE_START.*?\n(\|.+?)(?=\n<!--|\Z)", raw_text, re.DOTALL)
    if not match:
        return results

    current_company = ""
    for line in match.group(1).strip().splitlines():
        if not line.startswith("|") or "---" in line or "Company | Role" in line:
            continue
        cols = [c.strip() for c in line.split("|")[1:-1]]
        if len(cols) < 5:
            continue

        comp_raw, role_raw, loc_raw, app_raw, date_raw = (
            cols[0],
            cols[1],
            cols[2],
            cols[3],
            cols[4],
        )

        if "↳" in comp_raw:
            company = current_company
        else:
            company = parse_emojis(comp_raw)["clean_text"]
            current_company = company

        flags = parse_emojis(role_raw)
        clean_loc = re.sub(
            r"</?(details|summary|strong|b)>", "", loc_raw, flags=re.IGNORECASE
        )
        clean_loc = re.sub(
            r"\*\*\d+\s+locations\*\*", "", clean_loc, flags=re.IGNORECASE
        )
        locations = [
            normalize_location_name(l.strip())
            for l in re.split(r"</br>|<br\s*/?>|,", clean_loc)
            if l.strip()
        ]

        link_match = re.search(r'href=["\']([^"\']+)["\']', app_raw)
        link = clean_url(link_match.group(1)) if link_match else ""

        results.append(
            {
                "company": company,
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "is_closed": "🔒" in app_raw or flags["is_closed"],
                "date_posted": parse_relative_or_text_date(date_raw, current_time),
                "source": "vanshb03",
            }
        )
    return results


def verify_source_integrity(
    source_name: str, parsed_items: list[dict[str, Any]]
) -> bool:
    """Enforce structural validation gates to catch breaking upstream schema drift."""
    expected_min = MIN_EXPECTED_ITEMS.get(source_name, 5)
    if len(parsed_items) < expected_min:
        print(
            f"Schema check failed for {source_name}: count {len(parsed_items)} < expected {expected_min}"
        )
        return False

    valid_companies = sum(1 for item in parsed_items if item.get("company"))
    valid_roles = sum(1 for item in parsed_items if item.get("role"))

    if valid_companies < expected_min * 0.8 or valid_roles < expected_min * 0.8:
        print(f"Field presence check failed for {source_name}")
        return False

    return True


def are_locations_compatible(locs1: list[str], locs2: list[str]) -> bool:
    """Determine if location arrays overlap, treating broad terms as wildcards."""
    s1 = {normalize_location_name(l).lower() for l in locs1 if l}
    s2 = {normalize_location_name(l).lower() for l in locs2 if l}
    wildcards = {"united states", "remote", "remote (us)", "us"}
    if not s1 or not s2 or bool(s1 & wildcards) or bool(s2 & wildcards):
        return True
    return bool(s1 & s2)


def merge_listing_pair(
    earlier: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Merge duplicate records, prioritizing the earlier discovery metadata."""
    merged = earlier.copy()
    if not merged.get("link") and incoming.get("link"):
        merged["link"] = incoming["link"]
    merged["locations"] = list(
        dict.fromkeys(merged.get("locations", []) + incoming.get("locations", []))
    )
    return merged


def llm_verify_duplicate(item_a: dict[str, Any], item_b: dict[str, Any]) -> bool:
    """Adjudicate ambiguous same-company listings via DeepSeek."""
    if not client:
        return False

    prompt = f"""Compare these two internship job listings at company "{item_a["company"]}".
Determine if they represent the exact same internship opportunity or distinct roles.

Listing A: Role="{item_a["role"]}", Locations={item_a["locations"]}, Link="{item_a["link"]}"
Listing B: Role="{item_b["role"]}", Locations={item_b["locations"]}, Link="{item_b["link"]}"

Rules:
- If they are different engineering tracks (e.g., Firmware vs Backend vs Infrastructure), reply false.
- If they are different locations, reply false.
- Only reply true if they are formatting variants of the same role opening.

Respond strictly with valid JSON: {{"is_same_role": true}} or {{"is_same_role": false}}"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        res = json.loads(content)
        return bool(res.get("is_same_role", False))
    except (json.JSONDecodeError, KeyError, requests.RequestException):
        return False
    except Exception:  # noqa: BLE001
        return False


def deduplicate_lenient(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate records conservatively to prevent collapsing distinct tracks/offices."""
    sorted_items = sorted(items, key=lambda x: x.get("date_posted", "9999-99-99"))
    deduped: list[dict[str, Any]] = []

    for item in sorted_items:
        match_found = False

        for idx, existing in enumerate(deduped):
            if item["company"].lower().strip() != existing["company"].lower().strip():
                continue

            if (
                item.get("link")
                and existing.get("link")
                and item["link"] == existing["link"]
            ):
                deduped[idx] = merge_listing_pair(existing, item)
                match_found = True
                break

            norm_a = normalize_title_for_comparison(existing["role"])
            norm_b = normalize_title_for_comparison(item["role"])
            sim = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

            if are_locations_compatible(existing["locations"], item["locations"]):
                if sim >= 0.92 or norm_a == norm_b:
                    deduped[idx] = merge_listing_pair(existing, item)
                    match_found = True
                    break
                if (
                    sim >= 0.78
                    and client is not None
                    and llm_verify_duplicate(existing, item)
                ):
                    deduped[idx] = merge_listing_pair(existing, item)
                    match_found = True
                    break

        if not match_found:
            deduped.append(item)

    return deduped


def generate_markdown(listings: list[dict[str, Any]], current_time: datetime) -> None:
    """Render the active 30-day sliding window into the root README.md."""
    cutoff_date = (current_time - timedelta(days=SLIDING_WINDOW_DAYS)).strftime(
        "%Y-%m-%d"
    )

    active_listings = [
        item
        for item in listings
        if item.get("date_posted", "") >= cutoff_date and not item.get("is_closed")
    ]

    active_listings.sort(
        key=lambda x: (x.get("date_posted", ""), x.get("company", "")), reverse=True
    )

    now_str = current_time.strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Summer 2027 Tech Internships\n",
        f"> *Showing active listings from the last **{SLIDING_WINDOW_DAYS} days** (since `{cutoff_date}`).*  ",
        f"> *Last updated: `{now_str}`*\n",
        "> **Note:** This repository is an automated aggregator and deduplicator. Sourcing and curation credit belongs to:",
        "> - [sndsh404/summer-2027-internships](https://github.com/sndsh404/summer-2027-internships)",
        "> - [SimplifyJobs/Summer2027-Internships](https://github.com/SimplifyJobs/Summer2027-Internships)",
        "> - [vanshb03/Summer2027-Internships](https://github.com/vanshb03/Summer2027-Internships)\n",
        "| Date Posted | Company | Job Title | Locations | Application Link |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in active_listings:
        locs = item.get("locations", ["United States"])
        loc_display = ", ".join(locs[:3])
        if len(locs) > 3:
            loc_display += f" *(+{len(locs) - 3} more)*"

        link_display = (
            f"[Apply]({item['link']})" if item.get("link") else "Check Portal"
        )
        date_display = item.get("date_posted", "-")

        lines.append(
            f"| {date_display} | **{item['company']}** | {item['role']} | {loc_display} | {link_display} |"
        )

    lines.append(f"\n*Total Active Opportunities: {len(active_listings)}*")

    with open(OUTPUT_README, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs("data", exist_ok=True)
    current_time = datetime.now(timezone.utc)

    raw_history: list[dict[str, Any]] = []
    seen_hashes: dict[str, dict[str, Any]] = {}

    if os.path.exists(RAW_HISTORY_PATH):
        with open(RAW_HISTORY_PATH, "r", encoding="utf-8") as f:
            try:
                raw_history = json.load(f)
                for item in raw_history:
                    seen_hashes[generate_item_hash(item)] = item
            except json.JSONDecodeError:
                raw_history = []

    incoming_raw: list[dict[str, Any]] = []
    for src_name, url in SOURCES.items():
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue

            items: list[dict[str, Any]] = []
            if src_name == "sndsh404":
                items = parse_sndsh404(resp.text, current_time)
            elif src_name == "simplify":
                items = parse_simplify(resp.text, current_time)
            elif src_name == "vanshb03":
                items = parse_vanshb03(resp.text, current_time)

            if verify_source_integrity(src_name, items):
                incoming_raw.extend(items)
        except requests.RequestException:
            pass
        except Exception:  # noqa: BLE001, S110
            pass

    for item in incoming_raw:
        item_hash = generate_item_hash(item)

        if item_hash in seen_hashes:
            if item.get("is_closed"):
                seen_hashes[item_hash]["is_closed"] = True
            continue

        filtered_item = filter_us_listing(item)
        if filtered_item is not None:
            raw_history.append(filtered_item)
            seen_hashes[item_hash] = filtered_item

    with open(RAW_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_history, f, indent=2)

    deduped = deduplicate_lenient(raw_history)

    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)

    generate_markdown(deduped, current_time)


if __name__ == "__main__":
    main()
