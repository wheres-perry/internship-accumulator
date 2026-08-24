#!/usr/bin/env python3
import difflib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from openai import OpenAI

# =====================================================================
# Configuration & Constants
# =====================================================================

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

# Minimum row count to ensure upstream tables haven't changed format/structure
MIN_EXPECTED_ITEMS = {"sndsh404": 15, "simplify": 20, "vanshb03": 15}


# =====================================================================
# Normalization & Date Utilities
# =====================================================================


def clean_url(url: str | None) -> str:
    """Strips UTM parameters, tracking IDs, and trailing slashes for canonical matching."""
    if not url or url.startswith("#") or url == "🔒":
        return ""
    url = re.sub(
        r"(\?|&)(utm_[^&=]+|ref|gh_src|iis|iisn|icims|jr_id|ats|mobile|needsRedirect)=[^&=]+",
        "",
        url,
    )
    url = re.sub(r"\?&", "?", url).rstrip("?&").rstrip("/")
    return url


def parse_relative_or_text_date(raw_date: str | None, fallback_date: datetime) -> str:
    """Parses ISO dates, textual dates ('Aug 21'), and relative age tags ('0d', '2w', '1mo')."""
    if not raw_date or raw_date.strip() in ["-", "", "None"]:
        return fallback_date.strftime("%Y-%m-%d")

    raw = raw_date.strip().lower()

    # 1. Match relative formats: 0d, 4d, 2w, 1mo
    d_match = re.match(r"^(\d+)\s*d$", raw)
    if d_match:
        days = int(d_match.group(1))
        return (fallback_date - timedelta(days=days)).strftime("%Y-%m-%d")

    w_match = re.match(r"^(\d+)\s*w$", raw)
    if w_match:
        weeks = int(w_match.group(1))
        return (fallback_date - timedelta(days=weeks * 7)).strftime("%Y-%m-%d")

    mo_match = re.match(r"^(\d+)\s*mo$", raw)
    if mo_match:
        months = int(mo_match.group(1))
        return (fallback_date - timedelta(days=months * 30)).strftime("%Y-%m-%d")

    # 2. ISO or textual formats (e.g., '2026-07-21', 'Aug 21')
    try:
        parsed = date_parser.parse(raw_date, default=fallback_date)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, date_parser.ParserError):
        return fallback_date.strftime("%Y-%m-%d")


def parse_emojis(text: str) -> dict[str, Any]:
    """Extracts status and eligibility flags encoded as emojis."""
    return {
        "sponsorship": False if "🛂" in text else None,
        "requires_us_citizenship": "🇺🇸" in text,
        "is_closed": "🔒" in text,
        "clean_text": re.sub(r"[🛂🇺🇸🔒🎓🔥⏳*`]", "", text).strip(),
    }


def normalize_title_for_comparison(title: str) -> str:
    """Normalizes title strings to compare identical roles across repos."""
    t = title.lower()
    # Strip common noise keywords
    t = re.sub(
        r"\b(summer|fall|spring|winter|2026|2027|intern|internship|co-op|coop|program|undergraduate|bs|ms|phd)\b",
        "",
        t,
    )
    t = re.sub(r"[^a-z0-9]", " ", t)
    return " ".join(t.split())


# =====================================================================
# Upstream Parsers
# =====================================================================


def parse_sndsh404(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
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
            l.strip() for l in re.split(r"[/,]\s*(?![^()]*\))", loc_raw) if l.strip()
        ]

        results.append(
            {
                "company": parse_emojis(company_raw)["clean_text"],
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "sponsorship": flags["sponsorship"],
                "requires_us_citizenship": flags["requires_us_citizenship"],
                "is_closed": flags["is_closed"] or "🔒" in apply_raw,
                "date_posted": parse_relative_or_text_date(date_raw, current_time),
                "source": "sndsh404",
            }
        )
    return results


def parse_simplify(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
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
            l.strip()
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

        age_text = age_cell.get_text(strip=True)

        results.append(
            {
                "company": company,
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "sponsorship": flags["sponsorship"],
                "requires_us_citizenship": flags["requires_us_citizenship"],
                "is_closed": flags["is_closed"],
                "date_posted": parse_relative_or_text_date(age_text, current_time),
                "source": "simplify",
            }
        )
    return results


def parse_vanshb03(raw_text: str, current_time: datetime) -> list[dict[str, Any]]:
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
            l.strip() for l in re.split(r"</br>|<br\s*/?>|,", clean_loc) if l.strip()
        ]

        link_match = re.search(r'href=["\']([^"\']+)["\']', app_raw)
        link = clean_url(link_match.group(1)) if link_match else ""

        results.append(
            {
                "company": company,
                "role": flags["clean_text"],
                "locations": locations if locations else ["United States"],
                "link": link,
                "sponsorship": flags["sponsorship"],
                "requires_us_citizenship": flags["requires_us_citizenship"],
                "is_closed": "🔒" in app_raw or flags["is_closed"],
                "date_posted": parse_relative_or_text_date(date_raw, current_time),
                "source": "vanshb03",
            }
        )
    return results


# =====================================================================
# Integrity Verification Guard
# =====================================================================


def verify_source_integrity(
    source_name: str, parsed_items: list[dict[str, Any]]
) -> bool:
    """Verifies that the scraper parsed expected structure and hasn't silently broken."""
    expected_min = MIN_EXPECTED_ITEMS.get(source_name, 5)
    if len(parsed_items) < expected_min:
        print(
            f"⚠️ INTEGRITY FAILURE for '{source_name}': Expected >= {expected_min} items, got {len(parsed_items)}."
        )
        return False

    valid_companies = sum(1 for item in parsed_items if item.get("company"))
    valid_roles = sum(1 for item in parsed_items if item.get("role"))

    if valid_companies < expected_min * 0.8 or valid_roles < expected_min * 0.8:
        print(
            f"⚠️ SCHEMA FAILURE for '{source_name}': High proportion of missing fields."
        )
        return False

    print(
        f"✅ Integrity verified for '{source_name}': {len(parsed_items)} valid listings parsed."
    )
    return True


# =====================================================================
# Lenient Deduplication Engine
# =====================================================================


def are_locations_compatible(locs1: list[str], locs2: list[str]) -> bool:
    """Ensures we never combine different physical locations."""
    s1 = {l.lower().strip() for l in locs1 if l}
    s2 = {l.lower().strip() for l in locs2 if l}
    if not s1 or not s2:
        return True
    return bool(s1.intersection(s2))


def merge_listing_pair(
    earlier: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    """Merges two entries representing the exact same role, strictly preferring the earlier entry."""
    merged = earlier.copy()
    if not merged.get("link") and incoming.get("link"):
        merged["link"] = incoming["link"]
    if incoming.get("sponsorship") is False:
        merged["sponsorship"] = False
    if incoming.get("requires_us_citizenship"):
        merged["requires_us_citizenship"] = True
    merged["locations"] = list(
        dict.fromkeys(merged.get("locations", []) + incoming.get("locations", []))
    )
    return merged


def llm_verify_duplicate(item_a: dict[str, Any], item_b: dict[str, Any]) -> bool:
    """Calls DeepSeek only for edge cases where pattern match title ratio is borderline."""
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
    except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
        print(f"LLM dedupe fallback error: {e}")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"Unexpected LLM dedupe fallback error: {e}")
        return False


def deduplicate_lenient(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicates items with strict rules: never merge distinct roles or locations."""
    sorted_items = sorted(items, key=lambda x: x.get("date_posted", "9999-99-99"))
    deduped: list[dict[str, Any]] = []

    for item in sorted_items:
        match_found = False

        for idx, existing in enumerate(deduped):
            if item["company"].lower().strip() != existing["company"].lower().strip():
                continue

            # Direct ATS Link match
            if (
                item.get("link")
                and existing.get("link")
                and item["link"] == existing["link"]
            ):
                deduped[idx] = merge_listing_pair(existing, item)
                match_found = True
                break

            # Normalized role + compatible location
            norm_a = normalize_title_for_comparison(existing["role"])
            norm_b = normalize_title_for_comparison(item["role"])
            sim = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

            if are_locations_compatible(existing["locations"], item["locations"]):
                if sim >= 0.92 or norm_a == norm_b:
                    deduped[idx] = merge_listing_pair(existing, item)
                    match_found = True
                    break
                elif sim >= 0.78 and client is not None:
                    if llm_verify_duplicate(existing, item):
                        deduped[idx] = merge_listing_pair(existing, item)
                        match_found = True
                        break

        if not match_found:
            deduped.append(item)

    return deduped


# =====================================================================
# Sliding Window & Output
# =====================================================================


def generate_markdown(listings: list[dict[str, Any]], current_time: datetime):
    """Writes the active 30-day sliding window listings to README.md."""
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
        "> *Auto-scraped and deduplicated daily across community repositories.*  ",
        f"> *Showing active listings from the last **{SLIDING_WINDOW_DAYS} days** (since `{cutoff_date}`).*  ",
        f"> *Last updated: `{now_str}`*\n",
        "### Legend",
        "- 🛂 Does **not** offer visa sponsorship",
        "- 🇺🇸 Requires US Citizenship / Clearance",
        "- 🔒 Closed\n",
        "| Date Posted | Company | Job Title | Locations | Application Link |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for item in active_listings:
        badges = []
        if item.get("sponsorship") is False:
            badges.append("🛂")
        if item.get("requires_us_citizenship"):
            badges.append("🇺🇸")

        badge_str = f" {' '.join(badges)}" if badges else ""
        role_display = f"{item['role']}{badge_str}"

        locs = item.get("locations", ["United States"])
        loc_display = ", ".join(locs[:3])
        if len(locs) > 3:
            loc_display += f" *(+{len(locs) - 3} more)*"

        link_display = (
            f"[Apply]({item['link']})" if item.get("link") else "Check Portal"
        )
        date_display = item.get("date_posted", "-")

        lines.append(
            f"| {date_display} | **{item['company']}** | {role_display} | {loc_display} | {link_display} |"
        )

    lines.append(f"\n*Total Active Opportunities: {len(active_listings)}*")

    with open(OUTPUT_README, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =====================================================================
# Main Execution Pipeline
# =====================================================================


def main():
    os.makedirs("data", exist_ok=True)
    current_time = datetime.now(timezone.utc)

    # 1. Load historical raw data
    raw_history: list[dict[str, Any]] = []
    if os.path.exists(RAW_HISTORY_PATH):
        with open(RAW_HISTORY_PATH, "r", encoding="utf-8") as f:
            try:
                raw_history = json.load(f)
            except json.JSONDecodeError:
                raw_history = []

    # 2. Fetch and parse each upstream source
    incoming_raw: list[dict[str, Any]] = []
    for src_name, url in SOURCES.items():
        print(f"Fetching {src_name}...")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"⚠️ Failed to fetch {src_name}: HTTP {resp.status_code}")
                continue

            text = resp.text
            items = []
            if src_name == "sndsh404":
                items = parse_sndsh404(text, current_time)
            elif src_name == "simplify":
                items = parse_simplify(text, current_time)
            elif src_name == "vanshb03":
                items = parse_vanshb03(text, current_time)

            if verify_source_integrity(src_name, items):
                incoming_raw.extend(items)
        except requests.RequestException as e:
            print(f"⚠️ Network error parsing {src_name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Exception parsing {src_name}: {e}")

    # 3. Append to persistent raw history
    raw_history.extend(incoming_raw)
    with open(RAW_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_history, f, indent=2)

    # 4. Run lenient deduplication across full dataset
    print(f"Deduplicating {len(raw_history)} historical raw items...")
    deduped = deduplicate_lenient(raw_history)
    print(f"Resulted in {len(deduped)} canonical unique opportunities.")

    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)

    # 5. Generate sliding window README
    generate_markdown(deduped, current_time)
    print("README.md successfully updated.")


if __name__ == "__main__":
    main()
