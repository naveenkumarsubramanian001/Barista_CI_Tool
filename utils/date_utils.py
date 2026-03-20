from datetime import datetime, timedelta
import re


def is_within_range(date_str: str, days: int) -> bool:
    if not date_str:
        return False

    # Try multiple formats
    formats = [
        "%Y-%m-%d",  # 2026-02-28
        "%a, %d %b %Y %H:%M:%S %Z",  # Tue, 17 Feb 2026 03:11:49 GMT
        "%d %b %Y",  # 17 Feb 2026
        "%Y-%m-%dT%H:%M:%SZ",  # ISO
    ]

    found_date = None

    # regex for YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        try:
            found_date = datetime.strptime(match.group(1), "%Y-%m-%d")
        except Exception:
            pass

    if not found_date:
        for fmt in formats:
            try:
                found_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue

    if not found_date:
        # Fallback for "17 Feb 2026" type strings inside text
        m = re.search(r"(\d{1,2} [A-Z][a-z]{2} \d{4})", date_str)
        if m:
            try:
                found_date = datetime.strptime(m.group(1), "%d %b %Y")
            except Exception:
                pass

    if not found_date:
        return False

    now = datetime.now()
    range_ago = now - timedelta(days=days)

    # Buffer for timezones/delays
    return range_ago <= found_date <= (now + timedelta(days=1))


def is_within_last_14_days(date_str: str) -> bool:
    return is_within_range(date_str, 14)


def get_current_date_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")
