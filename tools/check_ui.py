"""Check that the page and its scripts agree about element ids and classes.

A typo in an id is invisible until the feature is clicked, so this walks the
template and the scripts and reports anything one side references and the other
does not provide.

    python tools/check_ui.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "index.html"
SCRIPTS = [ROOT / "static" / "app.js", ROOT / "static" / "zones.js"]
STYLES = ROOT / "static" / "styles.css"


def main() -> int:
    html = TEMPLATE.read_text(encoding="utf-8")
    scripts = "\n".join(path.read_text(encoding="utf-8") for path in SCRIPTS)
    css = STYLES.read_text(encoding="utf-8")

    html_ids = set(re.findall(r'\bid="([\w-]+)"', html))
    used_ids = set(re.findall(r'\$\("([\w-]+)"\)', scripts))
    used_ids |= set(re.findall(r'getElementById\("([\w-]+)"\)', scripts))

    problems = []

    missing = sorted(used_ids - html_ids)
    if missing:
        problems.append(f"referenced by a script but absent from the page: {missing}")

    duplicates = [i for i in html_ids if html.count(f'id="{i}"') > 1]
    if duplicates:
        problems.append(f"duplicate ids in the page: {sorted(duplicates)}")

    # Classes the scripts rely on for querying must exist in the markup.
    for selector in (".tab-btn", ".tab-content", "[data-zone-mode]", "[data-zone-action]"):
        token = selector.strip(".[]").split("=")[0]
        if token not in html:
            problems.append(f"selector {selector} is used by a script but not in the page")

    # Icons referenced with <use href="#..."> must be defined in the sprite.
    symbols = set(re.findall(r'<symbol id="([\w-]+)"', html))
    referenced = set(re.findall(r'<use href="#([\w-]+)"', html + scripts))
    unknown_icons = sorted(referenced - symbols)
    if unknown_icons:
        problems.append(f"icons used but not defined in the sprite: {unknown_icons}")

    # Every class used in the markup should be styled (a safety net for typos).
    html_classes = set()
    for value in re.findall(r'class="([^"]+)"', html):
        html_classes.update(value.split())
    unstyled = sorted(
        cls for cls in html_classes
        if f".{cls}" not in css and cls not in {"num", "label", "spacer", "bar", "dot"}
    )
    if unstyled:
        problems.append(f"classes in the page with no rule in the stylesheet: {unstyled}")

    if problems:
        print("UI consistency problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        f"UI consistent: {len(html_ids)} ids in the page, {len(used_ids)} used by scripts, "
        f"{len(symbols)} icons defined."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
