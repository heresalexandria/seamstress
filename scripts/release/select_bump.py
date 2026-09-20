"""One shared rule for PR checks and release preparation."""
from __future__ import annotations

import json
import os

LABELS = {'major', 'minor', 'patch', 'no-release'}


def select(labels: list[str]) -> str:
    matches = set(labels) & LABELS
    if len(matches) != 1:
        raise ValueError('Apply exactly one release label: major, minor, patch, or no-release.')
    selected = matches.pop()
    return 'none' if selected == 'no-release' else selected


if __name__ == '__main__':
    try:
        value = json.loads(os.environ.get('LABELS', '[]'))
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError('LABELS must be a JSON array of label names.')
        print(select(value))
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc
