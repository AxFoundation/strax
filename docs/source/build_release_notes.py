import os
import re

header = "# Release notes\n\n"


def convert_release_notes():
    """Write the release notes as Markdown with links to PRs."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    notes = os.path.join(this_dir, "..", "..", "HISTORY.md")
    with open(notes, "r", encoding="utf-8") as f:
        notes = f.read()

    def link_pull_request(match):
        number = match.group(1)
        return f"[#{number}](https://github.com/AxFoundation/strax/pull/{number})"

    notes = re.sub(r"(?<![\w/\[])#(\d+)", link_pull_request, notes)
    target = os.path.join(this_dir, "reference", "release_notes.md")
    with open(target, "w", encoding="utf-8") as f:
        f.write(header + notes)


if __name__ == "__main__":
    convert_release_notes()
