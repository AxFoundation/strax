from m2r import convert
import os

header = """
Release notes
==============

"""


def convert_release_notes():
    """Convert the release notes to an RST page with links to PRs"""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    notes = os.path.join(this_dir, '..', '..', 'HISTORY.md')
    with open(notes, 'r') as f:
        notes = f.read()
    rst = convert(notes)
    with_ref = ''
    for line in rst.split('\n'):
        # Get URL for PR
        if '#' in line:
            pr_number = line.split('#')[1]
            while len(pr_number):
                try:
                    pr_number = int(pr_number)
                    break
                except ValueError:
                    # Too many tailing characters to be an int
                    pr_number = pr_number[:-1]
            if pr_number:
                line = line.replace(f'#{pr_number}',
                                    f'`#{pr_number} <https://github.com/AxFoundation/strax/pull/{pr_number}>`_'
                                    )
        with_ref += line + '\n'
    target = os.path.join(this_dir, 'reference', 'release_notes.rst')

    with open(target, 'w') as f:
        f.write(header+with_ref)


if __name__ == '__main__':
    convert_release_notes()
