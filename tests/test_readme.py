import re
import pytest
import os

def generate_anchor(text):
    """
    Generates a GitHub-style anchor ID from header text.
    """
    # Lowercase
    anchor = text.lower()

    # GitHub's rule (simplified):
    # 1. Lowercase.
    # 2. Keep alphanumeric, spaces, hyphens, and underscores.
    # 3. Keep Variation Selector-16 (\ufe0f) as observed in this repo.
    # 4. Remove everything else.
    # 5. Replace spaces with hyphens.

    result = ""
    for c in anchor:
        if c.isalnum() or c in " -_":
            result += c
        elif c == "\ufe0f":
            result += c

    # Replace spaces with hyphens
    result = result.replace(' ', '-')

    return result

def get_expected_anchors(content):
    """
    Parses content for headers and generates the expected anchor for each.
    Handles duplicates by appending -1, -2, etc.
    """
    # Remove code blocks and HTML comments
    content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content_no_code = re.sub(r'<!--.*?-->', '', content_no_code, flags=re.DOTALL)

    # Standard GFM headers: # Header
    header_matches = re.findall(r'^(#+)\s+(.*)', content_no_code, re.MULTILINE)

    anchors_count = {}
    expected_anchors = {}

    for _, text in header_matches:
        base_anchor = generate_anchor(text.strip())
        anchor = base_anchor
        if anchor in anchors_count:
            anchors_count[anchor] += 1
            anchor = f"{anchor}-{anchors_count[anchor]}"
        else:
            anchors_count[anchor] = 0
        expected_anchors[anchor] = text.strip()

    return expected_anchors

def get_internal_links(content):
    """
    Finds all internal links in the content.
    Returns a list of (link_text, anchor_id, line_number).
    """
    links = []
    lines = content.split('\n')
    # Improved regex for internal links:
    # Matches [text](#anchor) where anchor doesn't contain spaces or closing paren
    # Handles optional title: [text](#anchor "title")
    link_pattern = re.compile(r'\[([^\]]+)\]\(#([^\s\)]+)(?:\s+"[^"]*")?\)')

    for i, line in enumerate(lines):
        matches = link_pattern.findall(line)
        for text, anchor in matches:
            links.append((text, anchor, i + 1))
    return links

def test_readme_internal_links():
    # Use absolute path to README.md relative to this test file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    readme_path = os.path.join(base_dir, 'README.md')

    if not os.path.exists(readme_path):
        pytest.skip(f"README.md not found at {readme_path}")

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    expected_anchors_dict = get_expected_anchors(content)
    expected_anchors = set(expected_anchors_dict.keys())
    internal_links = get_internal_links(content)

    broken_links = []
    for text, anchor, line_num in internal_links:
        if anchor not in expected_anchors:
            broken_links.append(f"Line {line_num}: [{text}](#{anchor})")

    if broken_links:
        pytest.fail(f"Found broken internal links in README.md:\n" + "\n".join(broken_links))

if __name__ == "__main__":
    # If run directly, look for README.md in current or parent directory
    readme_path = 'README.md'
    if not os.path.exists(readme_path):
        readme_path = '../README.md'

    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        expected = get_expected_anchors(content)
        links = get_internal_links(content)

        print(f"Total unique header anchors: {len(expected)}")
        print(f"Total internal links: {len(links)}")

        broken = [l for l in links if l[1] not in expected]
        if broken:
            print(f"\n{len(broken)} broken links found:")
            for text, anchor, line in broken:
                print(f"Line {line}: [{text}](#{anchor})")
        else:
            print("\nNo broken links found!")
    else:
        print("README.md not found!")
