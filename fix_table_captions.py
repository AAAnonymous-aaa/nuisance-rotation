"""Move every table caption in front of its tabular (ICLR requires the table
number and title before the table)."""

import glob
import io
import re


def find_caption(body):
    start = body.find("\\caption{")
    if start < 0:
        return None, None
    depth = 0
    for index in range(start + len("\\caption"), len(body)):
        if body[index] == "{":
            depth += 1
        elif body[index] == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    else:
        return None, None
    tail = body[end:]
    label = re.match(r"\s*\\label\{[^}]*\}", tail)
    if label:
        end += label.end()
    return start, end


def fix_block(match):
    head, body, tail = match.group(1), match.group(2), match.group(3)
    start, end = find_caption(body)
    if start is None:
        return match.group(0)
    caption = body[start:end].strip()
    rest = (body[:start] + body[end:]).strip("\n")
    lines = rest.split("\n")
    insert = 0
    while insert < len(lines) and lines[insert].strip() in (
            "\\centering", "\\small", "\\footnotesize", "\\scriptsize"):
        insert += 1
    lines = lines[:insert] + [caption] + lines[insert:]
    return head + "\n" + "\n".join(lines).rstrip() + "\n" + tail


def main():
    pattern = re.compile(r"(\\begin\{table\}(?:\[[^\]]*\])?)(.*?)(\\end\{table\})", re.S)
    changed = 0
    for path in sorted(glob.glob("paper/*.tex")):
        text = io.open(path, encoding="utf-8").read()
        new = pattern.sub(fix_block, text)
        if new != text:
            io.open(path, "w", encoding="utf-8", newline="\n").write(new)
            changed += 1
            print("fixed", path)
    print(f"{changed} files updated")


if __name__ == "__main__":
    main()
