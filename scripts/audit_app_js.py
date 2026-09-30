import sys

def audit_js(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    print(f"Total lines: {len(lines)}, Total characters: {len(content)}")

    pos = 0
    line_no = 1
    col_no = 1
    stack = []
    n = len(content)

    while pos < n:
        ch = content[pos]

        # Handle line numbering
        if ch == '\n':
            line_no += 1
            col_no = 1
            pos += 1
            continue

        # Single line comment
        if content[pos:pos+2] == '//':
            while pos < n and content[pos] != '\n':
                pos += 1
            continue

        # Multi-line comment
        if content[pos:pos+2] == '/*':
            start_l = line_no
            pos += 2
            col_no += 2
            while pos < n and content[pos:pos+2] != '*/':
                if content[pos] == '\n':
                    line_no += 1
                    col_no = 1
                else:
                    col_no += 1
                pos += 1
            pos += 2
            col_no += 2
            continue

        # Single quoted string
        if ch == "'":
            start_l = line_no
            start_c = col_no
            pos += 1
            col_no += 1
            while pos < n:
                if content[pos] == '\\':
                    pos += 2
                    col_no += 2
                    continue
                if content[pos] == "'":
                    pos += 1
                    col_no += 1
                    break
                if content[pos] == '\n':
                    line_no += 1
                    col_no = 1
                else:
                    col_no += 1
                pos += 1
            continue

        # Double quoted string
        if ch == '"':
            start_l = line_no
            start_c = col_no
            pos += 1
            col_no += 1
            while pos < n:
                if content[pos] == '\\':
                    pos += 2
                    col_no += 2
                    continue
                if content[pos] == '"':
                    pos += 1
                    col_no += 1
                    break
                if content[pos] == '\n':
                    line_no += 1
                    col_no = 1
                else:
                    col_no += 1
                pos += 1
            continue

        # Template literal
        if ch == '`':
            start_l = line_no
            start_c = col_no
            pos += 1
            col_no += 1
            while pos < n:
                if content[pos] == '\\':
                    pos += 2
                    col_no += 2
                    continue
                if content[pos:pos+2] == '${':
                    stack.append(('${', line_no, col_no))
                    pos += 2
                    col_no += 2
                    break
                if content[pos] == '`':
                    pos += 1
                    col_no += 1
                    break
                if content[pos] == '\n':
                    line_no += 1
                    col_no = 1
                else:
                    col_no += 1
                pos += 1
            continue

        # Open brackets
        if ch in '([{':
            stack.append((ch, line_no, col_no))
            pos += 1
            col_no += 1
            continue

        # Close brackets
        if ch in ')]}':
            if not stack:
                print(f"UNEXPECTED CLOSING: '{ch}' at line {line_no}, col {col_no}")
                start_idx = max(0, line_no - 6)
                end_idx = min(len(lines), line_no + 6)
                for i in range(start_idx, end_idx):
                    prefix = ">>>" if i + 1 == line_no else "   "
                    print(f"{prefix} {i+1:4d}: {lines[i]}")
                return False

            top, top_line, top_col = stack.pop()
            matching = {'(': ')', '[': ']', '{': '}', '${': '}'}
            if matching.get(top) != ch:
                print(f"MISMATCH: Expected '{matching.get(top)}' to close '{top}' (opened at line {top_line}, col {top_col}), but found '{ch}' at line {line_no}, col {col_no}")
                start_idx = max(0, line_no - 6)
                end_idx = min(len(lines), line_no + 6)
                for i in range(start_idx, end_idx):
                    prefix = ">>>" if i + 1 == line_no else "   "
                    print(f"{prefix} {i+1:4d}: {lines[i]}")
                return False

            # If closed a '${', resume template literal
            if top == '${':
                pos += 1
                col_no += 1
                while pos < n:
                    if content[pos] == '\\':
                        pos += 2
                        col_no += 2
                        continue
                    if content[pos:pos+2] == '${':
                        stack.append(('${', line_no, col_no))
                        pos += 2
                        col_no += 2
                        break
                    if content[pos] == '`':
                        pos += 1
                        col_no += 1
                        break
                    if content[pos] == '\n':
                        line_no += 1
                        col_no = 1
                    else:
                        col_no += 1
                    pos += 1
                continue

            pos += 1
            col_no += 1
            continue

        pos += 1
        col_no += 1

    print(f"Scanning finished. Unclosed items in stack: {len(stack)}")
    if stack:
        for item, l, c in stack:
            print(f"Unclosed '{item}' opened at line {l}, col {c}:")
            print(f"   Line {l}: {lines[l-1]}")
        return False

    print("SUCCESS: All brackets, parentheses, and template literals match perfectly!")
    return True

if __name__ == '__main__':
    audit_js('frontend/static/js/app.js')
