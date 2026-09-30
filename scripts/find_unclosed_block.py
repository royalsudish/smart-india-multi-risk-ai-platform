import re

def tokenize_and_track():
    with open("frontend/static/js/app.js", "r", encoding="utf-8") as f:
        src = f.read()

    lines = src.split("\n")
    stack = []
    
    pos = 0
    n = len(src)
    line = 1
    col = 1

    last_token_type = None

    while pos < n:
        ch = src[pos]

        # Line counting
        if ch == "\n":
            line += 1
            col = 1
            pos += 1
            continue

        # Whitespace
        if ch in " \t\r":
            col += 1
            pos += 1
            continue

        # Single line comment
        if ch == "/" and pos + 1 < n and src[pos+1] == "/":
            while pos < n and src[pos] != "\n":
                pos += 1
            continue

        # Multi line comment
        if ch == "/" and pos + 1 < n and src[pos+1] == "*":
            pos += 2
            col += 2
            while pos + 1 < n and not (src[pos] == "*" and src[pos+1] == "/"):
                if src[pos] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                pos += 1
            pos += 2
            col += 2
            continue

        # Single / Double quote strings
        if ch in ("'", '"'):
            quote = ch
            pos += 1
            col += 1
            while pos < n:
                if src[pos] == "\\":
                    pos += 2
                    col += 2
                    continue
                if src[pos] == quote:
                    pos += 1
                    col += 1
                    break
                if src[pos] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                pos += 1
            last_token_type = 'literal'
            continue

        # Template literal
        if ch == "`":
            pos += 1
            col += 1
            while pos < n:
                if src[pos] == "\\":
                    pos += 2
                    col += 2
                    continue
                if src[pos:pos+2] == "${":
                    # Expression inside template literal
                    stack.append(('${', line, col, lines[line-1]))
                    pos += 2
                    col += 2
                    last_token_type = '${'
                    break
                if src[pos] == "`":
                    pos += 1
                    col += 1
                    last_token_type = 'literal'
                    break
                if src[pos] == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
                pos += 1
            continue

        # Regex literal: if '/' occurs after punctuation or certain keywords
        if ch == "/":
            # Check if this '/' is division or regex
            is_regex = False
            if last_token_type in (None, 'punct', 'keyword', '${', 'open'):
                is_regex = True
            
            if is_regex:
                pos += 1
                col += 1
                in_class = False
                while pos < n:
                    if src[pos] == "\\":
                        pos += 2
                        col += 2
                        continue
                    if src[pos] == "[":
                        in_class = True
                    elif src[pos] == "]":
                        in_class = False
                    elif src[pos] == "/" and not in_class:
                        pos += 1
                        col += 1
                        # flags
                        while pos < n and src[pos] in "gimsuy":
                            pos += 1
                            col += 1
                        break
                    elif src[pos] == "\n":
                        # Unterminated regex
                        break
                    pos += 1
                    col += 1
                last_token_type = 'literal'
                continue
            else:
                last_token_type = 'punct'
                pos += 1
                col += 1
                continue

        # Open brackets
        if ch in "([{":
            stack.append((ch, line, col, lines[line-1]))
            last_token_type = 'open'
            pos += 1
            col += 1
            continue

        # Close brackets
        if ch in ")]}":
            if not stack:
                print(f"EXTRA CLOSING BRACKET '{ch}' at line {line}, col {col}")
                pos += 1
                col += 1
                continue
            
            top_type, top_line, top_col, top_src = stack.pop()
            match = {'(': ')', '[': ']', '{': '}', '${': '}'}
            if match.get(top_type) != ch:
                print(f"MISMATCH: Expected {match.get(top_type)} for {top_type} from L{top_line}:C{top_col}, but found '{ch}' at L{line}:C{col}")
                print(f"  Opener line: {top_src.strip()}")
                print(f"  Closer line: {lines[line-1].strip()}")
            
            if top_type == '${':
                # Resume template literal
                while pos < n:
                    if src[pos] == "\\":
                        pos += 2
                        col += 2
                        continue
                    if src[pos:pos+2] == "${":
                        stack.append(('${', line, col, lines[line-1]))
                        pos += 2
                        col += 2
                        last_token_type = '${'
                        break
                    if src[pos] == "`":
                        pos += 1
                        col += 1
                        last_token_type = 'literal'
                        break
                    if src[pos] == "\n":
                        line += 1
                        col = 1
                    else:
                        col += 1
                    pos += 1
                continue

            last_token_type = 'close'
            pos += 1
            col += 1
            continue

        # Other punctuation
        if ch in "=;:,+-*&|^!<>?~%":
            last_token_type = 'punct'
            pos += 1
            col += 1
            continue

        # Identifiers / keywords / numbers
        if ch.isalnum() or ch in "_$":
            start_pos = pos
            while pos < n and (src[pos].isalnum() or src[pos] in "_$"):
                pos += 1
                col += 1
            word = src[start_pos:pos]
            keywords = {"return", "case", "typeof", "delete", "void", "throw", "instanceof", "in", "yield", "await", "else", "if", "for", "while", "do"}
            if word in keywords:
                last_token_type = 'keyword'
            else:
                last_token_type = 'ident'
            continue

        pos += 1
        col += 1

    print(f"\nFinished parsing. Open stack depth: {len(stack)}")
    for item, l, c, line_text in stack:
        print(f"Unclosed '{item}' opened at line {l}, col {c}: {line_text.strip()[:80]}")

if __name__ == "__main__":
    tokenize_and_track()
