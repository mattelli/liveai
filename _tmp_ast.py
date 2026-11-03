import ast
src = open(r'C:\Users\matte\Documents\PYTHON\live_assistant\src\live_ai.py', encoding='utf-8').read()
tree = ast.parse(src)
for n in tree.body:
    if isinstance(n, ast.ClassDef):
        print(f'== {n.name} (L{n.lineno}) ==')
        for sub in n.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(sub) or ''
                print(f'  L{sub.lineno}: {sub.name} -- {doc.splitlines()[0][:100] if doc else ""}')
        print()
