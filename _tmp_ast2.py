import ast
src = open(r'_tmp_smol\x\smolagents\agents.py', encoding='utf-8').read()
print('LINES:', len(src.splitlines()))
t = ast.parse(src)
for n in t.body:
    if isinstance(n, ast.ClassDef):
        methods = [s.name for s in n.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
        print(f'{n.name}: {len(methods)} methods')
        for s in n.body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                print('   ', s.name)
        print()
