"""
Program-of-Thought (PoT) helper: let the reader emit an arithmetic expression, then
we execute it exactly (LLMs reason well but compute badly).

Flow: the model reasons, then on the last line writes  ANSWER = <expression>.
We safely evaluate the expression (numbers and + - * / ( ) only) with an AST walker
(no eval of arbitrary code). Non-numeric answers (span questions) pass through as text.
"""
import ast
import operator

POT_SYSTEM = (
    "You are a table QA assistant. Read the given text and table and answer the question. "
    "If the question requires calculation, reason briefly, then on the LAST line output "
    "'ANSWER = ' followed by a SINGLE arithmetic expression using only numbers and the "
    "operators + - * / and parentheses (no words, no %, no units, no commas). "
    "If no calculation is needed, output 'ANSWER = ' followed by the final short answer."
)

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos,
    ast.Pow: operator.pow,
}


def safe_eval(expr):
    """Evaluate a pure arithmetic expression. Raises on anything non-arithmetic."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("non-arithmetic")
    return ev(ast.parse(expr, mode="eval"))


def parse_pot(text):
    """
    Extract the model's final answer. If it's an arithmetic expression, execute it
    and return the exact numeric result (as a string); otherwise return the text.
    """
    if "ANSWER" in text:
        tail = text[text.rfind("ANSWER"):]
        rhs = tail.split("=", 1)[1] if "=" in tail else tail
    else:
        rhs = text
    line = rhs.strip().splitlines()[0].strip() if rhs.strip() else ""
    cleaned = line.replace(",", "").replace("%", "").replace("$", "").strip()
    try:
        val = safe_eval(cleaned)
        # tidy: drop trailing .0 for whole numbers
        return str(int(val)) if float(val).is_integer() else str(val)
    except Exception:
        return line
