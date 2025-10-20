"""Safe one-line expression evaluator for operator parameters.

This module provides a restricted Python eval for evaluating simple one-line
expressions with controlled builtins and locals. It is used for eval1 parameters
in operators (e.g., "eval:str(3)" or "eval:s[:100]").
"""


# Safe subset of builtins allowed in eval1 expressions
SAFE_BUILTINS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "slice": slice,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "sorted": sorted,
    "reversed": reversed,
    "any": any,
    "all": all,
}


def safe_eval_one_line(expr: str, s: str = "", p: dict | None = None) -> str:
    """Safely evaluate a one-line Python expression with restricted builtins.

    Args:
        expr: The Python expression to evaluate (must be a single line)
        s: The input text available as variable 's' in the expression
        p: Parameters dict available as variable 'p' in the expression

    Returns:
        The result of the evaluation converted to a string

    Raises:
        ValueError: If the expression is invalid or contains forbidden operations

    Examples:
        >>> safe_eval_one_line("s[:10]", s="Hello, world!")
        'Hello, wor'
        >>> safe_eval_one_line("str(3)")
        '3'
        >>> safe_eval_one_line("s.upper()", s="hello")
        'HELLO'
        >>> safe_eval_one_line("len(s)", s="test")
        '4'
    """
    if p is None:
        p = {}

    # Check for newlines (must be one line)
    if "\n" in expr:
        raise ValueError("eval1 expressions must be a single line (no newlines)")

    # Check for forbidden patterns
    forbidden = ["__import__", "import", "exec", "eval", "compile", "open",
                 "file", "__", "globals", "locals", "vars", "dir"]
    expr_lower = expr.lower()
    for pattern in forbidden:
        if pattern in expr_lower:
            raise ValueError(f"Forbidden pattern '{pattern}' in eval1 expression")

    # Set up restricted environment
    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
    }
    safe_locals = {
        "s": s,
        "p": p,
    }

    try:
        result = eval(expr, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expr}': {e}")


def resolve_eval_param(value: str, input_text: str = "", params: dict | None = None) -> str:
    """Resolve a parameter value, evaluating it if it starts with 'eval:'.

    Args:
        value: The parameter value (may be "eval:expression" or a literal string)
        input_text: The input text to make available as 's' in eval expressions
        params: The parameters dict to make available as 'p' in eval expressions

    Returns:
        The resolved value as a string. If value starts with "eval:", evaluates
        the expression and returns the result. Otherwise returns value as-is.
        If evaluation fails, returns the literal string (minus "eval:" prefix)
        with a warning.

    Examples:
        >>> resolve_eval_param("hello")
        'hello'
        >>> resolve_eval_param("eval:str(3)")
        '3'
        >>> resolve_eval_param("eval:s[:5]", input_text="Hello, world!")
        'Hello'
    """
    if not isinstance(value, str):
        return str(value)

    if not value.startswith("eval:"):
        return value

    expr = value[5:]  # Strip "eval:" prefix

    try:
        return safe_eval_one_line(expr, s=input_text, p=params or {})
    except Exception as e:
        # Fallback: treat as literal string with warning
        print(f"Warning: eval1 expression failed: {e}. Using literal value: {expr}")
        return expr
