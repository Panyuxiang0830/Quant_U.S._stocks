"""Read compatibility for the retired template engine's historical trial ledgers.

The production search entry does not import this candidate generator.
"""
from __future__ import annotations
import ast
import itertools
import numpy as np
import pandas as pd


def compute(expression, panels, universe=None):
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("expression too large")
    def window(node):
        n = visit(node)
        if not isinstance(n, (int, float)) or not float(n).is_integer() or not 1 <= n <= 504:
            raise ValueError("window must be an integer from 1 to 504")
        return int(n)
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name):
            if node.id not in panels:
                raise ValueError(f"unknown field {node.id}")
            return panels[node.id].where(universe.eq(1)) if universe is not None else panels[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and np.isfinite(node.value):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = visit(node.operand)
            return -val if isinstance(node.op, ast.USub) else val
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(b, pd.DataFrame): b = b.where(b.abs() > 1e-12)
            elif abs(b) <= 1e-12: return np.nan
            return a / b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            op, args = node.func.id, node.args
            if op in ("rank", "abs", "sign", "log", "neg") and len(args) == 1:
                x = visit(args[0])
                if not isinstance(x, pd.DataFrame): raise ValueError("operator requires a panel")
                if op == "rank": return x.rank(axis=1, pct=True)
                if op == "abs": return x.abs()
                if op == "sign": return np.sign(x)
                if op == "log": return np.log(x.where(x > 0))
                return -x
            if op in ("ts_mean", "ts_std", "ts_min", "ts_max", "ts_rank", "ts_delta", "ts_delay", "ts_sum", "decay_linear") and len(args) == 2:
                x, n = visit(args[0]), window(args[1])
                if not isinstance(x, pd.DataFrame): raise ValueError("operator requires a panel")
                if op == "ts_delay": return x.shift(n)
                if op == "ts_delta": return x - x.shift(n)
                roll = x.rolling(n, min_periods=n)
                if op == "ts_mean": return roll.mean()
                if op == "ts_std": return roll.std()
                if op == "ts_min": return roll.min()
                if op == "ts_max": return roll.max()
                if op == "ts_sum": return roll.sum()
                if op == "ts_rank": return roll.rank(pct=True)
                w = np.arange(1, n + 1, dtype=float); w /= w.sum()
                return roll.apply(lambda a: float(np.dot(a, w)), raw=True)
            if op in ("ts_corr", "ts_cov") and len(args) == 3:
                a, b, n = visit(args[0]), visit(args[1]), window(args[2])
                if not isinstance(a, pd.DataFrame) or not isinstance(b, pd.DataFrame): raise ValueError("operator requires panels")
                return a.rolling(n, min_periods=n).corr(b) if op == "ts_corr" else a.rolling(n, min_periods=n).cov(b)
        raise ValueError("unsupported syntax/operator; attributes, imports, arbitrary calls are forbidden")
    result = visit(tree)
    if not isinstance(result, pd.DataFrame):
        raise ValueError("factor must produce a date x security panel")
    return result.replace([np.inf, -np.inf], np.nan).astype("float32")


def enumerate_expressions(fields, windows=(5, 10, 20, 60), seeds=()):
    """Finite, deterministic exhaustive grammar, including seed-structure expansion.

    Budgets are hard limits enforced by search; there is no adaptive OOS feedback.
    """
    seen = set()
    def unique(x):
        if x not in seen:
            seen.add(x)
            return True
        return False
    for s in seeds:
        if unique(s): yield s
    for f in fields:
        for n in windows:
            structures = [f"rank(ts_delta({f},{n}))", f"rank({f}/ts_mean({f},{n})-1)",
                          f"rank(ts_std({f},{n}))", f"ts_rank({f},{n})",
                          f"rank(({f}-ts_mean({f},{n}))/ts_std({f},{n}))"]
            for s in structures:
                if unique(s): yield s
    for a, b in itertools.combinations(fields, 2):
        for n in windows:
            for s in (f"rank(ts_corr({a},{b},{n}))", f"rank(ts_delta({a},{n}))*rank(ts_delta({b},{n}))"):
                if unique(s): yield s
    for seed in seeds:
        for n in windows:
            for s in (f"ts_mean(({seed}),{n})", f"rank(ts_delta(({seed}),{n}))", f"ts_rank(({seed}),{n})"):
                if unique(s): yield s
