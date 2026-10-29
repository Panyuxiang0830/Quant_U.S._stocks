"""Data and expression adapters for the recursive FactorExpressionGenerator.

Candidate generation and numerical operators live in engine/; this adapter does
not replace them with fixed templates or change their rolling/ranking semantics.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .engine import FactorExpressionGenerator, Operators

ENGINE_VERSION = "recursive_factor_engine_v1"
DIMENSIONS = {"open": "price", "high": "price", "low": "price", "close": "price",
              "vwap": "price", "pre_close": "price", "volume": "volume", "vol": "volume",
              "shares_out": "volume", "total_share": "volume", "amount": "amount",
              "market_cap": "amount", "book_equity": "amount", "earnings_ttm": "amount",
              "revenue_ttm": "amount", "total_assets": "amount", "total_liabilities": "amount",
              "returns": "percentage", "pct_chg": "percentage", "turnover_rate": "percentage"}
OPS = Operators().get_available_operators()
ALIASES = {"ts_mean": "mean", "ts_std": "std", "ts_min": "min", "ts_max": "max",
           "ts_sum": "sum", "ts_delta": "delta", "ts_delay": "lag", "ts_corr": "correlation"}
TIME_OPS = {"mean", "std", "max", "min", "sum", "median", "skew", "kurt",
            "pct_change", "lag", "delta", "correlation", "beta", "alpha"}


def expression_tree(expression):
    """Accept tuple/JSON trees or safe call notation, never eval a factor string."""
    if isinstance(expression, (list, tuple)):
        return tuple(expression_tree(x) if isinstance(x, (list, tuple)) else x for x in expression)
    if not isinstance(expression, str): return expression
    if expression.startswith(("[", '"')):
        return expression_tree(json.loads(expression))
    def visit(node):
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = visit(node.operand)
            return -v if isinstance(v, (int, float)) else ("*", -1, v)
        if isinstance(node, ast.BinOp):
            names = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "pow"}
            if type(node.op) in names: return (names[type(node.op)], visit(node.left), visit(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            op = ALIASES.get(node.func.id, node.func.id)
            if op in OPS: return (op, *(visit(x) for x in node.args))
        raise ValueError("unsupported expression syntax")
    return visit(ast.parse(expression, mode="eval").body)


def compute(expression, panels, universe=None):
    """The original recursive calculator's operand dispatch and original Operators."""
    tree = expression_tree(expression)
    nodes = 0
    def visit(node, depth=0):
        nonlocal nodes
        nodes += 1
        if nodes > 2000 or depth > 50: raise ValueError("expression too large")
        if isinstance(node, (int, float)):
            if not np.isfinite(node): raise ValueError("non-finite constant")
            return node
        if isinstance(node, str):
            if node in panels:
                return panels[node].where(universe.eq(1)) if universe is not None else panels[node]
            try: return float(node)
            except ValueError: raise ValueError(f"unknown field: {node}") from None
        if not isinstance(node, tuple) or not node or node[0] not in OPS:
            raise ValueError("unknown operator or invalid tree")
        op = node[0]
        expected = OPS[op].__code__.co_argcount - 1
        if len(node) - 1 != expected: raise ValueError(f"{op} expects {expected} operands")
        if op in TIME_OPS:
            w = node[-1]
            if not isinstance(w, int) or not 1 <= w <= 504:
                raise ValueError("time windows must be positive integers <=504")
        return OPS[op](*(visit(x, depth + 1) for x in node[1:]))
    result = visit(tree)
    if not isinstance(result, pd.DataFrame): raise ValueError("factor must be a panel")
    return result


def canonical(expression):
    return json.dumps(expression_tree(expression), separators=(",", ":"), ensure_ascii=False)


def engine_hashes():
    root = Path(__file__).parent / "engine"
    return {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
            for n in ("expression_generator.py", "operators.py")}


def load_panels(store, reference, fields, seeds=(), seed_factor_dir=None, seed_factors=(),
                additional_field_dimensions=None):
    panels = {}
    dimensions = {**DIMENSIONS, **(additional_field_dimensions or {})}
    for name in fields:
        if name in ("returns", "pct_chg"):
            close = store.load("close_adj", reference)
            panels[name] = close / close.shift(1) - 1
        elif name == "turnover_rate" and not store.path(name).exists():
            panels[name] = store.load("volume", reference) / store.load("shares_out", reference).replace(0, np.nan)
        else:
            source = {"vol": "volume", "total_share": "shares_out"}.get(name, name)
            if source in ("open", "high", "low", "close", "vwap") and store.path(f"{source}_adj").exists():
                source += "_adj"
            panels[name] = store.load(source, reference)
        if name not in dimensions: raise ValueError(f"declare the dimension of field {name}")
    mask = store.load("universe", reference).eq(1)
    for idx, seed in enumerate(seeds):
        item = seed if isinstance(seed, dict) else {"name": f"seed_{idx:03d}", "expression": seed}
        name = item["name"]
        if name in panels: raise ValueError(f"duplicate seed field {name}")
        panels[name] = compute(item["expression"], panels, mask)
        # Rank/zscore seeds have a defined unitless output. Other structures need
        # an explicit declared dimension; never silently label unknowns unitless.
        tree = expression_tree(item["expression"])
        dim = item.get("dimension") or ("unitless" if isinstance(tree, tuple) and tree[0] in ("rank", "zscore") else None)
        if dim is None: raise ValueError(f"declare the dimension of seed {name}")
        dimensions[name] = dim
    if seed_factors:
        if seed_factor_dir is None: raise ValueError("seed_factor_dir required")
        from usq.data import PanelStore
        fs = PanelStore(seed_factor_dir)
        for item in seed_factors:
            if not isinstance(item, dict) or "dimension" not in item:
                raise ValueError("seed_factors entries need name and dimension")
            name = item["name"]
            if name in panels: raise ValueError(f"duplicate seed field {name}")
            panels[name] = fs.load(name, reference)
            dimensions[name] = item["dimension"]
    return panels, {name: dimensions[name] for name in panels}


def generate_candidates(fields, dimensions, windows=(5, 10, 20, 60), time_periods=(1, 5, 10),
                        complexity=3, dimension_filter=("unitless", "percentage"),
                        max_trials=5000, seed=2964, operators=None, required_fields=()):
    """Use the original generator, filters and ordered final max_count selection."""
    if complexity not in (1, 2, 3): raise ValueError("complexity must be 1, 2 or 3")
    if not dimension_filter: raise ValueError("dimension_filter must be explicit")
    for w in (*windows, *time_periods):
        if not isinstance(w, int) or not 1 <= w <= 504: raise ValueError("invalid time parameter")
    if operators is not None and not set(operators).issubset(OPS): raise ValueError("unknown operators")
    if not set(required_fields).issubset(fields): raise ValueError("required field absent from inputs")
    def references(tree):
        if isinstance(tree, str): return {tree}
        if isinstance(tree, tuple):
            return set().union(*(references(x) for x in tree[1:]))
        return set()
    def accept(info):
        return info["dimension"] in dimension_filter and (not required_fields or bool(set(required_fields) & references(info["expression"])))
    generator = FactorExpressionGenerator(list(fields), additional_field_dimensions=dimensions)
    params = {"stats_windows": list(windows), "time_periods": list(time_periods), "binary_random_seed": seed}
    raw = generator.auto_generate_all(complexity=complexity, params=params, operators=operators,
        final_filter=lambda name, info: accept(info), max_count=max_trials)
    selected = []
    for source_name, info in raw.items():
        if not accept(info): continue
        expr = canonical(info["expression"])
        selected.append({"name": "f_" + hashlib.sha256(expr.encode()).hexdigest()[:16],
                         "source_name": source_name, "expression": expr, "dimension": info["dimension"]})
        if len(selected) >= max_trials: break
    return selected, {"generated_before_selection": len(raw), "selected_count": len(selected)}
