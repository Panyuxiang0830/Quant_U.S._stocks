"""Original model constants; storage and US data mapping live in adapters."""
import os

MARKET = "Market"  # 市场因子：每只股票暴露为 1，收益≈全市场市值加权收益

# CNE5 风格因子 -> (细项, 权重)。缺失的细项在每个股票上按可用权重重新归一。
STYLE_SPEC: dict[str, list[tuple[str, float]]] = {
    "Size": [("LNCAP", 1.0)],
    "Beta": [("BETA", 1.0)],
    "Momentum": [("RSTR", 1.0)],
    "ResidualVolatility": [("DASTD", 0.74), ("CMRA", 0.16), ("HSIGMA", 0.10)],
    "BookToPrice": [("BTOP", 1.0)],
    "Liquidity": [("STOM", 0.35), ("STOQ", 0.35), ("STOA", 0.30)],
    "EarningsYield": [("EPFWD", 0.68), ("CETOP", 0.21), ("ETOP", 0.11)],
    "Growth": [("EGRLF", 0.18), ("EGRSF", 0.11), ("EGRO", 0.24), ("SGRO", 0.47)],
    "Leverage": [("MLEV", 0.38), ("DTOA", 0.35), ("BLEV", 0.27)],
}
# 正交化：因子 -> 需要剔除的因子（市值加权回归取残差）
ORTHOGONALIZE = {"ResidualVolatility": ["Beta", "Size"], "Liquidity": ["Size"]}
# 正交化回归权重的指数：cap**ORTHO_CAP_POWER。官方 CNE5 为 1（市值加权）；这里取 0.5，
# 与因子收益回归的 √市值 权重一致，否则 A 股市值高度集中时，Size 与 NonLinearSize
# 在回归口径下仍有约 0.47 的相关（2024-2025 实测），造成共线性。
ORTHO_CAP_POWER = 0.5
STYLE_FACTORS = [
    "Size", "Beta", "Momentum", "ResidualVolatility", "NonLinearSize",
    "BookToPrice", "Liquidity", "EarningsYield", "Growth", "Leverage",
]

# 时间窗（交易日）与半衰期，取自 Barra CNE5 Empirical Notes
BETA_WINDOW, BETA_HALF_LIFE, BETA_MIN_OBS = 252, 63, 126
DASTD_WINDOW, DASTD_HALF_LIFE = 252, 42
RSTR_WINDOW, RSTR_LAG, RSTR_HALF_LIFE, RSTR_MIN_OBS = 504, 21, 126, 252
CMRA_MONTHS, MONTH_DAYS = 12, 21
WINSOR_MAD = 5.0
RF_ANNUAL = float(os.environ.get("BARRA_RF_ANNUAL", "0.0"))
REGRESSION_MAX_ABS_RETURN = 0.35  # 超过视为异常（如新股），不进回归，但仍计算特异收益
