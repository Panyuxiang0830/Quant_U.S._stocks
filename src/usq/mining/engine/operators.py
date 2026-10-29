"""Operator library for factor expressions."""

__all__ = ['Operators']

import pandas as pd
import numpy as np
from typing import Callable, Dict, List


class Operators:
    """
    算子类，包含所有可用的因子计算算子
    """

    def __init__(self):
        self.operators = self._initialize_operators()

    def _initialize_operators(self) -> Dict[str, Callable]:
        # TODO 添加算子包括 > < quantile groupby index

        """初始化基础算子库"""

        operators = {
            # 基本数学运算
            '+': self._add,
            '-': self._subtract,
            '*': self._multiply,
            '/': self._divide,
            'pow': self._power,
            'log': self._log,
            # 'exp': self._exp,
            'abs': self._abs,

            # 统计运算
            'mean': self._mean,
            'std': self._std,
            'max': self._max,
            'min': self._min,
            'sum': self._sum,
            'median': self._median,
            'skew': self._skew,
            'kurt': self._kurt,
            'rank': self._rank,
            'zscore': self._zscore,
            'pct_change': self._pct_change,

            # 时间序列操作
            'lag': self._lag,
            'delta': self._delta,

            # 回归计算
            'correlation': self._correlation,
            'beta': self._beta,
            'alpha': self._alpha_op,
        }
        return operators

    # 基本数学运算
    def _add(self, a, b):
        return a + b

    def _subtract(self, a, b):
        return a - b

    def _multiply(self, a, b):
        return a * b

    def _divide(self, a, b):
        return a.where(b != 0, np.nan) / b.where(b != 0, 1)

    def _power(self, a, b):
        return a ** b

    def _log(self, a):
        a_positive = a.where(a > 0, np.nan)
        return np.log(a_positive)

    # def _exp(self, a):
    #     return np.exp(a)

    def _abs(self, a):
        return np.abs(a)

    # 统计运算
    def _mean(self, a, window):
        return a.rolling(window=window, min_periods=1).mean()

    def _std(self, a, window):
        return a.rolling(window=window, min_periods=3).std()

    def _max(self, a, window):
        return a.rolling(window=window, min_periods=3).max()

    def _min(self, a, window):
        return a.rolling(window=window, min_periods=3).min()

    def _sum(self, a, window):
        return a.rolling(window=window, min_periods=1).sum()

    def _median(self, a, window):
        return a.rolling(window=window, min_periods=1).median()

    def _skew(self, a, window):
        return a.rolling(window=window, min_periods=3).skew()

    def _kurt(self, a, window):
        return a.rolling(window=window, min_periods=5).kurt()

    def _rank(self, a):
        return a.rank(axis=1, method='min',pct = True)

    def _zscore(self, a):
        # 处理inf和- inf值，将其替换为NaN，以便在计算均值和标准差时被忽略
        a_filtered = a.where((a != np.inf) & (a != -np.inf), np.nan)

        # 计算均值和标准差时自动忽略NaN值
        mean_values = a_filtered.mean(axis=1)
        std_values = a_filtered.std(axis=1)

        # 为了避免除以0的情况，将标准差为0的地方设置为1
        # 这样z-score会变成0，这是合理的结果
        std_values = std_values.replace(0, 1)

        # 计算z-score
        zscore_result = (a_filtered.subtract(mean_values, axis=0).divide(std_values, axis=0))

        return zscore_result

    def _pct_change(self, a, periods):
        return a.pct_change(periods=periods,fill_method=None)

    # 时间序列操作
    def _lag(self, a, periods):
        return a.shift(periods)

    def _delta(self, a, periods):
        return a.diff(periods)

    # 回归计算
    def _correlation(self, a, b, window):
        return a.rolling(window=window, min_periods=3).corr(b)

    def _beta(self, a, b, window):
        div = b.rolling(window=window, min_periods=3).var()
        div = div.where(div != 0, np.nan)
        return a.rolling(window=window, min_periods=3).cov(b) / div

    def _alpha_op(self, returns: pd.DataFrame, benchmark: pd.DataFrame, window: int) -> pd.DataFrame:
        """计算alpha因子"""
        beta = self._beta(returns, benchmark, window)
        # excess_return = returns - benchmark
        alpha = returns - beta * benchmark
        return alpha

    def get_available_operators(self) -> Dict[str, Callable]:
        """获取可用算子字典"""
        return self.operators

    def get_operator(self, operator_name: str) -> Callable:
        """获取特定算子"""
        return self.operators.get(operator_name)

    def get_operator_names(self) -> List[str]:
        """获取所有算子名称"""
        return list(self.operators.keys())
