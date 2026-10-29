"""Factor expression generator for automated factor discovery."""

import random
from typing import Callable, Dict, List, Optional, Tuple, Union

from .operators import Operators

__all__ = ['FactorExpressionGenerator']

class FactorExpressionGenerator:
    """
    因子表达式自动生成器
    根据预定义规则和参数自动生成因子表达式
    """

    def __init__(self, base_fields, additional_field_dimensions = None, generator=None):
        """
        初始化因子表达式生成器

        参数:
        base_fields: 基础字段列表
        generator: FactorExpressionCalculator 实例（可选）
        """
        self.base_fields = base_fields
        # 创建一个临时的Operators实例来获取操作符列表
        operator_instance = Operators()
        self.operators = operator_instance.get_operator_names()
        
        # 定义字段的量纲类型，用于避免无意义的运算
        self.field_dimensions = self._initialize_field_dimensions(additional_field_dimensions)
        
        # 定义操作符的量纲规则 {operator: (input_dims, output_dim)}
        self.operator_dimension_rules = self._initialize_operator_dimension_rules()
        
        # 常用的参数
        self.default_stats_windows = [5, 10, 20]
        self.default_time_periods = [1, 5, 10]

        if generator is not None:
            self.generator = generator

    def _initialize_field_dimensions(self, additional_field_dimensions:Dict) -> Dict[str, str]:
        """
        初始化字段量纲类型
        通过字段名称推断其量纲类型
        """
        dimensions = {}

        price_cols = [
            'open', 'high', 'low', 'close', 'pre_close', 'change', 'vwap'
        ]

        vol_cols = [
            'vol', 'total_share', 'float_share',
            # 开盘10分钟成交量相关
            'open_10min_volume_sum',
            'open_10min_small_buy_volume_sum', 'open_10min_small_buy_volume_std',

            'open_10min_small_sell_volume_sum', 'open_10min_small_sell_volume_std',

            'open_10min_medium_buy_volume_sum', 'open_10min_medium_buy_volume_std',

            'open_10min_medium_sell_volume_sum', 'open_10min_medium_sell_volume_std',

            'open_10min_large_buy_volume_sum', 'open_10min_large_buy_volume_std',

            'open_10min_large_sell_volume_sum', 'open_10min_large_sell_volume_std',

            # 开盘60分钟成交量相关
            'open_60min_volume_sum',
            'open_60min_small_buy_volume_sum', 'open_60min_small_buy_volume_std',

            'open_60min_small_sell_volume_sum', 'open_60min_small_sell_volume_std',

            'open_60min_medium_buy_volume_sum', 'open_60min_medium_buy_volume_std',

            'open_60min_medium_sell_volume_sum', 'open_60min_medium_sell_volume_std',

            'open_60min_large_buy_volume_sum', 'open_60min_large_buy_volume_std',

            'open_60min_large_sell_volume_sum', 'open_60min_large_sell_volume_std',

            # 中间120分钟成交量相关
            'mid_120min_volume_sum',
            'mid_120min_small_buy_volume_sum', 'mid_120min_small_buy_volume_std',

            'mid_120min_small_sell_volume_sum', 'mid_120min_small_sell_volume_std',

            'mid_120min_medium_buy_volume_sum', 'mid_120min_medium_buy_volume_std',

            'mid_120min_medium_sell_volume_sum', 'mid_120min_medium_sell_volume_std',

            'mid_120min_large_buy_volume_sum', 'mid_120min_large_buy_volume_std',

            'mid_120min_large_sell_volume_sum', 'mid_120min_large_sell_volume_std',

            # 收盘前60分钟成交量相关
            'close_60min_volume_sum',
            'close_60min_small_buy_volume_sum', 'close_60min_small_buy_volume_std',

            'close_60min_small_sell_volume_sum', 'close_60min_small_sell_volume_std',

            'close_60min_medium_buy_volume_sum', 'close_60min_medium_buy_volume_std',

            'close_60min_medium_sell_volume_sum', 'close_60min_medium_sell_volume_std',

            'close_60min_large_buy_volume_sum', 'close_60min_large_buy_volume_std',

            'close_60min_large_sell_volume_sum', 'close_60min_large_sell_volume_std',

            # 收盘前10分钟成交量相关
            'close_10min_volume_sum',
            'close_10min_small_buy_volume_sum', 'close_10min_small_buy_volume_std',

            'close_10min_small_sell_volume_sum', 'close_10min_small_sell_volume_std',

            'close_10min_medium_buy_volume_sum', 'close_10min_medium_buy_volume_std',

            'close_10min_medium_sell_volume_sum', 'close_10min_medium_sell_volume_std',

            'close_10min_large_buy_volume_sum', 'close_10min_large_buy_volume_std',

            'close_10min_large_sell_volume_sum', 'close_10min_large_sell_volume_std',

        ]

        amount_cols = [
            'amount',
            # 开盘10分钟成交额相关
            'open_10min_amount_sum',
            'open_10min_small_buy_amount_sum', 'open_10min_small_buy_amount_std',

            'open_10min_small_sell_amount_sum', 'open_10min_small_sell_amount_std',

            'open_10min_medium_buy_amount_sum', 'open_10min_medium_buy_amount_std',

            'open_10min_medium_sell_amount_sum', 'open_10min_medium_sell_amount_std',

            'open_10min_large_buy_amount_sum', 'open_10min_large_buy_amount_std',

            'open_10min_large_sell_amount_sum', 'open_10min_large_sell_amount_std',

            # 开盘60分钟成交额相关
            'open_60min_amount_sum',
            'open_60min_small_buy_amount_sum', 'open_60min_small_buy_amount_std',

            'open_60min_small_sell_amount_sum', 'open_60min_small_sell_amount_std',

            'open_60min_medium_buy_amount_sum', 'open_60min_medium_buy_amount_std',

            'open_60min_medium_sell_amount_sum', 'open_60min_medium_sell_amount_std',

            'open_60min_large_buy_amount_sum', 'open_60min_large_buy_amount_std',

            'open_60min_large_sell_amount_sum', 'open_60min_large_sell_amount_std',

            # 中间120分钟成交额相关
            'mid_120min_amount_sum',
            'mid_120min_small_buy_amount_sum', 'mid_120min_small_buy_amount_std',

            'mid_120min_small_sell_amount_sum', 'mid_120min_small_sell_amount_std',

            'mid_120min_medium_buy_amount_sum', 'mid_120min_medium_buy_amount_std',

            'mid_120min_medium_sell_amount_sum', 'mid_120min_medium_sell_amount_std',

            'mid_120min_large_buy_amount_sum', 'mid_120min_large_buy_amount_std',

            'mid_120min_large_sell_amount_sum', 'mid_120min_large_sell_amount_std',

            # 收盘前60分钟成交额相关
            'close_60min_amount_sum',
            'close_60min_small_buy_amount_sum', 'close_60min_small_buy_amount_std',

            'close_60min_small_sell_amount_sum', 'close_60min_small_sell_amount_std',

            'close_60min_medium_buy_amount_sum', 'close_60min_medium_buy_amount_std',

            'close_60min_medium_sell_amount_sum', 'close_60min_medium_sell_amount_std',

            'close_60min_large_buy_amount_sum', 'close_60min_large_buy_amount_std',

            'close_60min_large_sell_amount_sum', 'close_60min_large_sell_amount_std',

            # 收盘前10分钟成交额相关
            'close_10min_amount_sum',
            'close_10min_small_buy_amount_sum', 'close_10min_small_buy_amount_std',

            'close_10min_small_sell_amount_sum', 'close_10min_small_sell_amount_std',

            'close_10min_medium_buy_amount_sum', 'close_10min_medium_buy_amount_std',

            'close_10min_medium_sell_amount_sum', 'close_10min_medium_sell_amount_std',

            'close_10min_large_buy_amount_sum', 'close_10min_large_buy_amount_std',

            'close_10min_large_sell_amount_sum', 'close_10min_large_sell_amount_std',

        ]

        percentage_cols = [
            'pct_chg', 'turnover_rate', 'pe', 'pe_ttm',
            # 开盘10分钟收益相关
            'open_10min_rtn_sum', 'open_10min_rtn_std', 'open_10min_rtn_min', 'open_10min_rtn_max',

            # 开盘60分钟收益相关
            'open_60min_rtn_sum', 'open_60min_rtn_std', 'open_60min_rtn_min', 'open_60min_rtn_max',

            # 中间120分钟收益相关
            'mid_120min_rtn_sum', 'mid_120min_rtn_std', 'mid_120min_rtn_min', 'mid_120min_rtn_max',

            # 收盘前60分钟收益相关
            'close_60min_rtn_sum', 'close_60min_rtn_std', 'close_60min_rtn_min', 'close_60min_rtn_max',

            # 收盘前10分钟收益相关
            'close_10min_rtn_sum', 'close_10min_rtn_std', 'close_10min_rtn_min', 'close_10min_rtn_max',

            # 影线相关
            'open_10min_up_shadow', 'open_10min_down_shadow',
            'open_60min_up_shadow', 'open_60min_down_shadow',
            'mid_120min_up_shadow', 'mid_120min_down_shadow',
            'close_60min_up_shadow', 'close_60min_down_shadow',
            'close_10min_up_shadow', 'close_10min_down_shadow',
            # 成交额比例相关
            'open_10min_amount_sum_ratio10day',
            'open_10min_large_buy_amount_sum_ratio10day',
            'open_10min_large_sell_amount_sum_ratio10day',
            'open_60min_amount_sum_ratio10day',
            'open_60min_large_buy_amount_sum_ratio10day',
            'open_60min_large_sell_amount_sum_ratio10day',
            'mid_120min_amount_sum_ratio10day',
            'mid_120min_large_buy_amount_sum_ratio10day',
            'mid_120min_large_sell_amount_sum_ratio10day',
            'close_60min_amount_sum_ratio10day',
            'close_60min_large_buy_amount_sum_ratio10day',
            'close_60min_large_sell_amount_sum_ratio10day',
            'close_10min_amount_sum_ratio10day',
            'close_10min_large_buy_amount_sum_ratio10day',
            'close_10min_large_sell_amount_sum_ratio10day',
            # 波动范围相关
            'open_10min_range', 'open_10min_range_std',
            'open_60min_range', 'open_60min_range_std',
            'mid_120min_range', 'mid_120min_range_std',
            'close_60min_range', 'close_60min_range_std',
            'close_10min_range', 'close_10min_range_std',
            # 换手率相关
            'open_10min_turnover',
            'open_60min_turnover',
            'mid_120min_turnover',
            'close_60min_turnover',
            'close_10min_turnover'
        ]

        bool_cols = ['ST', 'trade_status']

        unitless_cols = [
            'open_10min_rtn_amount_corr', 'open_60min_rtn_amount_corr',
            'mid_120min_rtn_amount_corr', 'close_60min_rtn_amount_corr', 'close_10min_rtn_amount_corr',
            'open_10min_price_amount_corr', 'open_60min_price_amount_corr',
            'mid_120min_price_amount_corr', 'close_60min_price_amount_corr', 'close_10min_price_amount_corr',
            'open_10min_large_sell_volume_skew', 'open_10min_large_buy_volume_skew',
            'open_10min_medium_sell_volume_skew', 'open_10min_medium_buy_volume_skew',
            'open_10min_small_sell_volume_skew', 'open_10min_small_buy_volume_skew',
            'open_60min_large_sell_volume_skew', 'open_60min_large_buy_volume_skew',
            'open_60min_medium_sell_volume_skew', 'open_60min_medium_buy_volume_skew',
            'open_60min_small_sell_volume_skew', 'open_60min_small_buy_volume_skew',
            'mid_120min_large_sell_volume_skew', 'mid_120min_large_buy_volume_skew',
            'mid_120min_medium_sell_volume_skew', 'mid_120min_medium_buy_volume_skew',
            'mid_120min_small_sell_volume_skew', 'mid_120min_small_buy_volume_skew',
            'close_10min_rtn_skew', 'close_60min_rtn_skew', 'mid_120min_rtn_skew', 'open_60min_rtn_skew',
            'open_10min_rtn_skew',
            'close_60min_large_sell_volume_skew', 'close_60min_large_buy_volume_skew',
            'close_60min_medium_sell_volume_skew', 'close_60min_medium_buy_volume_skew',
            'close_60min_small_sell_volume_skew', 'close_60min_small_buy_volume_skew',
            'close_10min_large_sell_volume_skew', 'close_10min_large_buy_volume_skew',
            'close_10min_medium_sell_volume_skew', 'close_10min_medium_buy_volume_skew',
            'close_10min_small_sell_volume_skew', 'close_10min_small_buy_volume_skew',
            'open_10min_large_sell_amount_skew', 'open_10min_large_buy_amount_skew',
            'open_10min_medium_sell_amount_skew', 'open_10min_medium_buy_amount_skew',
            'open_10min_small_sell_amount_skew', 'open_10min_small_buy_amount_skew',
            'open_60min_large_sell_amount_skew', 'open_60min_large_buy_amount_skew',
            'open_60min_medium_sell_amount_skew', 'open_60min_medium_buy_amount_skew',
            'open_60min_small_sell_amount_skew', 'open_60min_small_buy_amount_skew',
            'close_60min_small_sell_amount_skew', 'close_60min_small_buy_amount_skew',
            'mid_120min_large_sell_amount_skew', 'mid_120min_large_buy_amount_skew',
            'mid_120min_medium_sell_amount_skew', 'mid_120min_medium_buy_amount_skew',
            'mid_120min_small_buy_amount_skew', 'mid_120min_small_sell_amount_skew',
            'close_10min_small_sell_amount_skew', 'close_10min_small_buy_amount_skew',
            'close_60min_large_sell_amount_skew', 'close_60min_large_buy_amount_skew',
            'close_60min_medium_sell_amount_skew', 'close_60min_medium_buy_amount_skew',
            'close_10min_large_sell_amount_skew', 'close_10min_large_buy_amount_skew',
            'close_10min_medium_sell_amount_skew', 'close_10min_medium_buy_amount_skew'

        ]

        for field in self.base_fields:

            if field in price_cols:
                dimensions[field] = 'price'  # 价格类 (元/股)
            elif field in vol_cols:
                dimensions[field] = 'volume'  # 成交量类 (股)
            elif field in percentage_cols:
                dimensions[field] = 'percentage'  # 百分比类 (无量纲)
            elif field in amount_cols:
                dimensions[field] = 'amount'
            elif field in bool_cols:
                dimensions[field] = 'bool'
            elif field in unitless_cols:
                dimensions[field] = 'unitless'

        if additional_field_dimensions is not None:
            for key, value in additional_field_dimensions.items():
                dimensions[key] = value

        for field in self.base_fields:
            if field not in dimensions:
                raise ValueError(f"Field '{field}' not defined in field dimensions")
        return dimensions

    def _initialize_operator_dimension_rules(self) -> Dict[str, Tuple[List[str], str]]:
        """
        初始化操作符量纲规则
        格式: {operator: ([输入量纲列表], 输出量纲)}
        对于多个规则，输出量纲可以是函数或者标记
        """
        return {
            # 基本数学运算
            '+': (['same', 'same'], 'same'),  # 相同量纲才能相加
            '-': (['same', 'same'], 'same'),  # 相同量纲才能相减
            '*': (
            [['price', 'volume'], ['volume', 'price'], ['any', 'percentage'], ['percentage', 'any']], 'dependent'),
            # 根据输入确定输出
            '/': ([['amount', 'price'], ['any', 'percentage'], ['same', 'same']], 'dependent'),  # 根据输入确定输出
            'pow': (['any', 'any'], 'unitless'),  # 幂运算结果不确定

            # 无量纲操作符
            'log': (['any'], 'same_as_input'),  # 对数通常用于计算收益率
            'abs': (['any'], 'same_as_input'),  # 绝对值保持量纲
            'rank': (['any'], 'unitless'),  # 排名是序数
            'zscore': (['any'], 'unitless'),  # 标准化后无量纲

            # 统计运算
            'mean': (['any'], 'same_as_input'),  # 均值保持量纲
            'std': (['any'], 'same_as_input'),  # 标准差保持量纲
            'max': (['any'], 'same_as_input'),  # 最大值保持量纲
            'min': (['any'], 'same_as_input'),  # 最小值保持量纲
            'sum': (['any'], 'same_as_input'),  # 求和保持量纲
            'median': (['any'], 'same_as_input'),  # 中位数保持量纲
            'skew': (['any'], 'unitless'),  # 偏度无量纲
            'kurt': (['any'], 'unitless'),  # 峰度无量纲
            'pct_change': (['any'], 'percentage'),  # 百分比变化

            # 时间序列操作
            'lag': (['any'], 'same_as_input'),  # 滞后保持量纲
            'delta': (['any'], 'same_as_input'),  # 差分保持量纲

            # 双字段操作
            'correlation': (['any', 'any'], 'unitless'),  # 相关系数无量纲
            'beta': (['any', 'any'], 'unitless'),  # 贝塔系数无量纲
            'alpha': (['any', 'any'], 'unitless'),  # 阿尔法是收益率
        }

    def _get_expression_dimension(self, expression: Union[str, Tuple]) -> str:
        """
        获取表达式的量纲

        参数:
        expression: 表达式（字符串或元组）

        返回:
        量纲类型
        """
        if isinstance(expression, str):
            # 基础字段
            return self.field_dimensions[expression]
        elif isinstance(expression, tuple):
            # 复合表达式
            operator = expression[0]
            operands = expression[1:]

            if operator in self.operator_dimension_rules:
                rule = self.operator_dimension_rules[operator]
                output_dim = rule[1]

                # 处理特殊规则
                if output_dim == 'same_as_input':
                    # 返回第一个操作数的量纲
                    if operands:
                        return self._get_expression_dimension(operands[0])
                elif output_dim == 'same':
                    # 需要所有操作数具有相同量纲
                    operand_dims = [self._get_expression_dimension(op) for op in operands]
                    if len(set(operand_dims)) == 1:
                        return operand_dims[0]
                    else:
                        return 'inconsistent'
                elif output_dim == 'dependent':
                    # 根据操作符类型和输入量纲确定输出量纲
                    operand_dims = [self._get_expression_dimension(op) for op in operands]
                    if operator == '*':
                        if operand_dims in [['price', 'volume'], ['volume', 'price']]:
                            return 'amount'
                        elif operand_dims[1] == 'percentage':
                            return operand_dims[0]  # any * percentage -> any
                        elif operand_dims[0] == 'percentage':
                            return operand_dims[1]  # percentage * any -> any
                    elif operator == '/':
                        if operand_dims == ['amount', 'price']:
                            return 'volume'
                        elif operand_dims[1] == 'percentage':
                            return operand_dims[0]  # any / percentage -> any
                        elif len(set(operand_dims)) == 1:
                            return 'percentage'
                else:
                    return output_dim

        return 'unknown'

    def _is_dimension_compatible(self, expressions: List, operator: str) -> bool:
        """
        判断表达式列表在特定操作符下是否量纲兼容

        参数:
        expressions: 表达式列表
        operator: 操作符

        返回:
        是否量纲兼容
        """
        if operator not in self.operator_dimension_rules:
            return True  # 如果没有规则，默认兼容

        rule = self.operator_dimension_rules[operator]
        required_dims_patterns = rule[0]
        # 将单个规则转换为列表格式以统一处理
        if not isinstance(required_dims_patterns[0], list):
            required_dims_patterns = [required_dims_patterns]

        # 获取每个表达式的量纲
        expr_dims = [self._get_expression_dimension(expr) for expr in expressions]

        # 检查量纲是否匹配任一规则模式
        for required_dims in required_dims_patterns:
            if len(expr_dims) != len(required_dims):
                continue

            pattern_match = True
            for i, (actual_dim, required_dim) in enumerate(zip(expr_dims, required_dims)):
                if required_dim == 'any':
                    continue  # 任意量纲都可接受
                elif required_dim == 'same':
                    # 需要与第一个操作数相同量纲
                    if i > 0 and actual_dim != expr_dims[0]:
                        pattern_match = False
                        break
                else:
                    # 需要特定量纲
                    if actual_dim != required_dim:
                        pattern_match = False
                        break

            if pattern_match:
                return True

        return False

    def validate_expression(self, expression: Union[str, Tuple]) -> Dict[str, Union[bool, str]]:
        """
        验证给定表达式是否符合量纲要求以及_cross_operators_regulate要求

        参数:
        expression: 要验证的表达式（字符串或元组）

        返回:
        验证结果字典，包含:
        - is_valid: 是否有效
        - dimension: 量纲类型
        - errors: 错误信息列表
        """
        errors = []
        
        # 检查操作数数量
        operand_count_errors = self._check_operand_count(expression)
        if operand_count_errors:
            errors.extend(operand_count_errors)
        
        # 验证量纲兼容性
        try:
            dimension = self._get_expression_dimension(expression)
            if dimension == 'inconsistent':
                errors.append("表达式中存在量纲不一致的情况")
            elif dimension == 'unknown':
                errors.append("无法确定表达式的量纲")
        except Exception as e:
            errors.append(f"量纲计算出错: {str(e)}")
            dimension = 'unknown'
        
        # 验证_cross_operators_regulate要求
        cross_op_violations = self._check_cross_operators_regulations(expression)
        if cross_op_violations:
            errors.extend(cross_op_violations)
        
        is_valid = len(errors) == 0
        
        return {"is_valid": is_valid, "dimension": dimension, "errors": errors}
    
    def _check_cross_operators_regulations(self, expression: Union[str, Tuple]) -> List[str]:
        """
        检查表达式是否符合_cross_operators_regulate要求

        参数:
        expression: 要检查的表达式

        返回:
        违反规则的错误信息列表
        """
        errors = []
        
        if isinstance(expression, tuple) and len(expression) > 0:
            # 递归检查子表达式
            operator = expression[0]
            operands = expression[1:]
            
            # 如果是复合表达式，先检查子表达式
            for i, operand in enumerate(operands):
                if isinstance(operand, tuple):
                    sub_errors = self._check_cross_operators_regulations(operand)
                    errors.extend(sub_errors)
                    
                    # 检查当前操作符与子表达式操作符的兼容性
                    if isinstance(operand, tuple) and len(operand) > 0:
                        sub_operator = operand[0]
                        # 修正参数顺序：sub_operator 是内层操作符，operator 是外层操作符
                        if not self._cross_operators_regulate(sub_operator, operator):
                            errors.append(f"操作符 '{sub_operator}' 与外层操作符 '{operator}' 不兼容")
            
        return errors

    def _check_operand_count(self, expression: Union[str, Tuple]) -> List[str]:
        """
        检查表达式的操作数数量是否正确

        参数:
        expression: 要检查的表达式

        返回:
        错误信息列表
        """
        errors = []
        
        if isinstance(expression, tuple) and len(expression) > 0:
            operator = expression[0]
            operands = expression[1:]
            
            # 检查操作数数量是否符合规则
            if operator in self.operator_dimension_rules:
                # 特殊处理各种操作符
                if operator in ['sum', 'mean', 'std', 'max', 'min', 'median', 'skew', 'kurt']:
                    if len(operands) == 2:
                        # 有表达式操作数和窗口参数，检查窗口参数是否为整数
                        if not isinstance(operands[1], int):
                            errors.append(f"操作符 '{operator}' 的窗口参数必须是整数")
                    else:
                        errors.append(f"操作符 '{operator}' 期望2个操作数，但得到了 {len(operands)} 个")
                        
                elif operator in ['lag', 'delta', 'pct_change']:
                    # 时间序列运算: 1个表达式操作数 + 可选的整数参数
                    if len(operands) == 2:
                        # 有表达式操作数和时间参数，检查时间参数是否为整数
                        if not isinstance(operands[1], int):
                            errors.append(f"操作符 '{operator}' 的时间参数必须是整数")
                    else:
                        errors.append(f"操作符 '{operator}' 期望1个或2个操作数，但得到了 {len(operands)} 个")
                        
                elif operator in ['+', '-', '*', '/']:
                    # 基本数学运算符需要恰好2个操作数
                    if len(operands) != 2:
                        errors.append(f"操作符 '{operator}' 期望2个操作数，但得到了 {len(operands)} 个")
                elif operator in ['correlation', 'beta', 'alpha']:
                    if len(operands) == 3:
                        # 两个操作数 + 窗口参数
                        if not isinstance(operands[2], int):
                            errors.append(f"操作符 '{operator}' 的窗口参数必须是整数")
                    else:
                        errors.append(f"操作符 '{operator}' 期望3个操作数，但得到了 {len(operands)} 个")

                elif operator in ['log', 'abs', 'rank', 'zscore']:
                    # 单操作数操作符
                    if len(operands) != 1:
                        errors.append(f"操作符 '{operator}' 期望1个操作数，但得到了 {len(operands)} 个")
                        
                elif operator in ['pow']:
                    # 幂运算需要2个操作数
                    if len(operands) != 2:
                        errors.append(f"操作符 '{operator}' 期望2个操作数，但得到了 {len(operands)} 个")
                else:
                    # 其他操作符，暂时不检查操作数数量
                    pass
            
            # 递归检查子表达式
            for operand in operands:
                if isinstance(operand, tuple):
                    sub_errors = self._check_operand_count(operand)
                    errors.extend(sub_errors)
        
        return errors

    def _cross_operators_regulate(self, op1, op2):
        if op1 == 'pow':
            if op2 in ['log', 'pow']:
                return False

        if op1 == 'log':
            if op2 in ['pow', 'log']:
                return False

        if op1 == 'abs':
            if op2 in ['abs']:
                return False

        if op1 == 'rank':
            if op2 in ['rank', 'zscore', 'abs', 'log']:
                return False

        if op1 == 'zscore':
            if op2 in ['zscore', 'rank']:
                return False

        if op1 == 'lag':
            if op2 in ['lag']:
                return False

        if op1 == 'mean':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'std':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt', 'abs']:
                return False

        if op1 == 'max':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'min':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'sum':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'median':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'skew':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'kurt':
            if op2 in ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']:
                return False

        if op1 == 'correlation':
            if op2 in ['log']:
                return False

        if op1 == 'alpha':
            if op2 in ['log']:
                return False

        if op1 == 'beta':
            if op2 in ['log']:
                return False
        return True

    def generate_single_field_operators(self, fields: List[str] = None,
                                        operators: List[str] = None,
                                        params: Dict = None) -> Dict[str, Dict[str, Union[Tuple, str]]]:
        """
        生成单字段操作符因子表达式（一个操作符作用于一个字段）

        参数:
        fields: 数据字段列表，默认使用所有字段
        operators: 操作符列表
        params: 参数字典，如窗口大小等

        返回:
        因子表达式字典 {因子名: {"expression": 表达式, "dimension": 量纲}}
        """
        if fields is None:
            fields = self.base_fields

        expressions = {}

        # 无参数单字段操作符
        no_param_ops = ['abs', 'log', 'rank', 'zscore']
        if operators is not None:
            no_param_ops = [op for op in operators if op in no_param_ops]

        for field in fields:
            for op in no_param_ops:
                # 检查量纲兼容性
                if self._is_dimension_compatible([field], op):
                    factor_name = f"{field}_{op}"
                    expression = (op, field)
                    dimension = self._get_expression_dimension(expression)
                    expressions[factor_name] = {"expression": expression, "dimension": dimension}

        # 单参数单字段操作符（时间序列类）
        single_param_ops = ['lag', 'delta', 'pct_change']
        if operators is not None:
            single_param_ops = [op for op in operators if op in single_param_ops]

        default_periods = params.get('time_periods', self.default_time_periods) if params else self.default_time_periods
        for field in fields:
            for op in single_param_ops:
                for period in default_periods:
                    # 检查量纲兼容性
                    if self._is_dimension_compatible([field], op):
                        factor_name = f"{field}_{op}_{period}D"
                        expression = (op, field, period)
                        dimension = self._get_expression_dimension(expression)
                        expressions[factor_name] = {"expression": expression, "dimension": dimension}

        # 单参数单字段操作符（统计类）
        stat_ops = ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']
        if operators is not None:
            stat_ops = [op for op in operators if op in stat_ops]

        default_windows = params.get('stats_windows', self.default_stats_windows) if params else self.default_stats_windows
        for field in fields:
            for op in stat_ops:
                for window in default_windows:
                    # 检查量纲兼容性
                    if self._is_dimension_compatible([field], op):
                        factor_name = f"{field}_{op}_{window}D"
                        expression = (op, field, window)
                        dimension = self._get_expression_dimension(expression)
                        expressions[factor_name] = {"expression": expression, "dimension": dimension}

        return expressions

    def generate_dual_field_operators(self, fields: List[str] = None,
                                      operators: List[str] = None,
                                      params: Dict = None) -> Dict[str, Dict[str, Union[Tuple, str]]]:
        """
        生成双字段操作符因子表达式（一个操作符作用于两个字段）

        参数:
        fields: 数据字段列表，默认使用所有字段
        operators: 操作符列表
        params: 参数字典

        返回:
        因子表达式字典 {因子名: {"expression": 表达式, "dimension": 量纲}}
        """
        if fields is None:
            fields = self.base_fields

        expressions = {}

        # 双字段操作符
        dual_ops = ['+', '-', '*', '/', 'correlation', 'beta', 'alpha']
        if operators is not None:
            dual_ops = [op for op in operators if op in dual_ops]

        # 生成字段间的运算表达式
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:  # 这里不允许field1和field2相同！！！！！
                for op in dual_ops:
                    # 检查量纲兼容性
                    if self._is_dimension_compatible([field1, field2], op):
                        if op in ['+', '-', '*']:
                            factor_name = f"{field1}_{op}_{field2}"
                            expression = (op, field1, field2)
                            dimension = self._get_expression_dimension(expression)
                            expressions[factor_name] = {"expression": expression, "dimension": dimension}
                        elif op in ['correlation', 'beta', 'alpha']:
                            # 这些操作符还需要窗口参数
                            default_windows = params.get('stats_windows', self.default_stats_windows) if params else self.default_stats_windows
                            for window in default_windows:
                                factor_name = f"{field1}_{op}_{field2}_{window}D"
                                expression = (op, field1, field2, window)
                                dimension = self._get_expression_dimension(expression)
                                expressions[factor_name] = {"expression": expression, "dimension": dimension}

        # 有顺序差异的操作符
        for i, field1 in enumerate(fields):
            for field2 in fields:
                # 避免重复相同字段的组合
                if field1 == field2:
                    continue
                for op in dual_ops:
                    if op in ['/']:
                        # 检查量纲兼容性
                        if self._is_dimension_compatible([field1, field2], op):
                            factor_name = f"{field1}_{op}_{field2}"
                            expression = (op, field1, field2)
                            dimension = self._get_expression_dimension(expression)
                            expressions[factor_name] = {"expression": expression, "dimension": dimension}

        return expressions

    def auto_generate_all(self, fields: List[str] = None,
                          operators: List[str] = None,
                          params: Dict = None,
                          complexity: int = 2,
                          final_filter: Optional[Callable[[str, Dict], bool]] = None,
                          max_count: Optional[int] = None) -> Dict[str, Dict[str, Union[Tuple, str]]]:
        """
        自动生成所有因子表达式

        参数:
        fields: 数据字段列表
        operators: 操作符列表
        params: 参数字典
        complexity: 表达式复杂度
        final_filter: 仅应用于最终层的候选过滤函数
        max_count: 最终层通过 final_filter 的候选数量上限

        返回:
        所有生成的因子表达式字典，格式为 {因子名: {"expression": 表达式, "dimension": 量纲}}
        """
        if fields is None:
            fields = self.base_fields

        # 初始化所有表达式字典
        all_expressions = {}

        # 第0层复杂度：基础字段表达式（复杂度为0）
        base_expressions = {}
        for field in fields:
            dimension = self._get_expression_dimension(field)
            base_expressions[field] = {"expression": field, "dimension": dimension}

        # 第1层复杂度：直接基于基础字段的表达式
        single_expr = self.generate_single_field_operators(fields, operators, params)
        dual_expr = self.generate_dual_field_operators(fields, operators, params)

        # 合并第0层和第1层表达式
        current_level_expressions = {**base_expressions, **single_expr, **dual_expr}
        all_expressions.update(current_level_expressions)

        # 如果只需要第1层复杂度，直接返回
        if complexity <= 1:
            return all_expressions

        # 迭代生成更高层复杂度的表达式
        for level in range(2, complexity + 1):
            # 将当前所有表达式作为"字段"用于生成新的表达式
            # 但为了避免组合爆炸，我们只使用最近一层生成的表达式与之前所有表达式组合
            previous_expressions = current_level_expressions.copy()

            # 生成单字段操作表达式（基于之前所有表达式）
            single_field_new = {}
            is_final_level = level == complexity
            final_selected_count = 0
            stop_level = False

            def add_candidate(target, factor_name, expression):
                """写入候选，并在最终层达到配额时请求停止。"""
                nonlocal final_selected_count, stop_level

                dimension = self._get_expression_dimension(expression)
                factor_info = {"expression": expression, "dimension": dimension}
                if is_final_level and final_filter is not None:
                    if not final_filter(factor_name, factor_info):
                        return False
                    if max_count is not None and final_selected_count >= max_count:
                        stop_level = True
                        return True
                    final_selected_count += 1

                target[factor_name] = factor_info
                if (
                    is_final_level
                    and final_filter is not None
                    and max_count is not None
                    and final_selected_count >= max_count
                ):
                    stop_level = True
                return stop_level

            no_param_ops = ['abs', 'log', 'rank', 'zscore']
            if operators is not None:
                no_param_ops = [op for op in operators if op in no_param_ops]

            # 只对最近一层生成的表达式应用单字段操作符
            for name, expr_info in previous_expressions.items():
                expr = expr_info["expression"]
                # 检查是否是新生成的表达式（不是原始字段）
                if name not in base_expressions:
                    for op in no_param_ops:
                        # 检查跨层操作是否兼容
                        if self._cross_operators_regulate(expr[0] if isinstance(expr, tuple) else '', op):
                            # 检查量纲兼容性
                            if self._is_dimension_compatible([expr], op):
                                factor_name = f"{name}_{op}_L{level}"
                                new_expr = (op, expr)
                                if add_candidate(single_field_new, factor_name, new_expr):
                                    break
                        if stop_level:
                            break
                if stop_level:
                    break

            # 补齐其他统计类型操作
            stat_ops = ['mean', 'std', 'max', 'min', 'sum', 'median', 'skew', 'kurt']
            if operators is not None:
                stat_ops = [op for op in operators if op in stat_ops]
            
            # 默认窗口大小
            default_windows = params.get('stats_windows', self.default_stats_windows) if params else self.default_stats_windows
            
            for name, expr_info in previous_expressions.items():
                expr = expr_info["expression"]
                # 检查是否是新生成的表达式（不是原始字段）
                if name not in base_expressions:
                    for op in stat_ops:
                        for window in default_windows:
                            # 检查跨层操作是否兼容
                            if self._cross_operators_regulate(expr[0] if isinstance(expr, tuple) else '', op):
                                # 检查量纲兼容性
                                if self._is_dimension_compatible([expr], op):
                                    factor_name = f"{name}_{op}_{window}D_L{level}"
                                    new_expr = (op, expr, window)
                                    if add_candidate(single_field_new, factor_name, new_expr):
                                        break
                            if stop_level:
                                break
                        if stop_level:
                            break
                if stop_level:
                    break

            # 补齐 时间序列的操作
            time_series_ops = ['lag', 'delta', 'pct_change']
            if operators is not None:
                time_series_ops = [op for op in operators if op in time_series_ops]
            
            # 默认周期参数
            default_periods = params.get('time_periods', self.default_time_periods) if params else self.default_time_periods
            
            for name, expr_info in previous_expressions.items():
                expr = expr_info["expression"]
                # 检查是否是新生成的表达式（不是原始字段）
                if name not in base_expressions:
                    for op in time_series_ops:
                        for period in default_periods:
                            # 检查跨层操作是否兼容
                            if self._cross_operators_regulate(expr[0] if isinstance(expr, tuple) else '', op):
                                # 检查量纲兼容性
                                if self._is_dimension_compatible([expr], op):
                                    factor_name = f"{name}_{op}_{period}D_L{level}"
                                    new_expr = (op, expr, period)
                                    if add_candidate(single_field_new, factor_name, new_expr):
                                        break
                            if stop_level:
                                break
                        if stop_level:
                            break
                if stop_level:
                    break

            # 生成双字段操作表达式（基于之前所有表达式和基础字段）
            dual_field_new = {}
            if not stop_level:
                previous_items = list(previous_expressions.items())
                all_items = list(all_expressions.items())

                # 每层只抽取有限的非自配对，避免字段顺序决定 RHS。
                total_pairs = len(previous_items) * len(all_items)
                all_names = {name for name, _ in all_items}
                self_pair_count = sum(name in all_names for name, _ in previous_items)
                max_combinations = min(100, max(0, total_pairs - self_pair_count))
                random_seed = params.get('binary_random_seed', 0) if params else 0
                rng = random.Random(random_seed + level)
                sampled_pairs = set()
                while len(sampled_pairs) < max_combinations and total_pairs:
                    flat_index = rng.randrange(total_pairs)
                    left_index, right_index = divmod(flat_index, len(all_items))
                    if previous_items[left_index][0] == all_items[right_index][0]:
                        continue
                    sampled_pairs.add((left_index, right_index))

                dual_ops = ['+', '-', '*', '/', 'correlation', 'beta']
                if operators is not None:
                    dual_ops = [op for op in operators if op in dual_ops]

                for left_index, right_index in sorted(sampled_pairs):
                    name1, expr1_info = previous_items[left_index]
                    name2, expr2_info = all_items[right_index]
                    expr1 = expr1_info["expression"]
                    expr2 = expr2_info["expression"]
                    for op in dual_ops:
                        if self._is_dimension_compatible([expr1, expr2], op):
                            if op in ['+', '-', '*', '/']:
                                factor_name = f"{name1}_{op}_{name2}"
                                expression = (op, expr1, expr2)
                                if add_candidate(dual_field_new, factor_name, expression):
                                    break
                            elif op in ['correlation', 'beta', 'alpha']:
                                default_windows = params.get('stats_windows', self.default_stats_windows) if params else self.default_stats_windows
                                for window in default_windows:
                                    factor_name = f"{name1}_{op}_{name2}_{window}D"
                                    expression = (op, expr1, expr2, window)
                                    if add_candidate(dual_field_new, factor_name, expression):
                                        break
                        if stop_level:
                            break
                    if stop_level:
                        break

            # 更新当前层表达式
            current_level_expressions = {**single_field_new, **dual_field_new}
            all_expressions.update(current_level_expressions)

            # 如果当前层没有生成新表达式，则提前结束
            if not current_level_expressions:
                break

        return all_expressions

    def register_auto_factors(self, factor_dict: Dict[str, Dict[str, Union[Tuple, str]]],
                              prefix: str = "auto") -> None:
        """
        将自动生成的因子表达式注册到因子生成器

        参数:
        factor_dict: 因子表达式字典，格式为 {因子名: {"expression": 表达式, "dimension": 量纲}}
        prefix: 因子名称前缀
        """
        for name, info in factor_dict.items():
            factor_name = f"{prefix}_{name}" if prefix else name
            expression = info["expression"]
            dimension = info["dimension"]
            description = f"Auto-generated factor: {name} (Dimension: {dimension})"
            # 注意：这里的self.generator需要在使用时传入实际的generator实例
            # 如果没有传入，这个方法将无法正常工作
            if hasattr(self, 'generator') and self.generator:
                self.generator.register_factor(factor_name, expression, description)
