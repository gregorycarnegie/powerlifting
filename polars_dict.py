from typing import Literal, Union, TypedDict

import numpy.typing as npt
import polars as pl

from enums import Compare, Event, Sex

class PolarsDict(TypedDict):
    """
    A dictionary that only accepts keys of type str and values of type pl.Expr.
    """
    lift_values: npt.NDArray
    user_total: Union[float, Literal[0]]
    percentile: float
    user_event: Event
    user_sex: Sex
    compare_within_weight_class: Compare
    user_weight_class: str
    user_bodyweight: float
    df_filtered: pl.DataFrame
