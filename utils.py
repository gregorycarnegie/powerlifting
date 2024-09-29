import polars as pl
import numpy.typing as npt
from typing import Literal, Union
import matplotlib.pyplot as plt
from pathlib import Path

SEX = Literal['M', 'F']
COMPARE_WEIGHT_CLASS = Literal['Y', 'N']

EVENTS: dict[str, tuple[str, ...]] = {
    'SBD': ('Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg'),
    'BD': ('Best3BenchKg', 'Best3DeadliftKg'),
    'SD': ('Best3SquatKg', 'Best3DeadliftKg'),
    'SB': ('Best3SquatKg', 'Best3BenchKg'),
    'S': ('Best3SquatKg',),
    'B': ('Best3BenchKg',),
    'D': ('Best3DeadliftKg',)
}

# Updated weight class boundaries and labels for men and women
male_weight_classes: list[tuple[int, int, str]] = [
    (0, 52, '-52 kg'),
    (52, 56, '-56 kg'),
    (56, 60, '-60 kg'),
    (60, 67.5, '-67.5 kg'),
    (67.5, 75, '-75 kg'),
    (75, 82.5, '-82.5 kg'),
    (82.5, 90, '-90 kg'),
    (90, 100, '-100 kg'),
    (100, 110, '-110 kg'),
    (110, 125, '-125 kg'),
    (125, 140, '-140 kg'),
    (140, 1000, '140 kg+')
]

female_weight_classes: list[tuple[int, int, str]] = [
    (0, 44, '-44 kg'),
    (44, 48, '-48 kg'),
    (48, 52, '-52 kg'),
    (52, 56, '-56 kg'),
    (56, 60, '-60 kg'),
    (60, 67.5, '-67.5 kg'),
    (67.5, 75, '-75 kg'),
    (75, 82.5, '-82.5 kg'),
    (82.5, 90, '-90 kg'),
    (90, 1000, '90 kg+')
]


# Function to create the weight class assignment expression
def custom_weight_class_expr() -> pl.Expr:
    conditions = []
    # For male weight classes
    for min_wt, max_wt, label in male_weight_classes:
        condition = (pl.col('Sex') == 'M')
        condition &= pl.col('BodyweightKg') > min_wt
        condition &= pl.col('BodyweightKg') <= max_wt
        conditions.append((condition, label))
    # For female weight classes
    for min_wt, max_wt, label in female_weight_classes:
        condition = (pl.col('Sex') == 'F')
        condition &= pl.col('BodyweightKg') > min_wt
        condition &= pl.col('BodyweightKg') <= max_wt
        conditions.append((condition, label))
    # Build the when-then-otherwise chain
    expr = None
    for cond, label in conditions:
        if expr is None:
            expr = pl.when(cond).then(pl.lit(label))
        else:
            expr = expr.when(cond).then(pl.lit(label))
    expr = expr.otherwise(pl.lit('Unknown')).alias('CustomWeightClass')
    return expr


# Assign custom weight classes based on 'BodyweightKg' and 'Sex'
def assign_user_weight_class(sex: SEX, bw: float) -> str:
    weight_classes = male_weight_classes if sex == 'M' else female_weight_classes
    return next(
        (
            label
            for min_wt, max_wt, label in weight_classes
            if min_wt < bw <= max_wt
        ),
        'Unknown',
    )


def get_float_input(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_schema_overrides(data:Path) -> dict[str, Union[pl.Float64, pl.String]]:
    with open(data, 'r') as f:
        header = f.readline()
    column_names = header.strip().split(',')
    kg_columns = (col for col in column_names if col.endswith('Kg'))
    return {col: (pl.Float64 if col != 'WeightClassKg' else pl.String) for col in kg_columns}


def get_lifter_info() -> tuple[str, str, str]:
    user_sex = input("Enter your sex (M or F): ").strip().upper()
    if user_sex not in {'M', 'F'}:
        print("Invalid sex entered.")
        exit()

    user_event = input("Enter your event: ").strip().upper()
    if user_event not in EVENTS:
        print("Invalid event.")
        exit()

    user_bodyweight = get_float_input("Enter your body weight in kg: ")
    user_weight_class = assign_user_weight_class(user_sex, user_bodyweight)

    return user_sex, user_event, user_weight_class


def get_filter_expr(user_sex: SEX,
                   user_event: str, 
                   user_weight_class: str, 
                   relevant_columns: tuple[str, ...], 
                   compare_within_weight_class: COMPARE_WEIGHT_CLASS) -> pl.Expr:
    match compare_within_weight_class:
        case 'Y':
            return (
                pl.col('BodyweightKg').is_not_null() & 
                (pl.col('BodyweightKg') > 0) &
                (pl.col('Sex') == user_sex) &
                (pl.col('Event') == user_event) &
                (pl.col('CustomWeightClass') == user_weight_class) &
                pl.struct(relevant_columns).is_not_null()
            )
        case 'N':
            return (
                pl.col('BodyweightKg').is_not_null() & 
                (pl.col('BodyweightKg') > 0) &
                (pl.col('Sex') == user_sex) &
                (pl.col('Event') == user_event) &
                pl.struct(relevant_columns).is_not_null()
            )


def process_data(data: Path) -> tuple[npt.NDArray, Union[float, Literal[0]], float, str, str, str, str]:
    schema_overrides = get_schema_overrides(data)

    df = pl.read_csv(data, schema_overrides=schema_overrides)
    
    df = df.with_columns(custom_weight_class_expr())

    user_sex, user_event, user_weight_class = get_lifter_info()

    if user_weight_class == 'Unknown':
        print("No matching weight class found for your body weight.")
        exit()

    compare_within_weight_class = input("Do you want to compare your lifts within your weight class? (yes/no): ").strip().upper()[0]
    relevant_columns = EVENTS.get(user_event)

    if not relevant_columns:
        print("No relevant columns found for this event.")
        exit()

    # If the user wants to compare within their weight class, add the weight class filter
    df_filtered = df.filter(
        get_filter_expr(user_sex, user_event, user_weight_class, relevant_columns, compare_within_weight_class)
    ).with_columns(
        pl.sum_horizontal(pl.col(col) for col in relevant_columns).alias('Total')
    )

    lift_values = df_filtered['Total'].to_numpy()
    lift_values = lift_values[lift_values > 0]

    user_lift_inputs = {lift: get_float_input(f"Enter your best {lift.lower()} (kg): ") for lift in relevant_columns}
    user_total = sum(user_lift_inputs.values())

    percentile = df_filtered.filter(pl.col('Total') < user_total).shape[0] / df_filtered.shape[0] * 100

    return lift_values, user_total, percentile, user_event, user_sex, compare_within_weight_class, user_weight_class


def plot_histogram(lift_values: npt.NDArray, 
                   user_total: Union[float, Literal[0]], 
                   percentile: float, 
                   user_event: str, 
                   user_sex: str, 
                   compare_within_weight_class: str, 
                   user_weight_class: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(lift_values, bins=50, alpha=0.7, label='Lifters')
    plt.axvline(user_total, color='red', linestyle='dashed', linewidth=2, label='Your Lift')
    plt.annotate(f'Your Percentile: {percentile:.2f}%', xy=(user_total, plt.ylim()[1]*0.9),
                 xytext=(user_total, plt.ylim()[1]*0.9), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', ha='center')

    plt.xlabel(f'{user_event} Total (kg)')
    plt.ylabel('Number of Lifters')
    title = f'Distribution of {user_event} Total for {user_sex} lifters'
    if compare_within_weight_class == 'Y':
        title += f' in Weight Class {user_weight_class}'
    else:
        title += ' across all weight classes'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
