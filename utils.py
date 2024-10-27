import cProfile
import logging
import pstats
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Union, Optional, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

from enums import Compare, Event, Sex
from polars_dict import PolarsDict
from weight_class import WeightClass

CoeffSum = TypeVar('CoeffSum', float, pl.Expr)
DATA_FRAME = TypeVar('DATA_FRAME', pl.DataFrame, pl.LazyFrame)

N_BINS = 50

# OLD coefficients
# WILKS_COEFFICIENTS = {
#     'M': [-216.0475144, 16.2606339, -0.002388645, -0.00113732, 7.01863e-06, -1.291e-08],
#     'F': [594.31747775582, -27.23842536447, 0.82112226871, -0.00930733913, 4.731582e-05, -9.054e-08]
# }

# 2020 coefficients
WILKS_COEFFICIENTS = {
    'M': [47.46178854, 8.472061379, 0.07369410346, -0.001395833811, 7.07665973070743e-6, -1.20804336482315e-8],
    'F': [-125.4255398, 13.71219419, -0.03307250631, -0.001050400051, 9.38773881462799e-6, -2.3334613884954e-8]
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


EVENTS: dict[Event, tuple[str, ...]] = {
    Event.SBD: ('Best3SquatKg', 'Best3BenchKg', 'Best3DeadliftKg'),
    Event.BD: ('Best3BenchKg', 'Best3DeadliftKg'),
    Event.SD: ('Best3SquatKg', 'Best3DeadliftKg'),
    Event.SB: ('Best3SquatKg', 'Best3BenchKg'),
    Event.S: ('Best3SquatKg',),
    Event.B: ('Best3BenchKg',),
    Event.D: ('Best3DeadliftKg',)
}

SEXES: dict[Sex, str] = {Sex.MALE: 'Male', Sex.FEMALE: 'Female'}

# Updated weight class boundaries and labels for men and women
male_weight_classes: list[WeightClass] = [
    WeightClass(0, 52, '-52 kg'),
    WeightClass(52, 56, '-56 kg'),
    WeightClass(56, 60, '-60 kg'),
    WeightClass(60, 67.5, '-67.5 kg'),
    WeightClass(67.5, 75, '-75 kg'),
    WeightClass(75, 82.5, '-82.5 kg'),
    WeightClass(82.5, 90, '-90 kg'),
    WeightClass(90, 100, '-100 kg'),
    WeightClass(100, 110, '-110 kg'),
    WeightClass(110, 125, '-125 kg'),
    WeightClass(125, 140, '-140 kg'),
    WeightClass(140, 1000, '140 kg+')
]

female_weight_classes: list[WeightClass] = [
    WeightClass(0, 44, '-44 kg'),
    WeightClass(44, 48, '-48 kg'),
    WeightClass(48, 52, '-52 kg'),
    WeightClass(52, 56, '-56 kg'),
    WeightClass(56, 60, '-60 kg'),
    WeightClass(60, 67.5, '-67.5 kg'),
    WeightClass(67.5, 75, '-75 kg'),
    WeightClass(75, 82.5, '-82.5 kg'),
    WeightClass(82.5, 90, '-90 kg'),
    WeightClass(90, 1000, '90 kg+')
]


def profile_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        with cProfile.Profile() as profile:
            result = func(*args, **kwargs)
        metrics = pstats.Stats(profile)
        metrics.sort_stats(pstats.SortKey.CUMULATIVE)
        metrics.print_stats()
        return result

    return wrapper


def add_weight_conditions(sex: Sex,
                          weight_classes: list[WeightClass],
                          conditions: Optional[pl.Expr] = None):
    for weight_class in weight_classes:
        condition = (
            (pl.col('Sex') == sex) &
            (pl.col('BodyweightKg') > weight_class.min_weight) &
            (pl.col('BodyweightKg') <= weight_class.max_weight)
        )
        if conditions is None:
            conditions = pl.when(condition).then(pl.lit(weight_class.label))
        else:
            conditions = conditions.when(condition).then(pl.lit(weight_class.label))
    return conditions

def custom_weight_class_expr() -> pl.Expr:
    # Build the conditions for both male and female weight classes
    # Add conditions for male weight classes
    conditions = add_weight_conditions(Sex.MALE, male_weight_classes)
    
    # Add conditions for female weight classes
    conditions = add_weight_conditions(Sex.FEMALE, female_weight_classes, conditions)
    
    # Add default case
    return conditions.otherwise(pl.lit('Unknown')).alias('CustomWeightClass')


# Assign custom weight classes based on 'BodyweightKg' and 'Sex'
def assign_user_weight_class(sex: Sex, bw: float) -> str:
    weight_classes = male_weight_classes if sex == 'M' else female_weight_classes
    return next(
        (
            weight_class.label
            for weight_class in weight_classes
            if weight_class.min_weight < bw <= weight_class.max_weight
        ),
        'Unknown',
    )


def get_float_input(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError as e:
            logger.info(f"{e}: Invalid input. Please enter a valid number.")


def get_lifter_info() -> tuple[str, str, str, float, str]:
    while True:
        user_sex = input("Enter your sex (M or F): ").strip().upper()
        if user_sex in {'M', 'F'}:
            break

        logger.info("Invalid sex entered.")
        continue
    
    while True:
        user_event = input('\n'.join([
            'Enter your event:',
            '    SBD: Squat, Bench, Deadlift',
            '     BD: Bench, Deadlift',
            '     SD: Squat, Deadlift',
            '     SB: Squat, Bench',
            '      S: Squat',
            '      B: Bench',
            '      D: Deadlift',
            '\n'
        ])).strip().upper()
        if user_event in EVENTS:
            break

        logger.info("Invalid event.")
        continue
    
    while True:
        try:
            user_bodyweight = get_float_input("Enter your body weight in kg: ")
            user_weight_class = assign_user_weight_class(user_sex, user_bodyweight)
            if user_weight_class != 'Unknown':
                break
            logger.info("No matching weight class found for your body weight.")
            continue
        except ValueError:
            logger.info("Please enter a valid number for body weight.")
            continue

    while True:
        compare_within_weight_class = input("Do you want to compare your lifts within your weight class? (yes/no): ").strip().upper()[0]
        if compare_within_weight_class in {'Y', 'N'}:
            break
        logger.info("Invalid answer given.")
        continue

    return user_sex, user_event, user_weight_class, user_bodyweight, compare_within_weight_class


def get_filter_expr(user_sex: Sex,
                   user_event: Event, 
                   user_weight_class: str, 
                   relevant_columns: tuple[str, ...], 
                   compare_within_weight_class: Compare) -> pl.Expr:
    x = (
        pl.col('BodyweightKg').is_not_null() &
        (pl.col('BodyweightKg') > 0) &
        (pl.col('Sex') == user_sex) &
        (pl.col('Event') == user_event) &
        pl.struct(relevant_columns).is_not_null()
    )
    match compare_within_weight_class:
        case 'Y':
            return (
                x &
                (pl.col('CustomWeightClass') == user_weight_class)
            )
        case 'N':
            return x


def load_and_preprocess_data(data: Path) -> pl.LazyFrame:
    valid_sexes = ["M", "F"]

    # Load the Parquet file as a LazyFrame
    if data.suffix == '.csv':
        df = pl.scan_csv(data)
    elif data.suffix == '.parquet':
        df = pl.scan_parquet(data)
    else:
        raise ValueError(f"Unsupported file format: {data.suffix}")

    if kg_columns := [
        col for col in df.collect_schema().names() if col.endswith('Kg')
    ]:
        df = df.with_columns(
            [pl.col(col).cast(pl.Float64) for col in kg_columns if col != 'WeightClassKg']
        )

    # Continue with your data preprocessing
    return (
        df
        .filter(pl.col('Sex').is_in(valid_sexes))
        .with_columns(custom_weight_class_expr())
    )

def pl_max(*args: str) -> list[pl.Expr]:
    return [pl.max(arg).alias(arg) for arg in args]

def process_data(data: Path,
                 user_sex: str,
                 user_event: str,
                 user_weight_class: str,
                 user_bodyweight: float,
                 compare_within_weight_class: str) -> PolarsDict:
    relevant_columns = EVENTS[user_event]

    df = load_and_preprocess_data(data)
    # If the user wants to compare within their weight class, add the weight class filter
    agg_exprs = pl_max('Total', *relevant_columns) + [pl.col('Sex').first(), pl.col('BodyweightKg').mean(), pl.col('CustomWeightClass').first()] if compare_within_weight_class == 'Y' else pl_max('Total', *relevant_columns) + [pl.col('Sex').first(), pl.col('BodyweightKg').mean()]

    df_filtered = df.filter(
        get_filter_expr(user_sex, user_event, user_weight_class, relevant_columns, compare_within_weight_class)
    ).filter(
        pl.col(col) > 0 for col in relevant_columns
    ).with_columns(
        pl.sum_horizontal(pl.col(col) for col in relevant_columns).alias('Total')
    ).filter(
        pl.col('Total') > 0
    ).group_by('Name').agg(agg_exprs)

    user_lift_inputs = {lift: get_float_input(f"Enter your best {lift.lower()} (kg): ") for lift in relevant_columns}

    return {
        'user_total': sum(user_lift_inputs.values()),
        'user_event': user_event,
        'user_sex': user_sex,
        'compare_within_weight_class': compare_within_weight_class,
        'user_weight_class': user_weight_class,
        'user_bodyweight': user_bodyweight,
        'data': df_filtered
    }


def get_stats(lift_values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    # Calculate mean and standard deviation
    mean = np.mean(lift_values)
    std_dev = np.std(lift_values)
    # Define the range for the normal distribution curve
    xmin, xmax = lift_values.min(), lift_values.max()
    linspace = np.linspace(xmin, xmax, 1000)

    # Calculate the normal distribution PDF
    pdf = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((linspace - mean) / std_dev) ** 2)
    return pdf * (xmax - xmin) / N_BINS * len(lift_values), linspace

def get_event(event: Event) -> str:
    match event:
        case Event.SBD:
            return 'Squat, Bench, Deadlift'
        case Event.BD:
            return 'Bench, Deadlift'
        case Event.SD:
            return 'Squat, Deadlift'
        case Event.SB:
            return 'Squat, Bench'
        case Event.S:
            return 'Squat'
        case Event.B:
            return 'Bench'
        case Event.D:
            return 'Deadlift'

def calculate_stats(df_filtered: pl.LazyFrame,
                   user_val: Union[int, float],
                   header: Literal['Total', 'WilksScore']) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, str]:
    lift_values = df_filtered.select(pl.col(header)).collect().to_numpy()
    pdf, linspace = get_stats(lift_values)
    percentile = (lift_values < user_val).sum() / len(lift_values) * 100
    return lift_values, pdf, linspace, f'{percentile:.2f}'

def plot_histogram(data: PolarsDict):
    lift_values, pdf, x_axis, percentile = calculate_stats(data['data'], data['user_total'], 'Total')
    
    plt.figure(figsize=(10, 6))
    plt.hist(lift_values, bins=N_BINS, alpha=0.7, label='Lifters')
    plt.axvline(data['user_total'], color='red', linestyle='dashed', linewidth=2, label='Your Lift')
    plt.annotate(f'Your Percentile: {percentile}%', xy=(data['user_total'], plt.ylim()[1]*0.9),
                 xytext=(data['user_total'], plt.ylim()[1]*0.9), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', ha='center')

    # Overlay the normal distribution curve
    plt.plot(x_axis, pdf, 'k', linewidth=2, label='Normal Distribution')

    plt.xlabel(f'{get_event(data['user_event'])} Total (kg)')
    plt.ylabel('Number of Lifters')
    plt.title(adjust_plot_title(
        f'Distribution of {get_event(data['user_event'])} Total for {SEXES[data['user_sex']]} lifters',
        data['user_weight_class'],
        data['compare_within_weight_class']
    ))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(data: PolarsDict):
    # Scatter plot of lift vs. bodyweight
    filtered = data['data'].filter(pl.col('Total') > 0).select(
        pl.col('BodyweightKg'),
        pl.col('Total')
    ).collect()

    n = min(filtered.shape[0], 10000)

    filtered = filtered.sample(n, with_replacement=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(filtered['BodyweightKg'], filtered['Total'], alpha=0.6, edgecolors='w', linewidths=0.5, label='Lifters')
    plt.plot(data['user_bodyweight'], data['user_total'], marker='o', markerfacecolor='red', label='Your Lift')
    plt.xlabel('Bodyweight (kg)')
    lift_label = 'Lift Total (kg)'
    plt.ylabel(lift_label)
    plt.title(adjust_plot_title(
        f'{lift_label} vs. Bodyweight for {SEXES[data['user_sex']]} lifters, Event: {get_event(data['user_event'])}',
        data['user_weight_class'],
        data['compare_within_weight_class']
    ))
    plt.legend()
    plt.grid(True)
    plt.show()

def adjust_plot_title(title: str, user_weight_class: str, compare_within_weight_class: Compare) -> str:
    match compare_within_weight_class:
        case Compare.YES:
            return f'{title} in Weight Class {user_weight_class}'
        case Compare.NO:
            return f'{title} across all weight classes'


def sum_coefficients(bw_col: CoeffSum, sex: str) -> CoeffSum:
    coefficients = WILKS_COEFFICIENTS[sex]
    return 600 / sum(coeff * bw_col ** i for i, coeff in enumerate(coefficients))

def wilks_coefficient_expr(sex_col: pl.Expr, bw_col: pl.Expr) -> pl.Expr:
    return pl.when(sex_col == 'M').then(
        sum_coefficients(bw_col, 'M')
    ).otherwise(
        sum_coefficients(bw_col, 'F')
    )

def plot_wilks(data: PolarsDict):
    # Add 'WilksScore' column to df_filtered
    df_filtered = data['data'].with_columns(
        (
             pl.col('Total') * wilks_coefficient_expr(pl.col('Sex'), pl.col('BodyweightKg'))
        ).alias('WilksScore')
    ).filter(
        pl.col('WilksScore') > 0
    )

    # Compute user's Wilks score
    if data['user_sex'] == 'M':
        user_coeff = (
            sum_coefficients(data['user_bodyweight'], 'M')
        )
    else:
        user_coeff = (
            sum_coefficients(data['user_bodyweight'], 'F')
        )

    user_wilks = data['user_total'] * user_coeff

    wilks_vals, pdf, x_axis, percentile = calculate_stats(df_filtered, data['user_total'] * user_coeff, 'WilksScore')

    # Histogram of Wilks Scores
    plt.figure(figsize=(10, 6))
    plt.hist(wilks_vals, bins=N_BINS, alpha=0.7, label='Lifters')
    plt.axvline(user_wilks, color='red', linestyle='dashed', linewidth=2, label='Your Wilks Score')
    plt.annotate(f'Your Percentile: {percentile}%', xy=(user_wilks, plt.ylim()[1]*0.9),
                 xytext=(user_wilks, plt.ylim()[1]*0.9), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', ha='center')
    
    # Overlay the normal distribution curve
    plt.plot(x_axis, pdf, 'k', linewidth=2, label='Normal Distribution')

    plt.xlabel('Wilks Score')
    plt.ylabel('Number of Lifters')
    title = adjust_plot_title(
        f'Distribution of Wilks Scores for {SEXES[data['user_sex']]} lifters, Event: {get_event(data['user_event'])}',
        data['user_weight_class'],
        data['compare_within_weight_class']
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
