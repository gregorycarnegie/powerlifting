import logging
from typing import Literal, Union, Optional, TypeVar
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

from polars_dict import PolarsDict
from enums import Compare, Event, Sex


CoeffSum = TypeVar('CoeffSum', float, pl.Expr)

N_BINS = 50

# OLD coefficients
# m_wilks_coefficients = [-216.0475144, 16.2606339, -0.002388645, -0.00113732, 7.01863e-06, -1.291e-08]
# f_wilks_coefficients = [594.31747775582, -27.23842536447, 0.82112226871, -0.00930733913, 4.731582e-05, -9.054e-08]

# 2020 coefficients
m_wilks_coefficients = [47.46178854, 8.472061379, 0.07369410346, -0.001395833811, 7.07665973070743e-6, -1.20804336482315e-8]
f_wilks_coefficients = [-125.4255398, 13.71219419, -0.03307250631, -0.001050400051, 9.38773881462799e-6, -2.3334613884954e-8]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


EVENTS: dict[Event, tuple[str, ...]] = {
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

def add_weight_conditions(sex: Sex,
                          weight_classes: list[tuple[int, int, str]],
                          conditions: Optional[pl.Expr] = None):
    for min_wt, max_wt, label in weight_classes:
        condition = (
            (pl.col('Sex') == sex) &
            (pl.col('BodyweightKg') > min_wt) &
            (pl.col('BodyweightKg') <= max_wt)
        )
        if conditions is None:
            conditions = pl.when(condition).then(pl.lit(label))
        else:
            conditions = conditions.when(condition).then(pl.lit(label))
    return conditions

def custom_weight_class_expr() -> pl.Expr:
    # Build the conditions for both male and female weight classes
    # Add conditions for male weight classes
    conditions = add_weight_conditions('M', male_weight_classes)
    
    # Add conditions for female weight classes
    conditions = add_weight_conditions('F', female_weight_classes, conditions)
    
    # Add default case
    return conditions.otherwise(pl.lit('Unknown')).alias('CustomWeightClass')


# Assign custom weight classes based on 'BodyweightKg' and 'Sex'
def assign_user_weight_class(sex: Sex, bw: float) -> str:
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
        except ValueError as e:
            logger.info(f"{e}: Invalid input. Please enter a valid number.")


def get_schema_overrides(data:Path) -> dict[str, Union[pl.Float64, pl.String]]:
    with open(data, 'r') as f:
        header = f.readline()
    column_names = header.strip().split(',')
    kg_columns = (col for col in column_names if col.endswith('Kg'))
    return {col: (pl.Float64 if col != 'WeightClassKg' else pl.String) for col in kg_columns}


def get_lifter_info() -> tuple[str, str, float]:
    user_sex = input("Enter your sex (M or F): ").strip().upper()
    if user_sex not in {'M', 'F'}:
        logger.info("Invalid sex entered.")
        exit()
    
    user_event = input(
        '\n'.join([
            'Enter your event:',
            '    SBD: Squat, Bench, Deadlift',
            '     BD: Bench, Deadlift',
            '     SD: Squat, Deadlift',
            '     SB: Squat, Bench',
            '      S: Squat',
            '      B: Bench',
            '      D: Deadlift',
            '\n'
        ])
    ).strip().upper()
    if user_event not in EVENTS:
        logger.info("Invalid event.")
        exit()

    user_bodyweight = get_float_input("Enter your body weight in kg: ")
    user_weight_class = assign_user_weight_class(user_sex, user_bodyweight)

    return user_sex, user_event, user_weight_class, user_bodyweight


def get_filter_expr(user_sex: Sex,
                   user_event: Event, 
                   user_weight_class: str, 
                   relevant_columns: tuple[str, ...], 
                   compare_within_weight_class: Compare) -> pl.Expr:
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


def process_data(data: Path) -> PolarsDict:
    schema_overrides = get_schema_overrides(data)

    df = pl.read_csv(data, schema_overrides=schema_overrides)

    df = df.with_columns(custom_weight_class_expr())

    user_sex, user_event, user_weight_class, user_bodyweight = get_lifter_info()

    if user_weight_class == 'Unknown':
        logger.info("No matching weight class found for your body weight.")
        exit()

    compare_within_weight_class = input("Do you want to compare your lifts within your weight class? (yes/no): ").strip().upper()[0]
    relevant_columns = EVENTS[user_event]

    if not relevant_columns:
        logger.info("No relevant columns found for this event.")
        exit()

    # If the user wants to compare within their weight class, add the weight class filter
    df_filtered = df.filter(
        get_filter_expr(user_sex, user_event, user_weight_class, relevant_columns, compare_within_weight_class)
    ).with_columns(
        pl.sum_horizontal(pl.col(col) for col in relevant_columns).alias('Total')
    )

    df_filtered = df_filtered.filter(
        pl.col('Total') > 0
    )

    user_lift_inputs = {lift: get_float_input(f"Enter your best {lift.lower()} (kg): ") for lift in relevant_columns}
    user_total = sum(user_lift_inputs.values())

    return {
        'lift_values': df_filtered['Total'].to_numpy(),
        'user_total': user_total,
        'percentile': get_percentile(df_filtered, user_total, 'Total'),
        'user_event': user_event,
        'user_sex': user_sex,
        'compare_within_weight_class': compare_within_weight_class,
        'user_weight_class': user_weight_class,
        'user_bodyweight': user_bodyweight,
        'df_filtered': df_filtered
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

def get_percentile(df_filtered: pl.DataFrame,
                   user_val: Union[int, float],
                   header: Literal['Total', 'WilksScore']) -> str:
    result = df_filtered.filter(pl.col(header) < user_val).shape[0] / df_filtered.shape[0] * 100
    return f'{result:.2f}'

# def setup_plot(title: str, xlabel: str, ylabel: str):
#     plt.figure(figsize=(10, 6))
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)


def plot_histogram(data: PolarsDict):
    
    pdf, x_axis = get_stats(data['lift_values'])

    percentile = get_percentile(data['df_filtered'], data['user_total'], 'Total')
    
    plt.figure(figsize=(10, 6))
    plt.hist(data['lift_values'], bins=N_BINS, alpha=0.7, label='Lifters')
    plt.axvline(data['user_total'], color='red', linestyle='dashed', linewidth=2, label='Your Lift')
    plt.annotate(f'Your Percentile: {percentile}%', xy=(data['user_total'], plt.ylim()[1]*0.9),
                 xytext=(data['user_total'], plt.ylim()[1]*0.9), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', ha='center')

    # Overlay the normal distribution curve
    plt.plot(x_axis, pdf, 'k', linewidth=2, label='Normal Distribution')

    plt.xlabel(f'{get_event(data['user_event'])} Total (kg)')
    plt.ylabel('Number of Lifters')
    title = adjust_plot_title(
        f'Distribution of {get_event(data['user_event'])} Total for {data['user_sex']} lifters',
        data['user_weight_class'],
        data['compare_within_weight_class']
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(data: PolarsDict):
    # Scatter plot of lift vs. bodyweight
    filtered = data['df_filtered'].filter(pl.col('Total') > 0)
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered['BodyweightKg'], data['lift_values'], alpha=0.6, edgecolors='w', linewidths=0.5, label='Lifters')
    plt.plot(data['user_bodyweight'], data['user_total'], marker='o', markerfacecolor='red', label='Your Lift')
    plt.xlabel('Bodyweight (kg)')
    lift_label = 'Lift Total (kg)'
    title = adjust_plot_title(
        f'{lift_label} vs. Bodyweight for {data['user_sex']} lifters, Event: {get_event(data['user_event'])}',
        data['user_weight_class'],
        data['compare_within_weight_class']
    )
    plt.ylabel(lift_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def adjust_plot_title(title: str, user_weight_class: str, compare_within_weight_class: Compare) -> str:
    match compare_within_weight_class:
        case Compare.YES:
            return f'{title} in Weight Class {user_weight_class}'
        case Compare.NO:
            return f'{title} across all weight classes'


def sum_coefficients(bw_col: CoeffSum, coefficients: list[float]) -> CoeffSum:
    return 600 / sum(coeff  * bw_col ** i for i, coeff  in enumerate(coefficients))
    

# Function to compute the Wilks coefficient expression
def wilks_coefficient_expr(sex_col: pl.Expr, bw_col: pl.Expr) -> pl.Expr:
    return pl.when(sex_col == 'M').then(
        sum_coefficients(bw_col, m_wilks_coefficients)
    ).otherwise(
        sum_coefficients(bw_col, f_wilks_coefficients)
    )

def plot_wilks(data: PolarsDict):
    # Add 'WilksScore' column to df_filtered
    df_filtered = data['df_filtered'].with_columns(
        (
             pl.col('Total') * wilks_coefficient_expr(pl.col('Sex'), pl.col('BodyweightKg'))
        ).alias('WilksScore')
    )

    df_filtered = df_filtered.filter(
        pl.col('WilksScore') > 0
    )

    wilks_vals = df_filtered['WilksScore'].to_numpy()

    
    pdf, x_axis = get_stats(wilks_vals)

    # Compute user's Wilks score
    if data['user_sex'] == 'M':
        user_coeff = (
            sum_coefficients(data['user_bodyweight'], m_wilks_coefficients)
        )
    else:
        user_coeff = (
            sum_coefficients(data['user_bodyweight'], f_wilks_coefficients)
        )

    user_wilks = data['user_total'] * user_coeff

    percentile = get_percentile(df_filtered, user_wilks, 'WilksScore')

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
        f'Distribution of Wilks Scores for {data['user_sex']} lifters, Event: {get_event(data['user_event'])}',
        data['user_weight_class'],
        data['compare_within_weight_class']
    )
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
