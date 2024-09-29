import polars as pl

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
def get_weight_class_expr() -> pl.Expr:
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
def assign_user_weight_class(sex: str, bw: float) -> str:
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

def process_data(data: str):
    with open(data, 'r') as f:
        header = f.readline()
    column_names = header.strip().split(',')
    kg_columns = [col for col in column_names if col.endswith('Kg')]
    schema_overrides = {col: (pl.Float64 if col != 'WeightClassKg' else pl.String) for col in kg_columns}
    df = pl.read_csv(data, schema_overrides=schema_overrides)
    
    df = df.filter(pl.col('BodyweightKg').is_not_null() & (pl.col('BodyweightKg') > 0))
    df = df.filter(pl.col('Sex').is_in(['M', 'F']))
    df = df.with_columns(get_weight_class_expr())
    df = df.filter(pl.col('CustomWeightClass') != 'Unknown')

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
    if user_weight_class == 'Unknown':
        print("No matching weight class found for your body weight.")
        exit()

    compare_within_weight_class = input("Do you want to compare your lifts within your weight class? (yes/no): ").strip().lower()[0]
    relevant_columns = EVENTS.get(user_event)

    if not relevant_columns:
        print("No relevant columns found for this event.")
        exit()

    # If the user wants to compare within their weight class, add the weight class filter
    if compare_within_weight_class == 'y':
        df_filtered = df.filter(
            (pl.col('Sex') == user_sex) &
            (pl.col('Event') == user_event) &
            (pl.col('CustomWeightClass') == user_weight_class) &
            pl.struct(relevant_columns).is_not_null()
        ).with_columns(
            pl.sum_horizontal([pl.col(col) for col in relevant_columns]).alias('Total')
        )
    else:
        df_filtered = df.filter(
            (pl.col('Sex') == user_sex) &
            (pl.col('Event') == user_event) &
            pl.struct(relevant_columns).is_not_null()
        ).with_columns(
            pl.sum_horizontal([pl.col(col) for col in relevant_columns]).alias('Total')
        )

    lift_values = df_filtered['Total'].to_numpy()
    lift_values = lift_values[lift_values > 0]

    user_lift_inputs = {lift: get_float_input(f"Enter your best {lift.lower()} (kg): ") for lift in relevant_columns}
    user_total = sum(user_lift_inputs.values())

    percentile = df_filtered.filter(pl.col('Total') < user_total).shape[0] / df_filtered.shape[0] * 100

    return lift_values, user_total, percentile, user_event, user_sex, compare_within_weight_class, user_weight_class
