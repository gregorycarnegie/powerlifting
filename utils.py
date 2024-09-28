import polars as pl

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
