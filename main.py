import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import utils as ut

if __name__=='__main__':
    df = pl.read_csv("data/powerlifting_data.csv")

    # Filter out entries with missing or invalid 'BodyweightKg'
    df = df.filter(pl.col('BodyweightKg').is_not_null() & (pl.col('BodyweightKg') > 0))

    # Ignore lifters with 'Mx' as sex
    df = df.filter(pl.col('Sex').is_in(['M', 'F']))

    # Assign the custom weight classes using the generated expression
    df = df.with_columns(
        ut.get_weight_class_expr()
    )

    # Filter out rows with 'Unknown' weight class
    df = df.filter(pl.col('CustomWeightClass') != 'Unknown')

    # Ask user for input
    user_sex = input("Enter your sex (M or F): ").strip().upper()
    if user_sex not in {'M', 'F'}:
        print("Invalid sex entered.")
        exit()

    user_bodyweight = float(input("Enter your body weight in kg: "))

    print("\nSelect an event from the following options:")
    print("SBD (Full Power), BD (Bench-Deadlift), SD (Squat-Deadlift), SB (Squat-Bench), S (Squat-only), B (Bench-only), D (Deadlift-only)")
    user_event = input("Enter your event: ").strip().upper()

    if user_event not in {"SBD", "BD", "SD", "SB", "S", "B", "D"}:
        print("Invalid input. Please enter 'yes' or 'no'.")
        exit()

    user_weight_class = ut.assign_user_weight_class(user_sex, user_bodyweight)
    print(f"Your weight class: {user_weight_class}")

    if user_weight_class == 'Unknown':
        print("No matching weight class found for your body weight.")
        exit()

    # Ask the user if they want to compare within their weight class
    compare_within_weight_class = input("Do you want to compare your lifts within your weight class? (yes/no): ").strip().lower()[0]

    if compare_within_weight_class not in {'y', 'n'}:
        print("Invalid input. Please enter 'yes' or 'no'.")
        exit()

    # Depending on the event, ask for the relevant lifts
    if user_event == 'B':
        user_bench = float(input("Enter your best bench (kg): "))
        user_total = user_bench
    elif user_event == 'BD':
        user_bench = float(input("Enter your best bench (kg): "))
        user_deadlift = float(input("Enter your best deadlift (kg): "))
        user_total = user_bench + user_deadlift
    elif user_event == 'D':
        user_deadlift = float(input("Enter your best deadlift (kg): "))
        user_total = user_deadlift
    elif user_event == 'S':
        user_squat = float(input("Enter your best squat (kg): "))
        user_total = user_squat
    elif user_event == 'SB':
        user_squat = float(input("Enter your best squat (kg): "))
        user_bench = float(input("Enter your best bench (kg): "))
        user_total = user_squat + user_bench
    elif user_event == 'SBD':
        user_squat = float(input("Enter your best squat (kg): "))
        user_bench = float(input("Enter your best bench (kg): "))
        user_deadlift = float(input("Enter your best deadlift (kg): "))
        user_total = user_squat + user_bench + user_deadlift
    elif user_event == 'SD':
        user_squat = float(input("Enter your best squat (kg): "))
        user_deadlift = float(input("Enter your best deadlift (kg): "))
        user_total = user_squat + user_deadlift
    else:
        print("Invalid event selected.")
        exit()

    # Filter data for the user's sex and event
    df_filtered = df.filter(
        (pl.col('Sex') == user_sex) &
        (pl.col('Event') == user_event)
    )

    # If the user wants to compare within their weight class, add the weight class filter
    if compare_within_weight_class == 'y':
        df_filtered = df_filtered.filter(
            pl.col('CustomWeightClass') == user_weight_class
        )

    # Depending on the event, get the relevant lifts
    if user_event == 'SBD':
        df_filtered = df_filtered.filter(pl.col('TotalKg').is_not_null())
        lift_values = df_filtered['TotalKg'].to_numpy()
        lift_label = 'Total Lift (kg)'
        user_lift = user_total
    elif user_event == 'BD':
        df_filtered = df_filtered.filter(
            pl.col('Best3BenchKg').is_not_null() & pl.col('Best3DeadliftKg').is_not_null()
        )
        df_filtered = df_filtered.with_columns(
            (pl.col('Best3BenchKg') + pl.col('Best3DeadliftKg')).alias('TotalBD')
        )
        lift_values = df_filtered['TotalBD'].to_numpy()
        lift_label = 'Bench + Deadlift Total (kg)'
        user_lift = user_total
    elif user_event == 'SD':
        df_filtered = df_filtered.filter(
            pl.col('Best3SquatKg').is_not_null() & pl.col('Best3DeadliftKg').is_not_null()
        )
        df_filtered = df_filtered.with_columns(
            (pl.col('Best3SquatKg') + pl.col('Best3DeadliftKg')).alias('TotalSD')
        )
        lift_values = df_filtered['TotalSD'].to_numpy()
        lift_label = 'Squat + Deadlift Total (kg)'
        user_lift = user_total
    elif user_event == 'SB':
        df_filtered = df_filtered.filter(
            pl.col('Best3SquatKg').is_not_null() & pl.col('Best3BenchKg').is_not_null()
        )
        df_filtered = df_filtered.with_columns(
            (pl.col('Best3SquatKg') + pl.col('Best3BenchKg')).alias('TotalSB')
        )
        lift_values = df_filtered['TotalSB'].to_numpy()
        lift_label = 'Squat + Bench Total (kg)'
        user_lift = user_total
    elif user_event == 'S':
        df_filtered = df_filtered.filter(pl.col('Best3SquatKg').is_not_null())
        lift_values = df_filtered['Best3SquatKg'].to_numpy()
        lift_label = 'Best Squat (kg)'
        user_lift = user_total
    elif user_event == 'B':
        df_filtered = df_filtered.filter(pl.col('Best3BenchKg').is_not_null())
        lift_values = df_filtered['Best3BenchKg'].to_numpy()
        lift_label = 'Best Bench (kg)'
        user_lift = user_total
    elif user_event == 'D':
        df_filtered = df_filtered.filter(pl.col('Best3DeadliftKg').is_not_null())
        lift_values = df_filtered['Best3DeadliftKg'].to_numpy()
        lift_label = 'Best Deadlift (kg)'
        user_lift = user_total
    else:
        print("Invalid event.")
        exit()

    # Check if we have data to plot
    if len(lift_values) == 0:
        print("No data available for the selected event and criteria.")
        exit()

    # Calculate the percentile
    percentile = (np.sum(lift_values < user_lift) / len(lift_values)) * 100

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(lift_values, bins=50, alpha=0.7, label='Lifters')

    # Plot the user's lift
    plt.axvline(user_lift, color='red', linestyle='dashed', linewidth=2, label='Your Lift')

    # Add percentile annotation
    plt.annotate(f'Your Percentile: {percentile:.2f}%', xy=(user_lift, plt.ylim()[1]*0.9), xytext=(user_lift, plt.ylim()[1]*0.9),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, color='red', ha='center')

    plt.xlabel(lift_label)
    plt.ylabel('Number of Lifters')
    if compare_within_weight_class == 'y':
        plt.title(f'Distribution of {lift_label} for {user_sex} lifters in Weight Class {user_weight_class}, Event: {user_event}')
    else:
        plt.title(f'Distribution of {lift_label} for {user_sex} lifters across all weight classes, Event: {user_event}')
    plt.legend()
    plt.grid(True)
    plt.show()
