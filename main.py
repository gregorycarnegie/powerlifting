import matplotlib.pyplot as plt
import utils as ut


if __name__ == '__main__':
    data = "data/openpowerlifting-2024-09-21-f5f73164.csv"
    
    lift_values, user_total, percentile, user_event, user_sex, compare_within_weight_class, user_weight_class = ut.process_data(data)

    plt.figure(figsize=(10, 6))
    plt.hist(lift_values, bins=50, alpha=0.7, label='Lifters')
    plt.axvline(user_total, color='red', linestyle='dashed', linewidth=2, label='Your Lift')
    plt.annotate(f'Your Percentile: {percentile:.2f}%', xy=(user_total, plt.ylim()[1]*0.9),
                 xytext=(user_total, plt.ylim()[1]*0.9), arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, color='red', ha='center')

    plt.xlabel(f'{user_event} Total (kg)')
    plt.ylabel('Number of Lifters')
    title = f'Distribution of {user_event} Total for {user_sex} lifters'
    if compare_within_weight_class == 'y':
        title += f' in Weight Class {user_weight_class}'
    else:
        title += ' across all weight classes'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
