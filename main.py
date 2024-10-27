from pathlib import Path

import utils as ut

if __name__ == '__main__':
    data = Path("data/openpowerlifting-2024-09-21-f5f73164.parquet")
    # csv_to_parquet(data)
    user_sex, user_event, user_weight_class, user_bodyweight, compare_within_weight_class = ut.get_lifter_info()
    parameters = ut.process_data(data, user_sex, user_event, user_weight_class, user_bodyweight, compare_within_weight_class)

    ut.plot_histogram(parameters)

    ut.plot_scatter(parameters)
    
    ut.plot_wilks(parameters)
