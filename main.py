from pathlib import Path

import utils as ut


if __name__ == '__main__':
    data = Path("data/openpowerlifting-2024-09-21-f5f73164.csv")
    
    parameters = ut.process_data(data)

    # ut.plot_histogram(parameters)

    # ut.plot_scatter(parameters)
    
    ut.plot_wilks(parameters)
