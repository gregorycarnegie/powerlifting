import utils as ut
from pathlib import Path


if __name__ == '__main__':
    data = Path("data/powerlifting_data.csv")
    
    parameters = ut.process_data(data)

    ut.plot_histogram(*parameters)
