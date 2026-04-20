import sys

from system_level.forecasting.cli import main


if __name__ == "__main__":
    main(["run-all", *sys.argv[1:]])
