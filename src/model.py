from search import interactive_search as interactive_search_general
from ingest import main as data_prep
import argparse


#Could build timing collection here, priority after midterm?


def main():
    #gather necessary inputs
    parser = argparse.ArgumentParser(description="Run an interactive search")
    parser.add_argument("path", type=str, help="Data path")
    parser.add_argument("model", type=str, help = "General purpose search or trees?")
    args = parser.parse_args()
    #ingesting the data
    data_prep(args.path)

    #running the search function of choice
    interactive_search_general()


if __name__ == "__main__":
    main()
    

    