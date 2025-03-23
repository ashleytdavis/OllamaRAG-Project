from search import interactive_search as interactive_search_general
from trees import interactive_search as interactive_search_trees
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
    if args.model == 'general':
        interactive_search_general()
    elif args.model == 'trees':
        interactive_search_trees()
    else:
        print("That was not a possible model")


if __name__ == "__main__":
    main()
    

    