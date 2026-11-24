"""Submit an example to the active ZenML stack."""

import argparse

from examples.random_forest import random_forest_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--dataspace-model-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-name", default="")
    args = parser.parse_args()
    random_forest_pipeline.with_options(run_name=args.run_name)(
        dataset_id=args.dataset_id, asset_id=args.asset_id, dataspace_model_id=args.dataspace_model_id,
        run_name=args.run_name, experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
