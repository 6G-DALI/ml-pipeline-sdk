"""Submit the example to the active ZenML stack."""

import argparse

from zenml.config import DockerSettings

from examples.random_forest import random_forest_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--asset-id")
    parser.add_argument("--dataspace-model-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--runtime-image", help="Prebuilt Kubeflow image containing the SDK and example dependencies.")
    args = parser.parse_args()
    pipeline = random_forest_pipeline
    if args.runtime_image:
        pipeline = pipeline.with_options(
            settings={"docker": DockerSettings(parent_image=args.runtime_image, skip_build=True)}
        )
    pipeline.with_options(run_name=args.run_name)(
        dataset_id=args.dataset_id, asset_id=args.asset_id, dataspace_model_id=args.dataspace_model_id,
        run_name=args.run_name, experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
