from pathlib import Path

from pykeen.datasets.inductive.base import DisjointInductivePathDataset
from typing_extensions import Literal

__all__ = [
    "InductiveLPDataset",
]

DATA = Path(__file__).parent.parent.resolve().join("data")

Size = Literal["small", "large"]


class InductiveLPDataset(DisjointInductivePathDataset):
    """An inductive link prediction dataset for the `KG Course Competition <https://github.com/migalkin/kgcourse2021>`_."""

    def __init__(self, size: Size = "small", **kwargs):
        """Initialize the inductive link prediction dataset.

        :param size: "small" or "large"
        :param kwargs: keyword arguments to forward to the base dataset class
        """
        super().__init__(
            transductive_training_path=DATA.joinpath(size, "train.txt"),
            inductive_inference_path=DATA.joinpath(size, "inference.txt"),
            inductive_validation_path=DATA.joinpath(size, "inference_validation.txt"),
            inductive_testing_path=DATA.joinpath(size, "inference_test.txt"),
            create_inverse_triples=True,
            eager=True,
            **kwargs
        )
