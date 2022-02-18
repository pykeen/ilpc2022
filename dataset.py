from pykeen.datasets.inductive.base import DisjointInductivePathDataset

class InductiveLPDataset(DisjointInductivePathDataset):
    root_path = "./data/"

    def __init__(self, size: str = "small", **kwargs):
        """
        :param size: "small" or "large"
        """
        super().__init__(
            transductive_training_path=self.root_path+f"{size}"+"/train.txt",
            inductive_inference_path=self.root_path+f"{size}"+"/inference.txt",
            inductive_validation_path=self.root_path+f"{size}"+"/inference_validation.txt",
            inductive_testing_path=self.root_path+f"{size}"+"/inference_test.txt",
            create_inverse_triples=True,
            eager=True,
            **kwargs
        )

