from pykeen.datasets.inductive.base import DisjointInductivePathDataset

class InductiveLPDataset(DisjointInductivePathDataset):
    train_path = "./data/train.txt"
    inference_path = "./data/inference.txt"
    inference_val_path = "./data/inference_validation.txt"
    inference_test_path = "./data/inference_test.txt"

    def __init__(self, **kwargs):
        super().__init__(
            transductive_training_path=self.train_path,
            inductive_inference_path=self.inference_path,
            inductive_validation_path=self.inference_val_path,
            inductive_testing_path=self.inference_test_path,
            create_inverse_triples=True,
            eager=True,
            **kwargs
        )

