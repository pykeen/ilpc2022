import torch
import click

from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.losses import NSSALoss
from pykeen.stoppers import EarlyStopper
from pykeen.models.inductive import InductiveNodePiece, InductiveNodePieceGNN
from pykeen.trackers import WANDBResultTracker
from pykeen.typing import TRAINING, VALIDATION, TESTING

from torch.optim import Adam

from dataset import InductiveLPDataset

# fix the seed for reproducibility
torch.manual_seed(42)

@click.command()
@click.option('-dim', '--embedding_dim', type=int, default=100)
@click.option('-tokens', '--tokens_per_node', type=int, default=5)
@click.option('-lr', '--learning_rate', type=float, default=0.0005)
@click.option('-m', '--margin', type=float, default=15.0)
@click.option('-negs', '--num_negatives', type=int, default=4)
@click.option('-b', '--batch_size', type=int, default=256)
@click.option('-e', '--num_epochs', type=int, default=100)
@click.option('-wandb', '--wandb', type=bool, default=False)
@click.option('-save', '--save_model', type=bool, default=False)
@click.option('-gnn', '--gnn', type=bool, default=False)  # for the Inductive NodePiece GNN baseline
def main(
        embedding_dim: int,
        tokens_per_node: int,
        learning_rate: float,
        margin: float,
        num_negatives: int,
        batch_size: int,
        num_epochs: int,
        wandb: bool,
        save_model: bool,
        gnn: bool,
):

    # dataset loading
    dataset = InductiveLPDataset()
    loss = NSSALoss(margin=margin)

    # we have two baselines: InductiveNodePiece and InductiveNodePieceGNN
    # the GNN version uses a 2-layer CompGCN message passing encoder on the training / inference graphs
    # but feel free to create and attach your own GNN encoder via the gnn_encoder argument
    # and new inductive link prediction models in general
    model_cls = InductiveNodePieceGNN if gnn else InductiveNodePiece
    model = model_cls(
        embedding_dim=embedding_dim,
        triples_factory=dataset.transductive_training,
        inference_factory=dataset.inductive_inference,
        num_tokens=tokens_per_node,
        aggregation="mlp",
        loss=loss,
    )
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # default training regime is negative sampling (SLCWA)
    # you can also use the 1-N regime with the LCWATrainingLoop
    # the LCWA loop does not need negative sampling kwargs, but accepts label_smoothing in the .train() method
    training_loop = SLCWATrainingLoop(
        triples_factory=dataset.transductive_training,
        model=model,
        optimizer=optimizer,
        negative_sampler_kwargs=dict(num_negs_per_pos=num_negatives),  # affects training speed, the more - the better
        mode=TRAINING  # must be specified for the inductive setup
    )

    valid_evaluator = RankBasedEvaluator(mode=VALIDATION)
    test_evaluator = RankBasedEvaluator(mode=TESTING)

    if wandb:
        tracker = WANDBResultTracker(project="inductive_ilp", entity="pykeen")  # put here your project and entity
        tracker.start_run()
    else:
        tracker = None

    # we don't actually use the early stopper here by setting the patience to 100000
    early_stopper = EarlyStopper(
        model=model,
        relative_delta=0.0005,
        training_triples_factory=dataset.inductive_inference,
        evaluation_triples_factory=dataset.inductive_validation,
        frequency=1,
        patience=100000,
        result_tracker=tracker,
        evaluation_batch_size=batch_size,
        evaluator=valid_evaluator,
    )

    # model training and eval on validation starts here
    training_loop.train(
        triples_factory=dataset.transductive_training,
        stopper=early_stopper,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    # final eval on the test set
    result = test_evaluator.evaluate(
        model=model,
        mapped_triples=dataset.inductive_testing.mapped_triples,
        additional_filter_triples=[
            dataset.inductive_inference.mapped_triples,
            dataset.inductive_validation.mapped_triples
        ],  # filtering of other positive triples
        batch_size=batch_size,
    )

    # extracting final metrics
    results_dict = result.to_dict()
    print(f"Test MRR {results_dict['inverse_harmonic_mean_rank']['both']['realistic']:.5f}")
    for k in [10, 5, 3, 1]:
        print(f"Test Hits@{k} {results_dict['hits_at_k']['both']['realistic'][k]:.5f}")
    print(f"Test Arithmetic Mean Rank {results_dict['arithmetic_mean_rank']['both']['realistic']:.5f}")

    # you can also log the final results to wandb if you want
    if wandb:
        tracker.log_metrics(
            metrics=result.to_flat_dict(),
            step=num_epochs + 1,
            prefix='test',
        )

    # saving the final model
    if save_model:
        model.save_state("./data/model.pth")


if __name__ == "__main__":
    main()
