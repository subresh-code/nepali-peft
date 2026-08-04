"""Dataset loading for the three publicly released NLUE tasks.

Column names and split sizes below were verified empirically on
31 Jul 2026 via load_dataset against the canonical IRIIS-RESEARCH
repos (sentiment 65,106/16,279; acceptability 7,799/1,950;
coreference 564/142; train+test only, no validation split ships).

Design decisions:
- SPLIT_SEED is FIXED and separate from the training seed: every run
  in the 72-cell grid must see identical train/val partitions, or
  cross-condition comparisons are invalid. Only training randomness
  (init, shuffling, dropout) varies with the per-run seed.
- Validation = stratified 10% of the released train split, exactly as
  stated in the proposal.
"""

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset

SPLIT_SEED = 1234  # fixed forever; do NOT tie to the run seed
VAL_FRACTION = 0.10


@dataclass(frozen=True)
class TaskSpec:
    name: str
    repo: str
    text_fields: tuple          # 1 field = single sentence, 2 = pair
    label_field: str
    num_labels: int
    max_length: int
    report_mcc: bool            # MCC additionally for acceptability


TASKS = {
    "sentiment": TaskSpec(
        name="sentiment",
        repo="IRIIS-RESEARCH/Sentiment-Analysis-Nepali",
        text_fields=("sentences",),
        label_field="sentiment",
        num_labels=2,
        max_length=128,
        report_mcc=False,
    ),
    "acceptability": TaskSpec(
        name="acceptability",
        repo="IRIIS-RESEARCH/CoLA-Nepali",
        text_fields=("sentence",),
        label_field="label",
        num_labels=2,
        max_length=128,
        report_mcc=True,
    ),
    "coreference": TaskSpec(
        name="coreference",
        repo="IRIIS-RESEARCH/Co-Reference-Nepali",
        text_fields=("sentence1", "sentence2"),
        label_field="label",
        num_labels=2,
        max_length=256,  # passages are long (verified via examples)
        report_mcc=False,
    ),
}


def load_task(task_name: str) -> tuple:
    """Return (DatasetDict with train/validation/test, TaskSpec).

    The label column is renamed to 'labels' (what HF Trainer expects)
    and cast to ClassLabel so the validation split can stratify.
    """
    spec = TASKS[task_name]
    raw = load_dataset(spec.repo)

    # Normalise the label column name across tasks.
    if spec.label_field != "labels":
        raw = raw.rename_column(spec.label_field, "labels")

    # ClassLabel cast enables stratified splitting.
    raw = raw.class_encode_column("labels")

    split = raw["train"].train_test_split(
        test_size=VAL_FRACTION,
        stratify_by_column="labels",
        seed=SPLIT_SEED,
    )
    ds = DatasetDict(
        train=split["train"],
        validation=split["test"],
        test=raw["test"],
    )
    return ds, spec


def tokenize(ds: DatasetDict, spec: TaskSpec, tokenizer) -> DatasetDict:
    """Tokenize single-sentence or sentence-pair input per the task."""

    def _tok(batch):
        if len(spec.text_fields) == 1:
            return tokenizer(
                batch[spec.text_fields[0]],
                truncation=True,
                max_length=spec.max_length,
            )
        return tokenizer(
            batch[spec.text_fields[0]],
            batch[spec.text_fields[1]],
            truncation=True,
            max_length=spec.max_length,
        )

    remove = [c for c in ds["train"].column_names if c != "labels"]
    return ds.map(_tok, batched=True, remove_columns=remove)
