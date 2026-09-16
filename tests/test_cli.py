"""End-to-end tests for the Typer CLI, covering all four commands:
download-data, train, evaluate, and predict.

There is no config file or environment variable layer in this project --
every parameter is a CLI option declared directly in `cli.py`, with its
default declared there too. These tests exercise that surface directly
(passing flags, or relying on declared defaults) rather than via env vars.

These tests use the real `DigitClassifier` throughout -- it's small
enough (a couple of conv layers) to train in a fraction of a second on
the tiny curated sets used here, so there's no need for a stand-in model.
The one thing still patched is `DigitClassifier.load_from_checkpoint` for
`predict`/`evaluate` tests, since those pass a `--checkpoint-path` that
doesn't correspond to a real saved file -- the patch just constructs a
fresh (untrained) instance instead of reading one from disk.
"""

import glob
import json
import os

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from digit_classification import cli, evaluation
from digit_classification.data import SelectionStrategy
from digit_classification.model import DigitClassifier

runner = CliRunner()


def _normalized(text: str) -> str:
    """Strip Rich's box-drawing borders and collapse whitespace, so substring
    assertions survive panel word-wrapping and border characters."""
    stripped = text.replace("│", " ").replace("╭", " ").replace("╮", " ")
    stripped = stripped.replace("╰", " ").replace("╯", " ").replace("─", " ")
    return " ".join(stripped.lower().split())


def _parse_json_output(output: str) -> dict:
    """Parse the JSON object `predict` printed, ignoring any lines before it
    (e.g. the real model's `forward`/`training_step` timing prints)."""
    json_start = output.index("{")
    return json.loads(output[json_start:])


def _use_fresh_model_instead_of_checkpoint_file(monkeypatch):
    """Patch DigitClassifier.load_from_checkpoint to skip reading an actual
    checkpoint file and just construct a fresh instance -- for tests that
    pass a `--checkpoint-path` with no real file behind it."""
    monkeypatch.setattr(
        DigitClassifier,
        "load_from_checkpoint",
        classmethod(lambda cls, *a, **kw: cls()),
    )


class _CallTracker:
    """A stand-in function for `monkeypatch.setattr` that just remembers
    whether it was called -- for tests proving some function was, or
    wasn't, invoked at all. Accepts any arguments and ignores them, since
    it's standing in for functions with different real signatures.

    Pass `wraps=<the real function>` to also forward the call through to
    it (so the real behavior still happens, in addition to being tracked).
    """

    def __init__(self, wraps=None):
        self.was_called = False
        self._wraps = wraps

    def __call__(self, *args, **kwargs):
        self.was_called = True
        if self._wraps is not None:
            return self._wraps(*args, **kwargs)


def _patch_download_mnist(monkeypatch, sample, module=cli):
    """Patch `module.download_mnist` to return `sample` instead of
    downloading anything."""
    monkeypatch.setattr(module, "download_mnist", lambda data_dir: sample)


def _track_prepare_datasets_after_patching_download(monkeypatch, sample):
    """For tests asserting the pipeline was never reached because of an
    invalid argument: patch `download_mnist` (so nothing tries to hit the
    network) and replace `prepare_datasets` with a call-tracker. Returns
    the tracker -- check `tracker.was_called` after invoking the CLI."""
    _patch_download_mnist(monkeypatch, sample)
    tracker = _CallTracker()
    monkeypatch.setattr(cli, "prepare_datasets", tracker)
    return tracker


def _spy_on_call_kwargs(monkeypatch, obj, attr_name, *kwarg_names):
    """Patch `obj.attr_name` (a callable) to record the given keyword
    arguments it's called with, while still delegating to the real
    implementation. Returns the dict values get recorded into."""
    real_callable = getattr(obj, attr_name)
    captured = {}

    def spy(*args, **kwargs):
        for name in kwarg_names:
            captured[name] = kwargs.get(name)
        return real_callable(*args, **kwargs)

    monkeypatch.setattr(obj, attr_name, spy)
    return captured


def _data_dir(tmp_path):
    """The --data-dir path most CLI tests pass, under the test's own
    tmp dir."""
    return os.path.join(str(tmp_path), "data")


def _output_dir(tmp_path):
    """The --output-dir path most CLI tests pass, under the test's
    own tmp dir."""
    return os.path.join(str(tmp_path), "outputs")


class TestDownloadData:
    def test_invokes_mnist_download_with_given_dir(
        self, monkeypatch, tmp_path
    ) -> None:
        """download-data forwards --data-dir straight through to
        download_mnist."""
        calls = {}

        def fake_download(data_dir):
            calls["data_dir"] = data_dir
            return object()

        monkeypatch.setattr(cli, "download_mnist", fake_download)

        result = runner.invoke(
            cli.app, ["download-data", "--data-dir", str(tmp_path)]
        )

        assert result.exit_code == 0
        assert calls["data_dir"] == str(tmp_path)
        assert str(tmp_path) in result.output

    def test_uses_declared_default_when_flag_omitted(
        self, monkeypatch
    ) -> None:
        """download-data falls back to its declared default when
        --data-dir is omitted."""
        calls = {}
        monkeypatch.setattr(
            cli,
            "download_mnist",
            lambda data_dir: calls.setdefault("data_dir", data_dir),
        )

        result = runner.invoke(cli.app, ["download-data"])

        assert result.exit_code == 0
        assert calls["data_dir"] == "./data"


class TestTrain:
    def test_runs_end_to_end_and_writes_a_checkpoint(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A full train run succeeds and writes a checkpoint file
        under --output-dir."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        output_dir = _output_dir(tmp_path)
        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                output_dir,
                "--class-counts",
                "0=20,5=20,8=20",
                "--epochs",
                "1",
                "--batch-size",
                "4",
            ],
        )

        assert result.exit_code == 0, result.output
        assert os.path.exists(output_dir)
        assert glob.glob(os.path.join(output_dir, "*.ckpt"))
        assert "Training complete" in result.output

    def test_val_split_flag_reaches_prepare_datasets(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """--val-split is forwarded to prepare_datasets unchanged."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "prepare_datasets", "val_split"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=20,5=20,8=20",
                "--epochs",
                "1",
                "--val-split",
                "0.25",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["val_split"] == 0.25

    def test_omitting_val_split_defaults_to_point_fifteen(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """Omitting --val-split defaults to 0.15."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "prepare_datasets", "val_split"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=20,5=20,8=20",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["val_split"] == 0.15

    def test_val_split_outside_range_is_rejected(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """An out-of-range --val-split is rejected before
        prepare_datasets ever runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--val-split",
                "1.5",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_selection_strategy_flag_reaches_prepare_datasets(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """--selection-strategy is forwarded to prepare_datasets unchanged."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "prepare_datasets", "selection_strategy"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=15,5=15,8=15",
                "--epochs",
                "1",
                "--selection-strategy",
                "prototype",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["selection_strategy"] == SelectionStrategy.PROTOTYPE

    def test_omitting_selection_strategy_defaults_to_random(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """Omitting --selection-strategy defaults to
        SelectionStrategy.RANDOM."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "prepare_datasets", "selection_strategy"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=15,5=15,8=15",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["selection_strategy"] == SelectionStrategy.RANDOM

    def test_rejects_invalid_selection_strategy_before_running_the_pipeline(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """An invalid --selection-strategy value is rejected before
        prepare_datasets ever runs."""
        # An invalid choice must be rejected by the CLI's own argument
        # parsing -- prepare_datasets should never even be called.
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--selection-strategy",
                "bogus",
            ],
        )

        assert result.exit_code != 0
        assert "bogus" in _normalized(result.output)
        assert not prepare_datasets_tracker.was_called

    def test_balance_augment_flag_triggers_balancing(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """--balance-augment causes balance_with_augmentation to
        actually be called."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        balance_tracker = _CallTracker(wraps=cli.balance_with_augmentation)
        monkeypatch.setattr(cli, "balance_with_augmentation", balance_tracker)

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=15,5=15,8=15",
                "--epochs",
                "1",
                "--balance-augment",
            ],
        )

        assert result.exit_code == 0, result.output
        assert balance_tracker.was_called

    def test_omitting_balance_augment_skips_balancing(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """Omitting --balance-augment means balance_with_augmentation
        is never called."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        balance_tracker = _CallTracker()
        monkeypatch.setattr(cli, "balance_with_augmentation", balance_tracker)

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=15,5=15,8=15",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert not balance_tracker.was_called

    def test_epochs_flag_is_passed_to_trainer(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """--epochs is forwarded to the Lightning Trainer's max_epochs."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "Trainer", "max_epochs"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=12,5=12,8=12",
                "--epochs",
                "3",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["max_epochs"] == 3

    def test_loss_function_focal_alpha_and_gamma_reach_the_model(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """--loss-function/--focal-alpha/--focal-gamma are forwarded
        to DigitClassifier unchanged."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch,
            cli,
            "DigitClassifier",
            "loss_function",
            "focal_alpha",
            "focal_gamma",
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=12,5=12,8=12",
                "--epochs",
                "1",
                "--loss-function",
                "1",
                "--focal-alpha",
                "0.6",
                "--focal-gamma",
                "3.5",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["loss_function"] == 1
        assert captured["focal_alpha"] == 0.6
        assert captured["focal_gamma"] == 3.5

    def test_loss_function_focal_alpha_and_gamma_default(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """Omitting the loss-function flags falls back to their
        declared defaults."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch,
            cli,
            "DigitClassifier",
            "loss_function",
            "focal_alpha",
            "focal_gamma",
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=12,5=12,8=12",
                "--epochs",
                "1",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["loss_function"] == 0
        assert captured["focal_alpha"] == 0.2
        assert captured["focal_gamma"] == 2.0

    def test_loss_function_outside_0_1_is_rejected(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --loss-function value outside 0/1 is rejected before the
        pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--loss-function",
                "2",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_focal_alpha_outside_range_is_rejected(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --focal-alpha value outside 0.0-1.0 is rejected before
        the pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--focal-alpha",
                "1.5",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_focal_gamma_outside_range_is_rejected(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --focal-gamma value outside 0.0-10.0 is rejected before
        the pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--focal-gamma",
                "-1",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_omitting_epochs_defaults_to_fifteen(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """Omitting --epochs defaults to 15."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        captured = _spy_on_call_kwargs(
            monkeypatch, cli, "Trainer", "max_epochs"
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=12,5=12,8=12",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["max_epochs"] == 15

    def test_epochs_above_range_is_rejected_before_running_the_pipeline(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """An --epochs value above 20 is rejected before the pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--epochs",
                "21",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_epochs_below_range_is_rejected_before_running_the_pipeline(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """An --epochs value below 1 is rejected before the pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--epochs",
                "0",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_batch_size_out_of_range_is_rejected(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --batch-size outside 1-512 is rejected before the
        pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--batch-size",
                "0",
            ],
        )

        assert result.exit_code != 0
        assert not prepare_datasets_tracker.was_called

    def test_rejects_malformed_class_counts_before_running_the_pipeline(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A malformed --class-counts value is rejected before the
        pipeline runs."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0,5,8",  # missing "=count" per entry
            ],
        )

        assert result.exit_code != 0
        assert "class-counts" in _normalized(result.output)
        assert not prepare_datasets_tracker.was_called

    def test_rejects_duplicate_class_count_labels(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --class-counts value with a duplicate digit label is rejected."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=100,0=200",
            ],
        )

        assert result.exit_code != 0
        assert "duplicate" in _normalized(result.output)
        assert not prepare_datasets_tracker.was_called

    def test_rejects_class_count_label_outside_digit_range(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --class-counts label outside 0-9 is rejected."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "15=100",
            ],
        )

        assert result.exit_code != 0
        assert "not a valid mnist digit" in _normalized(result.output)
        assert not prepare_datasets_tracker.was_called

    def test_rejects_non_positive_class_count(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """A --class-counts count of zero or less is rejected."""
        prepare_datasets_tracker = (
            _track_prepare_datasets_after_patching_download(
                monkeypatch, real_mnist_sample
            )
        )

        result = runner.invoke(
            cli.app,
            [
                "train",
                "--data-dir",
                _data_dir(tmp_path),
                "--output-dir",
                _output_dir(tmp_path),
                "--class-counts",
                "0=0",
            ],
        )

        assert result.exit_code != 0
        assert "positive integer" in _normalized(result.output)
        assert not prepare_datasets_tracker.was_called


class TestEvaluate:
    def test_prints_classification_report(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """evaluate prints a classification_report covering every
        curated class."""
        monkeypatch.setattr(
            evaluation, "load_model", lambda checkpoint_path: DigitClassifier()
        )
        _patch_download_mnist(
            monkeypatch, real_mnist_sample, module=evaluation
        )

        result = runner.invoke(
            cli.app,
            [
                "evaluate",
                "--checkpoint-path",
                "unused.ckpt",
                "--data-dir",
                _data_dir(tmp_path),
                "--class-counts",
                "0=20,5=20,8=20",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "precision" in result.output
        assert "recall" in result.output

    def test_missing_checkpoint_path_option_fails_with_usage_error(
        self,
    ) -> None:
        """Omitting the required --checkpoint-path fails with a usage error."""
        result = runner.invoke(cli.app, ["evaluate", "--data-dir", "some/dir"])
        assert result.exit_code != 0
        assert "checkpoint-path" in _normalized(
            result.output
        ) or "missing" in _normalized(result.output)

    def test_rejects_invalid_selection_strategy(self) -> None:
        """An invalid --selection-strategy value is rejected."""
        result = runner.invoke(
            cli.app,
            [
                "evaluate",
                "--checkpoint-path",
                "unused.ckpt",
                "--selection-strategy",
                "bogus",
            ],
        )
        assert result.exit_code != 0
        assert "bogus" in _normalized(result.output)

    def test_rejects_malformed_class_counts(self) -> None:
        """A malformed --class-counts value is rejected."""
        result = runner.invoke(
            cli.app,
            [
                "evaluate",
                "--checkpoint-path",
                "unused.ckpt",
                "--class-counts",
                "not-valid",
            ],
        )
        assert result.exit_code != 0
        assert "class-counts" in _normalized(result.output)


class TestPredict:
    def test_outputs_probabilities_for_each_curated_class(
        self, monkeypatch, tmp_path
    ) -> None:
        """predict prints a JSON object with a probability for each
        curated digit."""
        _use_fresh_model_instead_of_checkpoint_file(monkeypatch)
        image_path = os.path.join(str(tmp_path), "digit.png")
        Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(image_path)

        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                image_path,
            ],
        )

        assert result.exit_code == 0, result.output
        probabilities = _parse_json_output(result.output)
        assert set(probabilities.keys()) == {"0", "5", "8"}
        assert abs(sum(probabilities.values()) - 1.0) < 1e-4

    def test_custom_classes_flag_changes_output_labels(
        self, monkeypatch, tmp_path
    ) -> None:
        """--classes changes which digit labels the JSON output is keyed by."""
        _use_fresh_model_instead_of_checkpoint_file(monkeypatch)

        image_path = os.path.join(str(tmp_path), "digit.png")
        Image.fromarray(np.zeros((28, 28), dtype=np.uint8)).save(image_path)

        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                image_path,
                "--classes",
                "1,2,3",
            ],
        )

        assert result.exit_code == 0, result.output
        probabilities = _parse_json_output(result.output)
        assert set(probabilities.keys()) == {"1", "2", "3"}

    def test_rejects_non_integer_classes(self, tmp_path) -> None:
        """A --classes value containing a non-integer is rejected."""
        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                os.path.join(str(tmp_path), "digit.png"),
                "--classes",
                "a,b,c",
            ],
        )

        assert result.exit_code != 0
        assert "classes" in _normalized(result.output)

    def test_rejects_duplicate_classes(self, tmp_path) -> None:
        """A --classes value with a duplicate digit is rejected."""
        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                os.path.join(str(tmp_path), "digit.png"),
                "--classes",
                "0,0,5",
            ],
        )

        assert result.exit_code != 0
        assert "duplicate" in _normalized(result.output)

    def test_rejects_classes_outside_digit_range(self, tmp_path) -> None:
        """A --classes value outside 0-9 is rejected."""
        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                os.path.join(str(tmp_path), "digit.png"),
                "--classes",
                "10,5,8",
            ],
        )

        assert result.exit_code != 0
        assert "not a valid mnist digit" in _normalized(result.output)

    def test_missing_image_file_fails_clearly(self, tmp_path) -> None:
        """A --input-path pointing at a nonexistent file fails with a
        clear message."""
        result = runner.invoke(
            cli.app,
            [
                "predict",
                "--checkpoint-path",
                "unused.ckpt",
                "--input-path",
                os.path.join(str(tmp_path), "does-not-exist.png"),
            ],
        )

        assert result.exit_code != 0
        assert "no such image file" in _normalized(result.output)

    def test_missing_required_options_fails_with_usage_error(self) -> None:
        """Omitting the required options fails with a usage error."""
        result = runner.invoke(cli.app, ["predict"])
        assert result.exit_code != 0


class TestReviewAugmentations:
    def test_saves_one_file_per_example(
        self, monkeypatch, tmp_path, real_mnist_sample
    ) -> None:
        """review-augmentations saves exactly one file per requested
        example."""
        _patch_download_mnist(monkeypatch, real_mnist_sample)

        output_dir = os.path.join(str(tmp_path), "review")
        result = runner.invoke(
            cli.app,
            [
                "review-augmentations",
                "--data-dir",
                _data_dir(tmp_path),
                "--digit",
                "0",
                "--num-examples",
                "3",
                "--output-dir",
                output_dir,
            ],
        )

        assert result.exit_code == 0, result.output
        saved_files = glob.glob(os.path.join(output_dir, "*.png"))
        assert len(saved_files) == 3
        assert "Saved 3 before/after example" in result.output

    def test_digit_outside_range_is_rejected(self) -> None:
        """A --digit value outside 0-9 is rejected."""
        result = runner.invoke(
            cli.app, ["review-augmentations", "--digit", "10"]
        )
        assert result.exit_code != 0

    def test_num_examples_outside_range_is_rejected(self) -> None:
        """A --num-examples value outside 1-10 is rejected."""
        result = runner.invoke(
            cli.app, ["review-augmentations", "--num-examples", "0"]
        )
        assert result.exit_code != 0
