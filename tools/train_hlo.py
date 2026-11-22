import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from hlo_tracking import Tracking


REPO = "https://github.com/asgmodel/recommendation-system.git"
SNAPSHOT_COMMIT = "94e208e9d88e511cf22e1f72b06737afdb973a8d"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def query_metrics(directory):
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return {key: metrics[key] for key in ("n", "answer_f1", "valid_rate")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default=SNAPSHOT_COMMIT)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--from-github", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--updated-zip", type=Path)
    parser.add_argument("--output", type=Path, default=Path("hlo-runs"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "hlo-recommendation"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()
    if args.wandb_mode not in {"online", "offline", "disabled"}:
        parser.error("WANDB_MODE must be online, offline, or disabled")

    source = args.project.resolve(strict=True)
    if args.output.resolve() == source:
        parser.error("--output must be a dedicated directory")
    if args.updated_zip and not args.from_github:
        parser.error("--updated-zip requires --from-github")

    run_id = datetime.now(timezone.utc).strftime("hlo-%Y%m%dT%H%M%S%fZ")
    run = (args.output / run_id).resolve()
    run.mkdir(parents=True, exist_ok=False)
    project = run / "project"
    data = run / "ontology-data"
    evaluation = run / "evaluation"
    log_path = run / "hlo-training.log"
    summary = {
        "repository": REPO,
        "requested_ref": args.ref,
        "started_utc": utc_now(),
        "stages": {},
    }

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        def emit(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        def command(stage, *argv, cwd=None):
            emit(f"\n[{utc_now()}] {stage}")
            emit("$ " + subprocess.list2cmdline([str(arg) for arg in argv]))
            started = monotonic()
            tracking.event("stage_started", stage=stage)
            result = summary["stages"][stage] = {
                "status": "running",
                "started_utc": utc_now(),
            }
            with subprocess.Popen(
                [str(arg) for arg in argv],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            ) as process:
                for line in process.stdout:
                    emit(line.rstrip("\n"))
                    if stage == "train_mt5" and line.startswith("HLO_EPOCH "):
                        epoch = json.loads(line.removeprefix("HLO_EPOCH "))
                        tracking.event("epoch", **epoch)
                        tracking.log(epoch)
                code = process.wait()
            result.update(
                status="passed" if code == 0 else "failed",
                exit_code=code,
                finished_utc=utc_now(),
                duration_seconds=round(monotonic() - started, 3),
            )
            tracking.event("stage_finished", stage=stage, **result)
            tracking.log({f"stages/{stage}/duration_seconds": result["duration_seconds"],
                          f"stages/{stage}/exit_code": code})
            if code:
                raise subprocess.CalledProcessError(code, argv)

        def hlo(stage, *argv):
            command(stage, sys.executable, "-u", "-m", "hlo_recommender", *argv,
                    cwd=project)

        def skip(stage, reason):
            summary["stages"][stage] = {"status": "skipped", "reason": reason}
            tracking.event("stage_skipped", stage=stage, reason=reason)
            emit(f"[SKIPPED] {stage}: {reason}")

        emit(f"Run directory: {run}")
        tracking = Tracking(run, args.wandb_mode, args.wandb_project, args.wandb_entity, emit)
        summary["wandb"] = tracking.state
        tracking.event("run_started", check_only=args.check_only, wandb_mode=args.wandb_mode)
        tracking.configure({"model": "google/mt5-small", "check_only": args.check_only})
        try:
            if args.from_github:
                command("git_clone", "git", "clone", "--no-checkout", REPO, project)
                command("git_checkout", "git", "checkout", "--detach", args.ref, cwd=project)
                commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=project, text=True
                ).strip()
                summary["git_commit"] = commit
                emit(f"Git commit: {commit}")
                if not args.updated_zip:
                    for path in (
                        "src/hlo_recommender/ontology_data.py",
                        "src/hlo_recommender/cli.py",
                        "src/hlo_recommender/data.py",
                        "src/hlo_recommender/training.py",
                    ):
                        if (project / path).is_file() and (project / path).read_bytes() == (source / path).read_bytes():
                            continue
                        if commit != SNAPSHOT_COMMIT:
                            raise ValueError("Use the bundled source or supply an updated ZIP for this checkout.")
                        shutil.copy2(source / path, project / path)
                    summary["ontology_integration"] = "bundled"
            else:
                excluded = {".git", ".venv", "__pycache__", "artifacts", "hlo-runs"}

                def ignore_copy(directory, names):
                    return [name for name in names if name in excluded
                            or name.endswith(".egg-info")
                            or (Path(directory) / name).resolve() == args.output.resolve()]

                shutil.copytree(source, project, ignore=ignore_copy)
                emit(f"Copied project: {source}")
                summary["source"] = str(source)

            if args.updated_zip:
                snapshot = args.updated_zip.resolve(strict=True)
                if commit != SNAPSHOT_COMMIT:
                    raise ValueError(
                        f"The updated ZIP was prepared for {SNAPSHOT_COMMIT}. "
                        "Use that commit with --ref when applying the ZIP."
                    )
                root = "recommendation-system/"
                with zipfile.ZipFile(snapshot) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir():
                            continue
                        if not entry.filename.startswith(root):
                            raise ValueError(f"Unexpected ZIP entry: {entry.filename}")
                        relative = Path(entry.filename.removeprefix(root))
                        if relative.is_absolute() or {"..", ".git"} & set(relative.parts):
                            raise ValueError(f"Unsafe ZIP entry: {entry.filename}")
                        target = project / relative
                        if not target.resolve().is_relative_to(project):
                            raise ValueError(f"ZIP entry escapes project: {entry.filename}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(entry) as source, target.open("wb") as dest:
                            shutil.copyfileobj(source, dest)
                summary["updated_zip_sha256"] = sha256(snapshot)
                emit(f"Applied updated project archive: {snapshot.name}")

            command("install", sys.executable, "-m", "pip", "install", "-e",
                    ".", cwd=project)
            command("unit_tests", sys.executable, "-m", "unittest", "discover",
                    "-s", "tests", "-v", cwd=project)

            config = project / "configs/mt5.toml"
            summary["mt5_config_sha256"] = sha256(config)
            with config.open("rb") as stream:
                training_config = tomllib.load(stream)
            tracking.configure({key: value for key, value in training_config.items()
                                if isinstance(value, (float, int))})
            hlo("generate_ontology_questions", "data", "ontology", "--output", data)
            hlo("validate_data", "data", "validate", "--directory", data)
            hlo("mt5_preflight", "train", "mt5", "--config", config,
                "--ontology-data", data, "--check-only")

            runtime = data / "runtime.toml"
            test_data = data / "questions.test.jsonl"
            hlo("evaluate_template", "--config", runtime, "evaluate", "queries",
                "--generator", "template", "--data", test_data,
                "--output", evaluation / "template")

            template_metrics = query_metrics(evaluation / "template")
            summary["query_metrics"] = {"template": template_metrics}
            tracking.event("evaluation", generator="template", **template_metrics)
            tracking.log({f"test/template/{key}": value for key, value in template_metrics.items()})
            if args.check_only:
                skip("train_mt5", "Preflight only.")
                skip("evaluate_mt5", "Preflight only.")
                summary["status"] = "preflight_completed"
            else:
                command("install_models", sys.executable, "-m", "pip", "install", "-e",
                        ".[model]", cwd=project)
                gpu_check = (
                    "import torch; "
                    "print('CUDA:', torch.cuda.is_available()); "
                    "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
                )
                if not args.allow_cpu:
                    gpu_check += "; assert torch.cuda.is_available(), 'Enable a GPU in Colab'"
                command("device_check", sys.executable, "-c", gpu_check, cwd=project)

                # Keep the original mt5.toml path for the pretrained base checkpoint.
                base = project / "artifacts/base/mt5-small"
                hlo("download_base", "download", "--model", "google/mt5-small",
                    "--revision", "73fb5dbe4756edadc8fbe8c769b0a109493acf7a", "--output", base)
                hlo("train_mt5", "train", "mt5", "--config", config,
                    "--ontology-data", data, "--output", run / "mt5")
                hlo("evaluate_mt5", "--config", runtime, "evaluate", "queries",
                    "--generator", "mt5", "--checkpoint", run / "mt5/checkpoint",
                    "--data", test_data, "--output", evaluation / "mt5")

                mt5_metrics = query_metrics(evaluation / "mt5")
                summary["query_metrics"]["mt5"] = mt5_metrics
                tracking.event("evaluation", generator="mt5", **mt5_metrics)
                tracking.log({f"test/mt5/{key}": value for key, value in mt5_metrics.items()})

                reviewed = project / "data/reviewed"
                parts = ("train", "validation", "test")
                if all((reviewed / f"assessments.{part}.jsonl").is_file() for part in parts):
                    hlo("train_bkt", "train", "bkt", "--config", "configs/bkt.toml")
                    hlo("evaluate_bkt", "evaluate", "bkt", "--model",
                        "artifacts/bkt/run-001/model.json", "--data",
                        "data/reviewed/assessments.test.jsonl",
                        "--output", evaluation / "bkt")
                else:
                    skip("bkt", "Reviewed learner assessments are missing.")

                if all((reviewed / f"transitions.{part}.jsonl").is_file()
                       for part in ("train", "validation")):
                    hlo("train_policy", "train", "policy", "--config", "configs/policy.toml")
                else:
                    skip("policy", "Reviewed transition files are missing.")

                summary["status"] = "completed"
        except Exception as error:
            summary.update(status="failed", error=str(error))
            emit(traceback.format_exc())
        except KeyboardInterrupt:
            summary.update(status="interrupted", error="Interrupted by user")
            emit("Run interrupted.")
        finally:
            tracking.finish(summary["status"])
            summary["finished_utc"] = utc_now()
            (run / "run-summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            emit(f"Run status: {summary['status']}")

    archive_path = run.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run)
            if not path.is_file() or {".git", "__pycache__"} & set(relative.parts):
                continue
            # The downloadable pretrained base is large. The fine-tuned checkpoint is included.
            if relative.parts[:4] == ("project", "artifacts", "base", "mt5-small"):
                continue
            archive.write(path, arcname=relative)
    print(f"Log: {log_path}")
    print(f"Archive: {archive_path}")
    return 0 if summary["status"] in {"completed", "preflight_completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
