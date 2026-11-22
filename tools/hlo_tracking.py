import json
from datetime import datetime, timezone


class Tracking:
    def __init__(self, directory, mode, project, entity, report):
        self.path = directory / "hlo-events.jsonl"
        self.run = None
        self.report = report
        self.state = {"mode": mode, "status": "disabled"}
        if mode == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project, entity=entity, name=directory.name,
                mode=mode, dir=str(directory),
                settings=wandb.Settings(console="off", disable_git=True,
                                        save_code=False, x_disable_meta=True),
            )
            self.state.update(status="active", run_id=self.run.id)
            if mode == "online":
                self.state["url"] = self.run.url
            self.run.define_metric("epoch")
            self.run.define_metric("train/*", step_metric="epoch")
            self.run.define_metric("validation/*", step_metric="epoch")
        except Exception as error:
            self.failed(error)

    def failed(self, error):
        self.state.update(status="error", error_type=type(error).__name__)
        self.report(f"W&B tracking error ({type(error).__name__}); see local logs or W&B diagnostics.")

    def event(self, kind, **fields):
        record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "event": kind, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def log(self, metrics):
        if self.run is not None and self.state["status"] == "active":
            try:
                self.run.log(metrics)
            except Exception as error:
                self.failed(error)

    def configure(self, values):
        if self.run is not None and self.state["status"] == "active":
            try:
                self.run.config.update(values)
            except Exception as error:
                self.failed(error)

    def finish(self, status):
        self.event("run_finished", status=status)
        if self.run is not None:
            try:
                self.run.summary["pipeline_status"] = status
                self.run.finish(exit_code=int(status not in {"completed", "preflight_completed"}))
                if self.state["status"] == "active":
                    self.state["status"] = "finished"
            except Exception as error:
                self.failed(error)
