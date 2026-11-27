# HLO duplication audit

The earliest visible duplicates are two notebook code blocks already present in the first commit. The largest repeated algorithm is BKT fitting in the standalone plotting script. The working tree also contains five groups of byte-identical tracked output files.

## Scope and attribution

The comparison covers all 13 commits in the local history and the current working tree. Python files and notebook code cells were compared as whole cells and as cross-file runs of at least five nonblank token lines containing control flow or a function definition, ignoring comments and indentation. The manual review covers overlapping behavior, generated notebooks, configuration, and output files. Notebook cell positions are one-based and include Markdown cells. A commit identifies the earliest evidence in this repository, and its author attribution comes from Git metadata. The first snapshot cannot establish where the code originated before import.

## Exact and generated code copies

| First seen | Earlier source and repeated locations | What repeats |
| --- | --- | --- |
| `7878dd0` (anooos2124) | Cell 2 of [Data](../Notebook/Data/data.ipynb), [Evaluation](../Notebook/Evaluation/evaluation.ipynb), [SPARQL](../Notebook/Queries/sparql.ipynb), [Recommendation](../Notebook/Recommendation/recommendation.ipynb), [BKT](../Notebook/Training/bkt.ipynb), [mT5](../Notebook/Training/mt5.ipynb), and [Policy](../Notebook/Training/policy.ipynb) | The whole setup cell matches in all seven notebooks. It finds the repository root and initializes the notebook session. Each notebook can start independently. The seven copies first appear together. |
| `7878dd0` (anooos2124) | [BKT](../Notebook/Training/bkt.ipynb), cell 3, and [mT5](../Notebook/Training/mt5.ipynb), cell 5, lines 7–19 in each cell | An identical block reads a base TOML file, resolves paths, applies the session's dataset and run directory, and writes a per-run config. The preceding lines select model-specific data and paths. Both copies first appear together. |
| `ffaae5c` (anooos2124) | [`fit_bkt`](../src/hlo_recommender/models/bkt.py#L52), present since `7878dd0`, and [`train_bkt_with_logging`](../tools/run_training_and_generate_plots.py#L56) | The nine significant lines of the likelihood calculation match. Sequence grouping, optimizer bounds, seeded starting points, minimization, and selection of the best finite fit also repeat. The plotting script records objective evaluations for its optimization plot. |
| `5766151` and `15c21e4` (anooos2124) | Historical `tools/build_notebook.py` and [`Notebook/pipeline.ipynb`](../Notebook/pipeline.ipynb) | All seven executable notebook cells in `5766151` match the generator's strings. All eight match in `15c21e4`, including the added agent consultation. The generator was deleted in `594fb2a`; the generated notebook is still present. |

The [reviewed BKT config](../configs/bkt.toml) and [development BKT config](../configs/bkt.development.toml) share the same schema and seed. The [reviewed mT5 config](../configs/mt5.toml) and [development mT5 config](../configs/mt5.development.toml) share the runtime, base checkpoint, and most hyperparameters. All four first appeared in `7878dd0`. The paired configs supply different data and run paths, and some training settings also differ.

## Overlapping behavior

| Commit | Earlier code | Added overlap |
| --- | --- | --- |
| `7878dd0` (anooos2124) | Resource [retrieval](../src/hlo_recommender/pipeline.py#L88), [schema construction and template generation](../src/hlo_recommender/knowledge.py#L59), and [eligibility](../src/hlo_recommender/recommendation.py#L43) | Retrieval, schema construction, and the template generator each resolve the question's skill aliases. The retrieval, template, and eligibility entry points each validate level and time. Both paths were already present in the first snapshot. |
| `7878dd0` (anooos2124) | [Runtime recommendation](../src/hlo_recommender/pipeline.py#L98) and [experiment ranking](../src/hlo_recommender/experiments.py#L32) | Both assemble a pool from retrieved IDs and configure the shared ranker with runtime weights, a cutoff, and initial mastery. The experiment retrieves once per generator to compare static and adaptive arms on the same candidates. |
| `ffaae5c` and `5766151` (anooos2124) | [BKT training and evaluation](../src/hlo_recommender/training.py#L16) and the [experiment runner](../src/hlo_recommender/experiments.py#L32) | The plotting script and pipeline notebook both load development assessments, evaluate predictions against a constant baseline, run the ranking comparison, and present calibration and ranking results. The notebook calls the shared fitter. The script duplicates the fitter in order to capture optimization traces. [Plotting script](../tools/run_training_and_generate_plots.py) · [Pipeline notebook](../Notebook/pipeline.ipynb) |
| `15c21e4` (anooos2124) | [Runtime retrieval](../src/hlo_recommender/pipeline.py#L88), [recommendation CLI](../src/hlo_recommender/cli.py#L225), and [learner store](../src/hlo_recommender/models/bkt.py#L88) | The agent resolves aliases and checks context before calling the runtime, which performs those checks again. The new CLI command repeats level choices and BKT model and database setup. The agent delegates query execution and resource ranking to the existing runtime. [Agent](../src/hlo_recommender/agent.py#L112) · [Agent CLI](../src/hlo_recommender/cli.py#L213) |
| `cef8d35` (anooos2124) | [Shared JSONL reader](../src/hlo_recommender/io.py#L18), [ontology training driver](../tools/train_hlo.py#L78), and [plotting script](../tools/run_training_and_generate_plots.py#L196) | The full training driver adds a second JSONL reader, another subprocess and CLI wrapper, and calibration and ranking charts. Its subprocess wrapper supports preview and captures structured CLI results. Its driver trains and evaluates resource recommendation. The earlier driver targets ontology parents and records stages with W&B. [Full training driver](../tools/train_pipeline.py#L28) |
| `f45fb54` (Copilot) | [Resource runtime](../src/hlo_recommender/pipeline.py#L21) and [shared file digest](../src/hlo_recommender/io.py#L12) | The ontology-parent runtime repeats knowledge and query-runner setup, checkpoint loading, and the single-column candidate loop with a distinct parent answer contract. Its training driver has a second SHA-256 helper that streams the file. [Ontology runtime](../src/hlo_recommender/ontology_data.py#L44) · [Training driver](../tools/train_hlo.py#L26) |

The Colab change `c7cc87a` replaces the prior ontology notebook with the [current Colab notebook](../Notebook/Colab/HLO-Colab-Training.ipynb). Both invoke `tools/train_hlo.py`. The old path is absent from the working tree.

## Byte-identical output files

SHA-256 comparison found five groups across nonempty tracked files. Each directory listed in a row contains the named file unless full paths are shown. Git records when each output was added. Matching bytes can also result from rerunning an experiment with the same inputs.

| Files or filename | Matching copies and history |
| --- | --- |
| BKT validation predictions | [`log/learner_mastery_events.jsonl`](../log/learner_mastery_events.jsonl) appeared in `0a17f98`. [`artifacts/bkt/development-001/validation_predictions.jsonl`](../artifacts/bkt/development-001/validation_predictions.jsonl) appeared in `c2543ca` with the same bytes. |
| `metrics.csv` | Six copies across `examples/results/` (`7878dd0`), `log/experiment_run/` (`6f2131e`), and four runs added in `c2543ca`: `artifacts/experiments/development-001/` and `artifacts/notebooks/experiment-586d680c/`, `experiment-b7ed3d20/`, and `experiment-dc9e88b6/`. |
| `template_adaptive_ranking.jsonl` | Five copies across `log/experiment_run/` (`6f2131e`) and the same four artifact runs (`c2543ca`). |
| `template_static_ranking.jsonl` | Five copies across `log/experiment_run/` (`6f2131e`) and the same four artifact runs (`c2543ca`). |
| Ranking report | [`log/evaluation_ranking.json`](../log/evaluation_ranking.json) and [`log/experiment_run/results.json`](../log/experiment_run/results.json) appeared together in `6f2131e`. |

## Verification records

The ontology experiment uses the [Colab notebook](../Notebook/Colab/HLO-Colab-Training.ipynb) or `python tools/train_hlo.py --check-only`. The [archived run summary](../verification/run-summary.json) records 642 successful template executions, Answer F1 of 1.000, and skipped mT5 training and evaluation. The November 27 local review reproduced the template results, confirmed that the ontology hash matches the archived value, and passed 23 automated tests after correcting the agent's resource durations, eligible counts, and empty-result status.
