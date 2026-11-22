from collections import Counter
import json
from pathlib import Path
import re
import tomllib

from rdflib import OWL, RDF, RDFS, URIRef

from .data import validate_partitions
from .io import config, grouped_split, new_run, provenance, resolve, write_json, write_jsonl
from .knowledge import HLO, Knowledge
from .models.mt5 import MT5Generator
from .query import QueryRejected, QueryRunner, validate

BODY = "SELECT DISTINCT ?parent WHERE { focus: relation:subClassOf ?parent . FILTER(isIRI(?parent)) }"
CLASS_ID = re.compile(r"\[HLO:([A-Za-z0-9_]+)\]$")


def ontology_query(term, body=BODY):
    return f"PREFIX focus: {term.n3()}\nPREFIX relation: <{RDFS}>\n{body}"


def class_iri(question):
    match = CLASS_ID.search(question)
    if not match:
        raise ValueError("Question must end with an HLO class ID")
    return HLO[match[1]]


class OntologyGrounder:
    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.sha256 = knowledge.ontology_sha256

    def schema_text(self, question):
        term = class_iri(question)
        if (term, RDF.type, OWL.Class) not in self.knowledge.schema:
            raise ValueError("Question refers to an unknown ontology class")
        return (f"OUTPUT SELECT DISTINCT ?parent\n"
                f"PREFIX focus: {term.n3()}\nPREFIX relation: <{RDFS}>\n"
                "PROJECT named direct parent IRIs via relation:subClassOf")


class OntologyRuntime:
    def __init__(self, cfg):
        self.config = cfg
        self.knowledge = Knowledge.load(resolve(cfg, cfg["ontology"]), resolve(cfg, cfg["catalog"]))
        self.grounder = OntologyGrounder(self.knowledge)
        self.runner = QueryRunner(self.knowledge, float(cfg.get("query_timeout", 10)),
                                  int(cfg.get("max_query_rows", 1000)))

    def prepare_references(self, rows):
        for row in rows:
            term = class_iri(row["question"])
            query = ontology_query(term)
            if query != row["sparql"]:
                raise ValueError(f"Unexpected reference query for {row['id']}")
            if query in self.runner._cache:
                continue
            checks = validate(query, self.knowledge.known_terms)
            result = self.knowledge.graph.query(query)
            answers = sorted({cell.parent.n3() for cell in result})
            expected = sorted({p.n3() for p in self.knowledge.schema.objects(term, RDFS.subClassOf)
                               if isinstance(p, URIRef)})
            if answers != expected or not expected:
                raise ValueError(f"Reference mismatch for {row['id']}")
            self.runner._cache[query] = {
                "checks": checks,
                "result": {"kind": "SELECT", "variables": ["parent"],
                           "rows": [[answer] for answer in answers], "truncated": False},
            }

    def select(self, question, candidates):
        errors = []
        for candidate in candidates:
            query = ontology_query(class_iri(question), candidate)
            try:
                result = self.runner.execute(query)["result"]
                if result["kind"] != "SELECT" or len(result["variables"]) != 1:
                    raise QueryRejected("Expected one SELECT column")
                return {"query": query, "candidate_errors": errors}
            except (QueryRejected, ValueError) as exc:
                errors.append(str(exc))
        return {"query": None, "candidate_errors": errors}

    def generator(self, name, checkpoint):
        if name == "template":
            return None
        if checkpoint is None:
            raise ValueError("mT5 inference requires --checkpoint")
        return MT5Generator(checkpoint, self.knowledge.ontology_sha256, self.grounder.sha256)

    def retrieve(self, question, context, generator):
        if generator is None:
            return {"query": ontology_query(class_iri(question)), "candidate_errors": []}
        schema = self.grounder.schema_text(question)
        return self.select(question, generator.generate(question, schema, context))


def generate_ontology_data(runtime_path, output):
    cfg = config(runtime_path)
    runtime = OntologyRuntime(cfg)
    graph = runtime.knowledge.schema
    labels = {}
    for term in sorted(set(graph.subjects(RDF.type, OWL.Class))):
        if not isinstance(term, URIRef) or not str(term).startswith(str(HLO)):
            continue
        if not any(isinstance(p, URIRef) for p in graph.objects(term, RDFS.subClassOf)):
            continue
        names = sorted({str(label).strip() for label in graph.objects(term, RDFS.label) if str(label).strip()})
        if names:
            labels[term] = names[0]
    counts = Counter(label.casefold() for label in labels.values())
    rows = []
    for term, label in labels.items():
        class_id = str(term).removeprefix(str(HLO))
        if counts[label.casefold()] != 1 or not CLASS_ID.fullmatch(f"[HLO:{class_id}]"):
            continue
        questions = (f"What are the named direct parent classes of {label}?",
                     f"List the immediate named superclasses of {label}.")
        for index, question in enumerate(questions, start=1):
            rows.append({"id": f"{class_id}-{index}", "family_id": class_id,
                         "question": f"{question} [HLO:{class_id}]",
                         "sparql": ontology_query(term), "context": {"level": "Junior", "minutes": 1},
                         "data_status": "ontology_derived"})
    partitions = grouped_split(rows, "family_id", cfg["seed"])
    counts = validate_partitions(partitions)
    runtime.prepare_references(rows)
    output = new_run(output)
    for name, partition in partitions.items():
        write_jsonl(output / f"questions.{name}.jsonl", partition)
    source = Path(runtime_path).resolve()
    lines = source.read_text(encoding="utf-8").splitlines()
    path_keys = {"ontology", "catalog", "aliases", "resources", "learner_database"}
    for index, line in enumerate(lines):
        key = line.partition("=")[0].strip()
        if key in path_keys:
            lines[index] = f"{key} = {json.dumps(str(resolve(cfg, cfg[key])), ensure_ascii=False)}"
    (output / "runtime.toml").write_text('task = "ontology_parents"\n' + "\n".join(lines) + "\n",
                                         encoding="utf-8")
    report = provenance(seed=cfg["seed"], files={"ontology": runtime.knowledge.ontology},
                        task="ontology_parents", questions=counts, verified_references=len(rows))
    write_json(output / "manifest.json", report)
    return report


def validate_ontology_training_data(directory):
    directory = Path(directory)
    with (directory / "runtime.toml").open("rb") as stream:
        if tomllib.load(stream).get("task") != "ontology_parents":
            raise ValueError("Expected an ontology-parent runtime configuration")
    train, validation = (directory / f"questions.{name}.jsonl" for name in ("train", "validation"))
    return directory / "runtime.toml", train, validation
