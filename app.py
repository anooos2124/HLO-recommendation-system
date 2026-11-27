"""Gradio entry point for the HLO Recommender Space."""

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from html import escape
from os import getenv
from pathlib import Path

from hlo_recommender.agent import HLORecommenderAgent
from hlo_recommender.knowledge import LEVELS
from hlo_recommender.models.bkt import LearnerStore, load_parameters
from hlo_recommender.pipeline import Runtime


ROOT = Path(__file__).resolve().parent
PRACTICE_HISTORIES = {
    "No assessments": (),
    "One incorrect answer": (False,),
    "One correct answer": (True,),
    "Two correct answers": (True, True),
}


@lru_cache(maxsize=1)
def load_engine():
    runtime = Runtime.from_file(ROOT / "configs/default.toml")
    parameters = load_parameters(ROOT / "examples/results/bkt_model.json")
    return runtime, parameters


def notice(message):
    return f'<p class="hlo-notice">{escape(message)}</p>'


def topic_labels(runtime):
    labels = {}
    for alias, uri in runtime.grounder.aliases.items():
        labels.setdefault(uri, alias.capitalize())
    return labels


def render_results(result, labels):
    resources = result["curriculum"]
    if not resources:
        messages = {
            "clarification_required": "Choose a supported topic or try one of the example questions.",
            "no_eligible_resource": "No resources match these settings. Try allowing more time.",
            "no_valid_query": "The resource search could not complete. Please try again.",
        }
        return notice(messages.get(result["status"], "Recommendations are currently unavailable."))

    mastery = "".join(
        f'<li><span>{escape(labels.get(skill["uri"], skill["label"]))}</span> '
        f'<strong>{result["learner_mastery_diagnosis"][skill["uri"]]["mastery"]:.0%} estimated mastery</strong></li>'
        for skill in result["ontology_grounding"]
    )
    cards = "".join(
        f'<li class="hlo-resource"><h3>{escape(item["title"])}</h3>'
        f'<div class="hlo-metrics"><span>{item["duration_minutes"]} min</span>'
        f'<span>Score {item["score"]:.2f}</span>'
        f'<span>Mastery gap {item["mastery_gap"]:.0%}</span>'
        f'<span>Readiness {item["readiness"]:.0%}</span></div>'
        f'<p>{escape(item["rationale"])}</p></li>'
        for item in resources
    )
    return (
        f'<p class="hlo-count">Showing {len(resources)} of '
        f'{result["total_eligible_resources"]} eligible resources</p>'
        f'<ul class="hlo-mastery" aria-label="Estimated mastery">{mastery}</ul>'
        f'<ol class="hlo-resources">{cards}</ol>'
    )


def recommend(question, level, minutes, practice_history):
    question = (question or "").strip()
    if not question or len(question) > 300:
        return notice("Enter a learning question between 1 and 300 characters.")
    if level not in LEVELS or practice_history not in PRACTICE_HISTORIES:
        return notice("Select a learner level and a practice history.")
    if type(minutes) is not int or not 1 <= minutes <= 120:
        return notice("Enter a whole time limit between 1 and 120 minutes.")

    runtime, parameters = load_engine()
    # Each consultation has its own synthetic assessment history.
    store = LearnerStore(":memory:", parameters)
    try:
        agent = HLORecommenderAgent(runtime, parameters, store)
        learner = "demo-learner"
        history = PRACTICE_HISTORIES[practice_history]
        if history:
            start = datetime.now(timezone.utc)
            for skill in runtime.grounder.skills(question):
                for index, correct in enumerate(history):
                    agent.record_assessment({
                        "event_id": f"{skill}-{index}",
                        "learner_id": learner,
                        "skill_id": skill,
                        "correct": correct,
                        "timestamp": (start + timedelta(seconds=index)).isoformat(),
                        "data_status": "synthetic",
                    })
        result = agent.consult(question, {"level": level, "minutes": minutes}, learner_id=learner)
    finally:
        store.close()
    return render_results(result, topic_labels(runtime))


def build_demo():
    import gradio as gr

    runtime, _ = load_engine()
    topics = sorted(topic_labels(runtime).values())
    examples = [[f"Find resources for {topic.lower()}."] for topic in topics]
    with gr.Blocks(title="HLO Learning Recommender", analytics_enabled=False) as demo:
        gr.Markdown(
            "# HLO Learning Recommender\n"
            "Explore ontology-guided recommendations with a synthetic learning-resource catalog.",
            elem_id="hlo-header",
        )
        with gr.Row(elem_id="hlo-workspace"):
            with gr.Column(scale=4, min_width=300, elem_classes="hlo-panel"):
                gr.Markdown("## Learning request")
                question = gr.Textbox(
                    label="Learning question",
                    placeholder="What would you like to study?",
                    value=examples[0][0],
                    lines=1,
                    max_length=300,
                )
                with gr.Row():
                    level = gr.Dropdown(LEVELS, value=LEVELS[0], label="Learner level")
                    minutes = gr.Number(
                        value=15, label="Minutes per resource", minimum=1, maximum=120, precision=0,
                    )
                history = gr.Dropdown(
                    list(PRACTICE_HISTORIES),
                    value=next(iter(PRACTICE_HISTORIES)),
                    label="Simulated practice history",
                    info="Applies to each matched skill in this request.",
                )
                submit = gr.Button("Find recommendations", variant="primary")
                gr.Examples(
                    examples,
                    inputs=question,
                    label="Supported topics",
                    examples_per_page=len(examples),
                )
            with gr.Column(scale=6, min_width=300, elem_classes="hlo-panel"):
                gr.Markdown("## Recommendations")
                results = gr.HTML(notice("Submit a learning request to see ranked resources."))
        gr.on(
            triggers=[submit.click, question.submit],
            fn=recommend,
            inputs=[question, level, minutes, history],
            outputs=results,
            api_name="recommend",
            concurrency_limit=1,
        )
    return demo.queue(max_size=16)


if __name__ == "__main__":
    import gradio as gr

    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate").set(
        block_title_text_color="#075b53",
        block_info_text_color="#475569",
        button_primary_background_fill="#08796f",
        button_primary_background_fill_hover="#06665d",
        button_primary_text_color="#ffffff",
    )
    build_demo().launch(
        theme=theme,
        css_paths=ROOT / "styles.css",
        footer_links=[],
        run_history=False,
        server_name=getenv("GRADIO_SERVER_NAME") or ("0.0.0.0" if getenv("SPACE_ID") else "127.0.0.1"),
    )
