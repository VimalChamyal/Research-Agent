from datetime import date
from graphs.build_graph import build_app

app = build_app()

def run(topic: str):
    out = app.invoke({
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": date.today().isoformat(),
        "recency_days": 7,
        "sections": [],
        "final": "",
    })

    print(out["final"])

if __name__ == "__main__":
    run("Write a blog on Self Attention")