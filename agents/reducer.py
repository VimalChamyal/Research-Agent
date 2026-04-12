from pathlib import Path


def reducer_node(state: dict) -> dict:
    plan = state["plan"]

    if plan is None:
        raise ValueError("No plan found")

    # Sort sections
    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    body = "\n\n".join(ordered_sections).strip()

    final_md = f"# {plan.blog_title}\n\n{body}\n"

    # Save file
    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}