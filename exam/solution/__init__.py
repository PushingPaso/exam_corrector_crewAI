import json

from pydantic import BaseModel, Field
from yaml import safe_dump, safe_load
from exam import DIR_ROOT, Question
from exam.llm_provider import AIOracle
from exam.rag import sqlite_vector_store

FILE_TEMPLATE = DIR_ROOT / "exam" / "solution" / "prompt-template.txt"
DIR_SOLUTIONS = DIR_ROOT / "solutions"
DIR_SOLUTIONS.mkdir(exist_ok=True)


class Answer(BaseModel):
    core: list[str] = Field(
        description="Essential elements that must be present in the perfect answer to address the most important part of the question. Each item is a Markdown string.",
    )
    details_important: list[str] = Field(
        description="Important details that should be mentioned to enrich the answer. Each item is a Markdown string.",
    )

    def pretty(self, indent=0, prefix="\t") -> str:
        result = "Core (essential elements):\n"
        if self.core:
            result += "\n".join(f"- {item}" for item in self.core) + "\n"
        else:
            result += "- <none>\n"

        result += "Details - Important:\n"
        if self.details_important:
            result += "\n".join(f"- {item}" for item in self.details_important) + "\n"
        else:
            result += "- <none>\n"

        result = result.strip()
        if indent > 0:
            result = (indent * prefix) + result.replace("\n", "\n" + indent * prefix)
        return result


TEMPLATE = FILE_TEMPLATE.read_text(encoding="utf-8")


def get_prompt(question: str, *helps: str) -> str:
    """
    Creates the prompt using standard Python formatting.
    """
    help_string = "\n\n".join(helps) if helps else ""

    return TEMPLATE.format(
        class_name=Answer.__name__,
        question=question,
        help=help_string
    )


def cache_file(question: Question):
    return DIR_SOLUTIONS / f"{question.id}.yaml"


def save_cache(
        question: Question,
        answer: Answer,
        helps: list[str] = None,
        model_name: str = None,
        model_provider: str = None):
    cache_file_path = cache_file(question)
    with open(cache_file_path, "w", encoding="utf-8") as f:
        print(f"Saving answer to {cache_file_path}")
        yaml = answer.model_dump()
        yaml["question"] = question.text
        yaml["helps"] = helps
        yaml["id"] = question.id
        if model_name:
            yaml["model_name"] = model_name
        if model_provider:
            yaml["model_provider"] = model_provider
        yaml["prompt_template"] = TEMPLATE
        safe_dump(yaml, f, sort_keys=True, allow_unicode=True)
        return yaml


def load_cache(question: Question) -> Answer | None:
    cache_file_path = cache_file(question)
    if not cache_file_path.exists():
        return None
    with open(cache_file_path, "r", encoding="utf-8") as f:
        print(f"Loading cached answer from {cache_file_path}")
        try:
            cached_answer = safe_load(f)
            return Answer(
                core=cached_answer.get("core", []),
                details_important=cached_answer.get("details_important", []),
            )
        except Exception as e:
            print(f"Error loading cached answer from {cache_file_path}: {e}")
            cache_file_path.unlink()
            return None


class SolutionProvider(AIOracle):
    def __init__(self, model_name: str = None):
        super().__init__(model_name)
        self.__vector_store = sqlite_vector_store()
        self.__use_helps = self.__vector_store.dims > 0

    def answer(self, question: Question, max_helps=5) -> Answer:
        if cache := load_cache(question):
            return cache
        text = question.text
        helps = []

        if self.__use_helps:
            helps = [doc['content'] for doc in self.__vector_store.search(text, k=max_helps)]

        prompt = get_prompt(text, *helps)

        result_msg = self.llm.call(prompt)

        result_clean = result_msg.strip()
        if result_clean.startswith("```json"):
            result_clean = result_clean[7:]
        if result_clean.startswith("```"):
            result_clean = result_clean[3:]
        if result_clean.endswith("```"):
            result_clean = result_clean[:-3]
        result_clean = result_clean.strip()

        data = json.loads(result_clean)

        answer = Answer(**data)
        print(answer)
        save_cache(question, answer, helps, self.model_name, self.model_provider)
        return answer