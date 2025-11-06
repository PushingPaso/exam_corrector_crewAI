"""
Sistema di generazione checklist con CrewAI.
Sostituisce exam/solution/__init__.py basato su LangChain.
"""

from pathlib import Path
from typing import List

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from pydantic import BaseModel, Field
from yaml import safe_dump, safe_load

from exam import DIR_ROOT, Question
from exam.llm_provider import get_llm
from exam.rag import sqlite_vector_store


# ============================================================================
# MODELLI PYDANTIC (invariati)
# ============================================================================

class Answer(BaseModel):
    """Checklist per valutazione risposta."""
    core: list[str] = Field(
        description="Elementi essenziali che devono essere presenti nella risposta perfetta"
    )
    details_important: list[str] = Field(
        description="Dettagli importanti che dovrebbero essere menzionati"
    )

    def pretty(self, indent=0, prefix="\t") -> str:
        result = "Core (elementi essenziali):\n"
        if self.core:
            result += "\n".join(f"- {item}" for item in self.core) + "\n"
        else:
            result += "- <none>\n"

        result += "Details - Importanti:\n"
        if self.details_important:
            result += "\n".join(f"- {item}" for item in self.details_important) + "\n"
        else:
            result += "- <none>\n"

        result = result.strip()
        if indent > 0:
            result = (indent * prefix) + result.replace("\n", "\n" + indent * prefix)
        return result


# ============================================================================
# TOOLS PER RAG
# ============================================================================

@tool("Search Course Material")
def search_course_material(query: str, max_results: int = 5) -> str:
    """
    Search course slides for relevant content.

    Args:
        query: Search query (the question text)
        max_results: Number of results to return

    Returns:
        Relevant course material snippets
    """
    try:
        vector_store = sqlite_vector_store()
        results = vector_store.similarity_search(query, k=max_results)

        snippets = []
        for i, doc in enumerate(results, 1):
            snippets.append(f"[Snippet {i}]\n{doc.page_content}\n---")

        return "\n\n".join(snippets)

    except Exception as e:
        return f"Error searching: {str(e)}"


# ============================================================================
# CACHE MANAGEMENT (invariato)
# ============================================================================

DIR_SOLUTIONS = DIR_ROOT / "solutions"
DIR_SOLUTIONS.mkdir(exist_ok=True)


def cache_file(question: Question) -> Path:
    """Percorso file cache per una domanda."""
    return DIR_SOLUTIONS / f"{question.id}.yaml"


def save_cache(
        question: Question,
        answer: Answer,
        helps: List[str] = None,
        model_name: str = None
) -> dict:
    """Salva checklist in cache."""
    cache_path = cache_file(question)

    yaml_data = answer.model_dump()
    yaml_data["question"] = question.text
    yaml_data["helps"] = helps or []
    yaml_data["id"] = question.id
    yaml_data["model_name"] = model_name

    with open(cache_path, "w", encoding="utf-8") as f:
        safe_dump(yaml_data, f, sort_keys=True, allow_unicode=True)

    print(f"✓ Saved checklist to {cache_path}")
    return yaml_data


def load_cache(question: Question) -> Answer | None:
    """Carica checklist dalla cache."""
    cache_path = cache_file(question)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = safe_load(f)
            return Answer(
                core=data.get("core", []),
                details_important=data.get("details_important", [])
            )
    except Exception as e:
        print(f"✗ Error loading cache: {e}")
        return None


# ============================================================================
# CREWAI AGENTS PER GENERAZIONE CHECKLIST
# ============================================================================

class ChecklistGenerationCrew:
    """
    Crew per generare checklist di valutazione.
    Sostituisce SolutionProvider basato su LangChain.
    """

    def __init__(self):
        """
        Inizializza il crew per generazione checklist.

        Args:
            model_name: Modello LLM da usare
        """
        self.llm_config = get_llm()
        self.vector_store = None

        # Verifica disponibilità RAG
        try:
            self.vector_store = sqlite_vector_store()
            self.use_rag = self.vector_store.get_dimensionality() > 0
        except:
            self.use_rag = False

        print(f"RAG {'enabled' if self.use_rag else 'disabled'}")

        # Crea agenti
        self.agents = self._create_agents()

    def _create_agents(self) -> tuple[Agent, Agent]:
        """Crea gli agenti specializzati."""

        # AGENT 1: Course Material Researcher
        researcher = Agent(
            role="Course Material Researcher",
            goal="Find relevant course content to inform assessment criteria",
            backstory="""You are an expert at searching course materials.
            You find the most relevant slides and lecture notes that explain
            the concepts being tested in each exam question.""",
            tools=[search_course_material] if self.use_rag else [],
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False
        )

        # AGENT 2: Assessment Criteria Designer
        designer = Agent(
            role="Assessment Criteria Designer",
            goal="Create clear, fair assessment checklists for exam questions",
            backstory="""You are a Software Engineering professor with years of experience
            designing fair and effective assessment rubrics. You create checklists that:

            1. CORE elements (70% weight): Essential concepts that MUST be present
               - Usually 1-2 items for simple questions
               - Focus on the fundamental understanding

            2. IMPORTANT details (30% weight): Significant additions
               - Only for complex questions
               - Enrich but don't duplicate core items

            Your checklists are specific, verifiable, and actionable.
            You avoid vague or fluffy criteria.""",
            llm=self.llm_config,
            verbose=True,
            allow_delegation=False
        )

        return researcher, designer

    def generate_checklist(self, question: Question) -> Answer:
        """
        Genera checklist per una domanda.

        Args:
            question: Domanda da cui generare checklist

        Returns:
            Answer object con core e details_important
        """
        # Check cache
        cached = load_cache(question)
        if cached:
            print(f"✓ Using cached checklist for {question.id}")
            return cached

        print(f"\n{'=' * 70}")
        print(f"GENERATING CHECKLIST: {question.id}")
        print(f"{'=' * 70}\n")

        researcher, designer = self.agents

        # TASK 1: Research (se RAG disponibile)
        tasks = []

        if self.use_rag:
            research_task = Task(
                description=f"""Find relevant course material for this question:

                {question.text}

                Search for slides/notes that explain the concepts being tested.
                Return the most relevant snippets that will help create assessment criteria.""",
                expected_output="Relevant course material snippets",
                agent=researcher
            )
            tasks.append(research_task)

        # TASK 2: Design Checklist
        design_description = f"""Create an assessment checklist for this question:

**Question:** {question.text}

Create a structured checklist with:

1. **CORE elements (70% weight)**
   - The essential concepts that MUST be present
   - Usually 1-2 items for simple questions
   - These directly answer the main question

2. **IMPORTANT details (30% weight)**  
   - Significant additions that enrich the answer
   - Only include if the question is complex
   - Do NOT duplicate core items

Rules:
- Be specific and verifiable
- No vague criteria
- Each item should be a clear statement
- Short questions → fewer items
- Complex questions → more items"""

        if self.use_rag:
            design_description += "\n\nUse the course material provided by the researcher to inform your criteria."

        design_task = Task(
            description=design_description,
            expected_output="A structured Answer object with core and details_important lists",
            agent=designer,
            context=tasks if self.use_rag else None,
            output_pydantic=Answer  # Output strutturato!
        )
        tasks.append(design_task)

        # Crea Crew
        crew = Crew(
            agents=[researcher, designer] if self.use_rag else [designer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Esegui
        result = crew.kickoff()

        # Result è già un Answer object grazie a output_pydantic!
        if isinstance(result, Answer):
            checklist = result
        else:
            # Fallback se non è già parsed
            checklist = Answer.model_validate_json(result)

        # Salva in cache
        helps = []
        if self.use_rag and len(tasks) > 1:
            helps = [tasks[0].output.raw_output] if tasks[0].output else []

        save_cache(
            question=question,
            answer=checklist,
            helps=helps,
            model_name=self.llm_config["model_name"]
        )

        print(f"\n{'=' * 70}")
        print("CHECKLIST GENERATED")
        print(f"{'=' * 70}")
        print(checklist.pretty())
        print(f"{'=' * 70}\n")

        return checklist

    def generate_all(self, questions: List[Question]) -> dict[str, Answer]:
        """
        Genera checklist per multiple domande.

        Args:
            questions: Lista di domande

        Returns:
            Dict {question_id: Answer}
        """
        results = {}

        print(f"\n{'=' * 70}")
        print(f"BATCH GENERATION: {len(questions)} questions")
        print(f"{'=' * 70}\n")

        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question.id}")
            results[question.id] = self.generate_checklist(question)

        print(f"\n{'=' * 70}")
        print(f" COMPLETED: {len(results)} checklists generated")
        print(f"{'=' * 70}\n")

        return results


# ============================================================================
# FUNZIONI DI COMPATIBILITÀ
# ============================================================================

class SolutionProvider:
    """
    Compatibilità con codice esistente.
    Wrapper che usa ChecklistGenerationCrew internamente.
    """

    def __init__(self, model_name: str = None, model_provider: str = None):
        # Ignora model_provider (sempre groq in CrewAI)
        self.crew = ChecklistGenerationCrew(model_name or "llama-3.3-70b-versatile")

    def answer(self, question: Question, max_helps: int = 5) -> Answer:
        """Genera checklist (compatibilità)."""
        return self.crew.generate_checklist(question)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    from exam import get_questions_store

    questions_store = get_questions_store()
    crew = ChecklistGenerationCrew()

    if len(sys.argv) > 1:
        # Domande specifiche
        targets = [questions_store.question(qid.strip()) for qid in sys.argv[1:]]
    else:
        # Tutte le domande
        targets = questions_store.questions

    crew.generate_all(targets)

    print("\n✅ Done!")