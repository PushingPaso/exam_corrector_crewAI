"""
MCP Client con CrewAI Multi-Agent System.
Sostituisce il sistema basato su LangChain agents.
"""

import asyncio
import os

from crewai import Agent, Task, Crew, Process

from exam import DIR_ROOT
from exam.assess import (
    create_assessment_agents, assess_feature_tool
)
from exam.llm_provider import get_llm


class CrewAIExamClient:
    """
    Client principale basato su CrewAI.
    Sostituisce MCPClientDemo con sistema multi-agente.
    """
    def __init__(self):

        self.llm_config = get_llm()
        self.agents = create_assessment_agents(self.llm_config)
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)


    async def single_student(self):
        pass


    async def full_exam(
            self,
            exam_date: str,
            num_workers: int = 3
    ) -> dict:
        """
        Valuta esame con workers paralleli.
        In CrewAI, usiamo Process.hierarchical per parallelismo.

        Args:
            exam_date: Data esame
            num_workers: Numero di worker paralleli

        Returns:
            Risultati aggregati
        """
        print(f"\n{'=' * 70}")
        print(f"PARALLEL ASSESSMENT with {num_workers} workers")
        print(f"{'=' * 70}\n")

        # In CrewAI, il parallelismo è gestito con Process.hierarchical
        # e un manager agent che delega ai worker

        # Crea manager agent
        manager = Agent(
            role="Assessment Manager",
            goal="Coordinate parallel assessment of multiple students",
            backstory="You manage a team of assessors to evaluate exams efficiently",
            llm=self.llm_config,
            verbose=True
        )

        # Crea worker agents
        workers = [
            Agent(
                role=f"Assessment Worker {i + 1}",
                goal="Assess assigned student responses quickly and accurately",
                backstory=f"You are worker #{i + 1} in the assessment team",
                tools=[assess_feature_tool],
                llm=self.llm_config,
                verbose=True
            )
            for i in range(num_workers)
        ]

        # Crea task principale
        main_task = Task(
            description=f"""Assess all students in exam {exam_date}.

            Delegate work evenly among your {num_workers} workers.
            Each worker should assess approximately the same number of students.
            Collect and aggregate results.""",
            expected_output="Aggregated assessment results for all students",
            agent=manager
        )

        # Crea crew gerarchico
        crew = Crew(
            agents=[manager] + workers,
            tasks=[main_task],
            process=Process.hierarchical,
            manager_llm=self.llm_config,
            verbose=True
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "exam_date": exam_date,
            "num_workers": num_workers,
            "result": result
        }


async def main():
    """Entry point principale."""

    # Verifica API key
    if not os.environ.get("GROQ_API_KEY"):
        print("\nGROQ_API_KEY not set!")
        print("Get free key at: https://console.groq.com/keys")
        print("\nSet it with:")
        print("  export GROQ_API_KEY='your-key-here'")
        return

    print("\n" + "="*70)
    print("🤖 CREWAI EXAM ASSESSMENT SYSTEM")
    print("="*70)
    print("\nThis system uses CrewAI multi-agent orchestration to:")
    print("  ✓ Load exam data and checklists")
    print("  ✓ Assess student answers with specialized agents")
    print("  ✓ Generate comprehensive reports")
    print("  ✓ Support parallel processing for faster evaluation")

    print("\n" + "="*70)
    print("SELECT MODE")
    print("="*70)
    print("\n1. Single student assessment")
    print("2. Full exam")

    client = CrewAIExamClient()
    choice = input("\nChoice (1-5): ").strip()

    if choice == "1":
        await client.single_student()
    elif choice == "2":
        await client.full_exam()


    print("\n✅ All done!")


if __name__ == "__main__":
    asyncio.run(main())