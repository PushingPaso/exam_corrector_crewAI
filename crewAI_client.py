"""
MCP Client con CrewAI Multi-Agent System.
Sostituisce il sistema basato su LangChain agents.
"""

import asyncio
import os

from crewai import Agent, Task, Crew, Process

from exam import DIR_ROOT
from exam.agents import createAgents
from exam.llm_provider import get_llm
from exam.tasks import create_assessment_tasks


class CrewAIExamClient:
    """
    Client principale basato su CrewAI.
    Sostituisce MCPClientDemo con sistema multi-agente.
    """
    def __init__(self):

        self.llm_config = get_llm()
        self.agents = createAgents(self.llm_config)
        self.evaluations_dir = DIR_ROOT / "evaluations"
        self.evaluations_dir.mkdir(parents=True, exist_ok=True)


    async def single_student(self, exam_date:str):
        pass


    async def full_exam(
            self,
            exam_date: str,
            num_workers: int = 3
    ) -> dict:

        print("Exam ASSESSMENT")

        crew = Crew(
            agents=self.agents,
            tasks=create_assessment_tasks(self.agents,exam_date),
            llm=self.llm_config,
            verbose=False
        )

        result = crew.kickoff()

        return {
            "status": "success",
            "exam_date": exam_date,
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

    print(" CREWAI EXAM ASSESSMENT SYSTEM")
    print("\nThis system uses CrewAI multi-agent orchestration to:")
    print("  - Load exam data and checklists")
    print("  - Assess student answers with specialized agents")
    print("  - Generate comprehensive reports")



    client = CrewAIExamClient()
    exam_date = input("\nEnter exam date (format: YYYY-MM-DD, e.g., 2025-06-05): ").strip()

    await client.full_exam(exam_date)


    print("\nAll done!")


if __name__ == "__main__":
    asyncio.run(main())