"""
MCP Client con CrewAI Multi-Agent System.
Sostituisce il sistema basato su LangChain agents.
"""

import asyncio
import os

from crewai import Agent, Task, Crew, Process

from exam.assess import (
    ExamAssessmentCrew
)


class CrewAIExamClient:
    """
    Client principale basato su CrewAI.
    Sostituisce MCPClientDemo con sistema multi-agente.
    """

    def __init__(self):
        """
        Inizializza il client CrewAI.

        Args:
            model_name: Modello LLM da usare
        """
        self.assessment_crew = ExamAssessmentCrew()

        print(f"\n{'='*70}")
        print("CREWAI EXAM ASSESSMENT SYSTEM")
        print(f"{'='*70}\n")

    async def assess_single_student(
        self,
        exam_date: str,
        student_email: str
    ) -> dict:
        """
        Valuta un singolo studente.

        Args:
            exam_date: Data esame (YYYY-MM-DD)
            student_email: Email dello studente

        Returns:
            Risultati valutazione
        """
        print(f"\n📝 Assessing student: {student_email[:40]}...")

        result = self.assessment_crew.assess_exam(
            exam_date=exam_date,
            student_email=student_email,
            process_type=Process.sequential
        )

        return result

    async def assess_full_exam(
        self,
        exam_date: str,
        parallel: bool = False,
        num_workers: int = 3
    ) -> dict:
        """
        Valuta un esame completo.

        Args:
            exam_date: Data esame (YYYY-MM-DD)
            parallel: Se True, usa workers paralleli
            num_workers: Numero di workers (solo se parallel=True)

        Returns:
            Risultati valutazione
        """
        print(f"\n📚 Assessing full exam: {exam_date}")

        if parallel:
            print(f"⚡ Using {num_workers} parallel workers")
            result = self.assessment_crew.assess_exam_parallel(
                exam_date=exam_date,
                num_workers=num_workers
            )
        else:
            print("📝 Sequential processing")
            result = self.assessment_crew.assess_exam(
                exam_date=exam_date,
                student_email=None,
                process_type=Process.sequential
            )

        return result

    async def interactive_assessment(self):
        """
        Modalità interattiva per valutazione esami.
        """
        print("\n" + "="*70)
        print("INTERACTIVE EXAM ASSESSMENT")
        print("="*70)

        # Chiedi data esame
        exam_date = input("\nExam date (YYYY-MM-DD, default: 2025-06-05): ").strip()
        if not exam_date:
            exam_date = "2025-06-05"

        print(f"\n📅 Selected exam: {exam_date}")

        # Chiedi modalità
        print("\nAssessment mode:")
        print("  1. Single student")
        print("  2. Full exam (sequential)")
        print("  3. Full exam (parallel)")

        mode = input("\nChoice (1-3): ").strip()

        if mode == "1":
            student_email = input("\nStudent email (or first 20 chars): ").strip()
            await self.assess_single_student(exam_date, student_email)

        elif mode == "2":
            await self.assess_full_exam(exam_date, parallel=False)

        elif mode == "3":
            num_workers = input("\nNumber of workers (default: 3): ").strip()
            num_workers = int(num_workers) if num_workers else 3
            await self.assess_full_exam(exam_date, parallel=True, num_workers=num_workers)

        else:
            print("Invalid choice")
            return

        print("\n" + "="*70)
        print("✅ ASSESSMENT COMPLETED")
        print("="*70)


# ============================================================================
# DEMO E MAIN
# ============================================================================

async def demo_single_student():
    """Demo: valuta un singolo studente."""
    print("\n🎯 DEMO: Single Student Assessment")

    client = CrewAIExamClient()

    await client.assess_single_student(
        exam_date="2025-06-05",
        student_email="1377db8e05e4"  # Primi caratteri
    )


async def demo_full_exam_sequential():
    """Demo: valuta esame completo in modo sequenziale."""
    print("\n🎯 DEMO: Full Exam Sequential")

    client = CrewAIExamClient()

    await client.assess_full_exam(
        exam_date="2025-06-05",
        parallel=False
    )


async def demo_full_exam_parallel():
    """Demo: valuta esame completo in parallelo."""
    print("\n🎯 DEMO: Full Exam Parallel")

    client = CrewAIExamClient()

    await client.assess_full_exam(
        exam_date="2025-06-05",
        parallel=True,
        num_workers=3
    )


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
    print("SELECT DEMO MODE")
    print("="*70)
    print("\n1. Single student assessment")
    print("2. Full exam (sequential)")
    print("3. Full exam (parallel)")
    print("4. Interactive mode")
    print("5. Exit")

    choice = input("\nChoice (1-5): ").strip()

    if choice == "1":
        await demo_single_student()
    elif choice == "2":
        await demo_full_exam_sequential()
    elif choice == "3":
        await demo_full_exam_parallel()
    elif choice == "4":
        client = CrewAIExamClient()
        await client.interactive_assessment()
    elif choice == "5":
        print("\nGoodbye!")
        return
    else:
        print("\nInvalid choice")
        return

    print("\n✅ All done!")


if __name__ == "__main__":
    asyncio.run(main())