
from crewai import Agent, Task, Crew, Process


def create_assessment_tasks(
        agents: tuple[Agent, Agent, Agent],
        exam_date: str,
        student_email: str = None
) -> List[Task]:
    """
    Crea le task per il processo di valutazione.

    Args:
        agents: Tuple di (loader, assessor, reporter)
        exam_date: Data dell'esame (es. "2025-06-05")
        student_email: Email studente (None = tutti)

    Returns:
        Lista di Task CrewAI
    """
    loader_agent, assessor_agent, reporter_agent = agents

    # TASK 1: Load Exam
    load_task = Task(
        description=f"""Load the exam from date {exam_date}.

        Steps:
        1. Load questions from: se-{exam_date}-questions.yml
        2. Load responses from: se-{exam_date}-responses.yml
        3. Load grades from: se-{exam_date}-grades.yml (if available)
        4. Load checklists for each question

        Return a summary with number of questions and students loaded.""",
        expected_output="JSON summary of loaded exam data",
        agent=loader_agent
    )

    # TASK 2: Assess Answers
    if student_email:
        assess_desc = f"Assess all answers for student: {student_email}"
    else:
        assess_desc = "Assess all answers for all students in the exam"

    assess_task = Task(
        description=f"""{assess_desc}

        For each student response:
        1. Check each core element (70% weight)
        2. Check each important detail (30% weight)
        3. Calculate the final score
        4. Provide constructive feedback

        Use the assess_feature_tool for each feature check.
        Be systematic and thorough.""",
        expected_output="Complete assessment results for all evaluated answers",
        agent=assessor_agent,
        context=[load_task]  # Depends on load_task
    )

    # TASK 3: Generate Report
    report_task = Task(
        description="""Generate a comprehensive assessment report.

        Include:
        1. Overall statistics (average, min, max scores)
        2. Individual student results
        3. Comparison with original Moodle grades (if available)
        4. Score breakdown by question

        Format the report clearly and professionally.""",
        expected_output="Markdown formatted assessment report",
        agent=reporter_agent,
        context=[assess_task]  # Depends on assess_task
    )

    return [load_task, assess_task, report_task]