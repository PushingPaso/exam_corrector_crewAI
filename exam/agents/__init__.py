from crewai import Agent
from exam.mcp import ExamMCPServer

mcp = ExamMCPServer()


def createAgents(llm_config: dict) -> list[Agent]:
    # --- CORREZIONE QUI ---
    # Assegna il riferimento alla funzione, non eseguirla.
    # L'agente la chiamerà con gli ID giusti in un secondo momento.


    loader_agent = Agent(
        role="Exam Data Loader",
        goal="First Load exam questions, student responses, and assessment using load_exam_from_yaml_tool"
             "After load checklists efficiently with load_checklistand the end say to the assessot agent that he can start assessing",
        backstory="""You are a meticulous data manager for the Software Engineering course.
            Your job is to load and organize exam data, ensuring all required information
            is available for the assessment team.""",
        tools=[mcp.load_exam_from_yaml_tool, mcp.load_checklist],
        llm=llm_config,
        verbose=True,
        allow_delegation=False
    )

    # AGENT 2: Answer Assessor
    assessor_agent = Agent(
        role="Answer Assessor",
        goal="After loader agents, has ended to upload assess:"
             "get the steudents email using list_students_tool"
             "assess them all calling assess_students_batch",

        backstory="""You are an experienced Software Engineering professor.
            You evaluate student answers by checking if core concepts and important details
            are present. You are thorough but fair, giving credit where due.""",
        tools=[mcp.list_students, mcp.assess_students_batch],
        llm=llm_config,
        verbose=True,
        allow_delegation=False
    )



    return [loader_agent, assessor_agent]


#def getWorkerAgents(num_workers: int, llm_config: dict) -> list[Agent]:
    workers = []
    # (Questa funzione era già corretta)
    assess_feature_tool = mcp.assess_student_exam
    for i in range(num_workers):
        workers.append(Agent(
            role="Answer Assessor",
            goal="Evaluate student answers against assessment criteria with fairness and consistency",
            backstory="""You are an experienced Software Engineering professor.
            You evaluate student answers by checking if core concepts and important details
            are present. You are thorough but fair, giving credit where due.""",
            tools=[assess_feature_tool],
            llm=llm_config,
            verbose=True,
            allow_delegation=False))
    return workers