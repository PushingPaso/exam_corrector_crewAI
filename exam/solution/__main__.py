import sys

from exam import *
from exam.solution import SolutionProvider



questions = get_questions_store()
llm = SolutionProvider()

if len(sys.argv) > 1:
    targets = [questions.question(id.strip()) for id in sys.argv[1:]]
else:
    targets = questions.questions

for q in targets:
        print(q.id)
        print("\t", q.text)
        try:
            a = llm.answer(q)
            print(a.pretty(indent=1))
        except Exception as e:
            print(f"# ERROR processing question {q.id}: {e}")
            import traceback
            traceback.print_exc()
            print("# Skipping to next question...")
        print("---")

        print("Done.")
