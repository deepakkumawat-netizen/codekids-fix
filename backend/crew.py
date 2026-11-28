from ai_client import client as _fallback_client
"""Multi-agent system for CodeKids - code teaching coordination"""
import os
from dotenv import load_dotenv

load_dotenv()

class Agent:
    def __init__(self, role: str, goal: str):
        self.role = role
        self.goal = goal
        self.client = _fallback_client

    def execute(self, task: str) -> str:
        response = self.client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[
                {"role": "system", "content": f"You are a {self.role}. Goal: {self.goal}"},
                {"role": "user", "content": task}
            ],
            max_tokens=1200,
            temperature=0.7
        )
        return response.choices[0].message.content

def teach_coding_concept(concept: str, language: str = "Python") -> dict:
    """Coordinate teacher and mentor agents for code learning"""

    teacher_agent = Agent("Code Teacher", "Explain programming concepts for beginners")
    mentor_agent = Agent("Code Mentor", "Create practice exercises and feedback")

    # Teaching task
    teach_prompt = f"Teach {concept} in {language} with simple examples and analogies"
    lesson = teacher_agent.execute(teach_prompt)

    # Exercise task
    exercise_prompt = f"Create beginner exercises for learning {concept} in {language}"
    exercises = mentor_agent.execute(exercise_prompt)

    return {
        "lesson": lesson,
        "exercises": exercises,
        "concept": concept,
        "language": language
    }
