# Contributor: Charansagar Ramanujam (Task 3).
"""Task 3 structured-output LLM command parser."""

# TODO: define schema and convert English utterances to ordered robot actions.
import os
import ollama
from typing import Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load variables from .env
load_dotenv()


class RobotAction(BaseModel):
    """One structured action that the robot can perform."""

    action: Literal[
        "move",
        "turn",
        "goto_object",
        "stop",
        "chat",
    ]

    # move parameters
    vx: Optional[float] = None
    vy: Optional[float] = None
    wz: Optional[float] = None
    duration: Optional[float] = None

    # turn parameter
    angle_deg: Optional[float] = None

    # goto_object parameters
    class_name: Optional[str] = None
    color: Optional[str] = None

    # chat parameter
    message: Optional[str] = None


class RobotCommand(BaseModel):
    """Ordered list of actions produced from one user command."""

    actions: list[RobotAction] = Field(min_length=1)


class GeminiCommandParser:
    """Convert natural-language commands into structured robot actions."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=api_key)

    def parse(self, command):
        """Convert an English command into validated robot actions."""

        prompt = f"""
You are the natural-language command parser for a quadruped robot.

Convert the user's command into an ordered list of robot actions.

Available actions:

1. move
   Parameters:
   vx, vy, wz, duration

2. turn
   Parameter:
   angle_deg

3. goto_object
   Parameters:
   class_name, color

4. stop

5. chat
   Parameter:
   message

Robot velocity convention:
- vx > 0 means forward
- vx < 0 means backward
- vy > 0 means left
- vy < 0 means right
- wz > 0 means counter-clockwise / left rotation
- wz < 0 means clockwise / right rotation

Default translational speeds:
- forward: vx = 0.4
- backward: vx = -0.4
- left: vy = 0.4
- right: vy = -0.4
- unused velocity components must be 0.0

Turn convention:
- positive angle_deg means left / counter-clockwise
- negative angle_deg means right / clockwise

Rules:
- Preserve the order of actions.
- Do not invent missing essential information.
- Use the default speeds unless the user explicitly specifies a valid speed.
- For movement, duration must be specified by the user.
- For turning, angle must be specified by the user.
- For goto_object, identify both object class and color when provided.
- If essential information is missing or the request cannot be interpreted
  as a valid robot command, return a chat action explaining the problem.
- If the command is not in English, return a chat action asking the user
  to repeat the command in English.

User command:
{command}
"""

        response = self.client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RobotCommand,
            ),
        )

        # Gemini's response is validated against RobotCommand
        result = RobotCommand.model_validate_json(response.text)

        # Return a normal Python list of dictionaries
        return [
            action.model_dump(exclude_none=True)
            for action in result.actions
        ]
class OllamaCommandParser:
    """Natural-language command parser using local Qwen through Ollama."""

    def __init__(self):
        self.model = "qwen3:4b"

    def parse(self, command):
        prompt = f"""
You are the natural-language command parser for a quadruped robot.

Convert the user's command into an ordered list of robot actions.

Available actions:

1. move
   Parameters:
   vx, vy, wz, duration

2. turn
   Parameter:
   angle_deg

3. goto_object
   Parameters:
   class_name, color

4. stop

5. chat
   Parameter:
   message

Robot velocity convention:
- vx > 0 means forward
- vx < 0 means backward
- vy > 0 means left
- vy < 0 means right
- wz > 0 means counter-clockwise / left rotation
- wz < 0 means clockwise / right rotation

Default translational speeds:
- forward: vx = 0.4
- backward: vx = -0.4
- left: vy = 0.4
- right: vy = -0.4
- unused velocity components must be 0.0

Turn convention:
- positive angle_deg means left / counter-clockwise
- negative angle_deg means right / clockwise

Rules:
- Preserve the order of actions.
- Do not invent missing essential information.
- Use the default speeds unless the user explicitly specifies a valid speed.
- For movement, duration must be specified by the user.
- For turning, angle must be specified by the user.
- For goto_object, identify both object class and color when provided.
- If essential information is missing or the request cannot be interpreted
  as a valid robot command, return a chat action explaining the problem.
- If the command is not in English, return a chat action asking the user
  to repeat the command in English.

User command:
{command}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format=RobotCommand.model_json_schema(),
            think=False,
        )

        result = RobotCommand.model_validate_json(
            response["message"]["content"]
        )

        return [
            action.model_dump(exclude_none=True)
            for action in result.actions
        ]
