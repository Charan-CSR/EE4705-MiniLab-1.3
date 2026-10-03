# Contributor: Charansagar Ramanujam (Task 3).
"""Execute parsed Task 3 commands using Task 2/4 APIs."""

# TODO: print required [EXEC] / [DONE] logs and dispatch actions.
class CommandExecutor:
    """Dispatch validated LLM actions to the robot motion functions."""

    def __init__(self, move_fn, turn_fn, stop_fn, goto_object_fn=None):
        self.move_fn = move_fn
        self.turn_fn = turn_fn
        self.stop_fn = stop_fn
        self.goto_object_fn = goto_object_fn

    def execute(self, actions):
        """Validate and dispatch an ordered list of structured actions."""

        for action in actions:
            action_type = action.get("action")

            if action_type == "move":
                required = ["vx", "vy", "wz", "duration"]

                if not all(key in action for key in required):
                    print("[REJECT] Move command is missing required parameters.")
                    continue

                vx = action["vx"]
                vy = action["vy"]
                wz = action["wz"]
                duration = action["duration"]

                if not all(-1.0 <= value <= 1.0 for value in [vx, vy, wz]):
                    print("[REJECT] Velocity must be within [-1.0, 1.0].")
                    continue

                if duration <= 0:
                    print("[REJECT] Move duration must be greater than 0.")
                    continue

                self.move_fn(vx, vy, wz, duration)

            elif action_type == "turn":
                if "angle_deg" not in action:
                    print("[REJECT] Turn command is missing an angle.")
                    continue

                self.turn_fn(action["angle_deg"])

            elif action_type == "stop":
                self.stop_fn()

            elif action_type == "goto_object":
                class_name = action.get("class_name")
                color = action.get("color")

                if not class_name or not color:
                    print("[REJECT] Object navigation requires a class and color.")
                    continue

                if self.goto_object_fn is None:
                    print("[CHAT] Object navigation is not available yet.")
                else:
                    self.goto_object_fn(class_name, color)

            elif action_type == "chat":
                message = action.get("message")

                if not message:
                    print("[REJECT] Chat response is missing a message.")
                    continue

                print(f"[CHAT] {message}")

            else:
                print(f"[REJECT] Unknown action: {action_type}")
