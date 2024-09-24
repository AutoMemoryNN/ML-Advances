class TState:
    state = None

    def __init__(self, goal: tuple[int, int], position: tuple[int, int], width: int, height: int):
        self.goal = goal
        self.position = position
        self.width = width
        self.height = height
        self.state = [''] * (width * height)

    def isTerminalState(self) -> bool:
        return self.position == self.goal

    def move(self, newPosition: tuple[int, int]):
        self.position = newPosition

    def getPosition(self) -> tuple[int, int]:
        return self.position

    def getGoal(self) -> tuple[int, int]:
        return self.goal

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height

    def getState(self) -> list[chr]:
        return self.state
