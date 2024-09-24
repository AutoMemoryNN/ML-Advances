from algorithms.search.utils.Tree import Tree


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


class AStarBasic:
    def __init__(self, startState: TState):
        self.startState = startState
        self.currentState = self.startState

    def initial(self) -> TState:
        return self.startState

    def isGoal(self, e: TState) -> bool:
        return e.isTerminalState()

    def actionCost(self, e1: TState, e2: TState) -> int:
        # Distancia de Manhattan como costo de avance
        x1, y1 = e1.getPosition()
        x2, y2 = e2.getPosition()
        return abs(x1 - x2) + abs(y1 - y2)

    def heuristicCost(self, e: TState) -> int:
        # Distancia de Manhattan entre la posición actual y la meta
        x, y = e.getPosition()
        goal_x, goal_y = e.getGoal()
        return abs(x - goal_x) + abs(y - goal_y)
