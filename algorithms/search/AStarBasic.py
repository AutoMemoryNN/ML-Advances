class TState:
    state = None

    def __init__(self, goal: tuple[int, int], position: tuple[int, int], width: int, height: int):
        self.goal = goal
        self.position = position
        self.width = width
        self.height = height

        self.state = ['_'] * (self.width * self.height)
        self.state[(goal[1] * width) + goal[0]] = 'X'
        self.state[(position[1] * width) + position[0]] = 'O'

    def isTerminalState(self) -> bool:
        return self.position == self.goal

    def move(self, newPosition: tuple[int, int]):
        self.state[(self.position[1] * self.width) + self.position[0]] = '_'
        self.state[(newPosition[1] * self.width) + newPosition[0]] = 'O'
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

    def __str__(self):
        state_str = ""
        for i in range(self.height):
            row = self.state[i * self.width:(i + 1) * self.width]
            state_str += " ".join(row) + "\n"

        return (f"Goal: {self.goal}\n"
                f"Position: {self.position}\n"
                f"State:\n{state_str}")


class AStarBasic:
    def __init__(self, startState: TState):
        self.startState = startState
        self.currentState = self.startState

    def initial(self) -> TState:
        return self.startState

    def actionResults(self, e: TState) -> set[TState]:
        successors = set()
        x, y = e.getPosition()
        width, height = e.getWidth(), e.getHeight()

        # Movimiento arriba, abajo, izquierda, derecha (chequeando límites)
        possible_moves = [
            (x - 1, y),  # Izquierda
            (x + 1, y),  # Derecha
            (x, y - 1),  # Arriba
            (x, y + 1)  # Abajo
        ]

        for move in possible_moves:
            if 0 <= move[0] < width and 0 <= move[1] < height:
                new_state = TState(e.getGoal(), move, width, height)
                successors.add(new_state)

        return successors

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


def main():
    state = TState((4 , 4), (0, 0), 5 , 5)

    game = AStarBasic(state)

    for s in game.actionResults(state):
        print(s)

main()