class Tree:
    def __init__(self, node: tuple[int, int]):
        """
        Crea un nuevo árbol con los datos proporcionados como raíz.

        :param node: Los datos del árbol, representados como una tupla (x, y).
        """
        self.root = node
        self.children = []
        self.parent = None

    def addChild(self, node: tuple[int, int], child: tuple[int, int]):
        """
        Busca el nodo dado en el árbol. Cuando se encuentra el nodo,
        se agrega el 'child' como hijo del nodo.

        :param node: El nodo a buscar.
        :param child: El hijo a añadir.
        """
        if child == self.root:
            return
        if node == self.root:
            new_child = Tree(child)
            new_child.parent = self
            self.children.append(new_child)
        else:
            for tree in self.children:
                tree.addChild(node, child)

    def getAllParents(self, leaf: tuple[int, int]):
        """
        Retorna todos los padres de la hoja dada.

        :param leaf: La hoja a buscar.
        :return: La lista de padres de la hoja.
        """
        current = self.findNode(leaf)
        parents = [current]
        while current.parent is not None:
            parents.append(current.parent)
            current = current.parent
        return parents

    def findNode(self, leaf: tuple[int, int]):
        """
        Busca el nodo dado en el árbol.

        :param leaf: El nodo a buscar.
        :return: El nodo si se encuentra, None en caso contrario.
        """
        if self.root == leaf:
            return self
        for tree in self.children:
            found = tree.findNode(leaf)
            if found is not None:
                return found
        return None

    def isLeaf(self):
        """
        Verifica si el nodo es una hoja.

        :return: True si es una hoja, False en caso contrario.
        """
        return len(self.children) == 0

    def isRoot(self):
        """
        Verifica si el nodo es la raíz.

        :return: True si es la raíz, False en caso contrario.
        """
        return self.parent is None

    def __str__(self):
        """
        Representación en string del árbol.

        :return: El string que representa los datos del nodo y sus hijos.
        """
        return f"{self.root} {self.children}"

    def printTree(self, level=0):
        """
        Imprime el árbol de manera jerárquica con indentación adecuada.

        :param level: El nivel actual del nodo (se usa para la indentación).
        """
        indent = " " * (level * 4)
        print(f"{indent}{self.root}")
        for child in self.children:
            child.printTree(level + 1)

    def __str__(self):
        """
        Representación en string del árbol.

        :return: El string que representa los datos del nodo y sus hijos.
        """
        return f"{self.root} {self.children}"
