from math import hypot
root2 = pow(2, 1/2)
phi = (5**(1/2) + 1) / 2

class Polyhedron:
    @staticmethod
    def version():
        return "version 1.0"
    
    def __lt__(self, other):
        return self.volume < other.volume
    
    def __gt__(self, other):
        return self.volume > other.volume
    
    def __eq__(self, other):
        return self.volume == other.volume
    
    def scale(self, factor):
        return type(self)(v=self.volume * factor ** 3, e=self.edge * factor)
    
    def __repr__(self):
        return "{}(v={})".format(type(self).__name__, self.volume)

class Tetrahedron(Polyhedron):
    "Self dual, space filler with octahedron"

    def __init__(self, v=1, e=1):
        self.volume = v
        self.edge = e
        self.edges, self.vertexes, self.faces = (6, 4, 4)
        self.name = "Tetrahedron"

class Icosahedron(Polyhedron):
    "Jitterbugs with cuboctahedron"

    def __init__(self, v=5 * root2 * phi**2, e=2):
        self.volume = v
        self.edge = e
        self.edges, self.vertexes, self.faces = (30, 12, 20)
        self.name = "Icosahedron"

class Cuboctahedron(Polyhedron):
    "Dual of RH Dodecahedron"

    def __init__(self, v=20, e=2):
        self.volume = v
        self.edge = e
        self.edges, self.vertexes, self.faces = (24, 12, 14)
        self.name = "Cuboctahedron"

mypolys = [Tetrahedron(), Icosahedron(), Cuboctahedron()]

for poly in mypolys:
    print(poly, "edges:", poly.edges, "vertexes:", poly.vertexes, "faces:", poly.faces)
print("Sorting by volume:")
volume_ordered = sorted(mypolys, key=lambda x: x.volume)
for poly in volume_ordered:
    print(poly, "edges:", poly.edges, "vertexes:", poly.vertexes, "faces:", poly.faces)