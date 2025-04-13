import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para plotagem 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------
# Funções de suporte para geração do poliedro de Goldberg
# ----------------------------------------------------------------

def normalize(v):
    return v / np.linalg.norm(v)

def create_icosahedron():
    """Retorna vértices e faces (índices) de um icosaedro unitário."""
    phi = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [0, -1,  phi],
        [0,  1,  phi],
        [0, -1, -phi],
        [0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float64)
    verts = np.array([normalize(v) for v in verts])
    
    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1]
    ], dtype=np.int32)
    return verts, faces

def subdivide_triangle(vA, vB, vC, n):
    """
    Subdivide um triângulo (vA, vB, vC) em triângulos menores, dividindo
    cada aresta em n segmentos. Os pontos são calculados usando coordenadas
    baricêntricas e projetados na esfera.
    """
    local_verts = []
    index_map = {}  # mapeia (i,j) para o índice do vértice local
    for i in range(n+1):
        for j in range(n+1 - i):
            k = n - i - j
            P = (vB * i + vC * j + vA * k) / n
            P = normalize(P)
            index_map[(i, j)] = len(local_verts)
            local_verts.append(P)
    
    local_faces = []
    for i in range(n):
        for j in range(n - i):
            v0 = index_map[(i, j)]
            v1 = index_map[(i+1, j)]
            v2 = index_map[(i, j+1)]
            local_faces.append([v0, v1, v2])
            if i + j < n - 1:
                v3 = index_map[(i+1, j+1)]
                local_faces.append([v1, v3, v2])
    return local_verts, local_faces

def merge_meshes(global_verts, global_faces, new_verts, new_faces, tol=1e-6):
    """
    Mescla os vértices de uma subdivisão na malha global, evitando duplicatas
    por meio de arredondamento das coordenadas.
    """
    vert_map = {}
    for idx, v in enumerate(global_verts):
        key = tuple(np.round(v, decimals=6))
        vert_map[key] = idx

    face_indices = []
    for face in new_faces:
        global_face = []
        for idx in face:
            v = new_verts[idx]
            key = tuple(np.round(v, decimals=6))
            if key in vert_map:
                global_idx = vert_map[key]
            else:
                global_idx = len(global_verts)
                global_verts.append(v)
                vert_map[key] = global_idx
            global_face.append(global_idx)
        face_indices.append(global_face)
    global_faces.extend(face_indices)
    return global_verts, global_faces

def subdivide_icosahedron(icosa_verts, icosa_faces, n):
    """
    Subdivide cada face triangular do icosaedro com o parâmetro n
    e une os vértices resultantes em uma malha única.
    """
    global_verts = []
    global_faces = []
    for face in icosa_faces:
        vA = icosa_verts[face[0]]
        vB = icosa_verts[face[1]]
        vC = icosa_verts[face[2]]
        local_vs, local_fs = subdivide_triangle(vA, vB, vC, n)
        global_verts, global_faces = merge_meshes(global_verts, global_faces, local_vs, local_fs)
    global_verts = np.array(global_verts)
    global_faces = np.array(global_faces)
    return global_verts, global_faces

def compute_face_centers(verts, faces):
    """
    Calcula o centro (média dos vértices, normalizado) de cada face triangular.
    """
    centers = []
    for face in faces:
        pts = verts[face]
        center = np.mean(pts, axis=0)
        center = normalize(center)
        centers.append(center)
    return np.array(centers)

def compute_dual(verts, faces):
    """
    Computa o dual de uma malha triangular. No dual, cada face original torna-se 
    um vértice (localizado no centro da face original), e cada vértice original
    gera uma face, conectando os centros das faces incidentes.
    """
    face_centers = compute_face_centers(verts, faces)
    
    vertex_to_faces = defaultdict(list)
    for f_idx, face in enumerate(faces):
        for v in face:
            vertex_to_faces[v].append(f_idx)
    
    dual_faces = []
    for v_idx, face_indices in vertex_to_faces.items():
        V = verts[v_idx]
        centers = face_centers[face_indices]
        if np.allclose(V, [0,0,1], atol=1e-6):
            up = np.array([0,1,0])
        else:
            up = np.array([0,0,1])
        e1 = np.cross(V, up)
        if np.linalg.norm(e1) < 1e-6:
            up = np.array([0,1,0])
            e1 = np.cross(V, up)
        e1 = normalize(e1)
        e2 = np.cross(V, e1)
        angles = []
        for center in centers:
            proj = center - np.dot(center, V) * V
            proj = normalize(proj)
            angle = math.atan2(np.dot(proj, e2), np.dot(proj, e1))
            angles.append(angle)
        sorted_indices = np.argsort(angles)
        dual_face = [face_indices[i] for i in sorted_indices]
        dual_faces.append(dual_face)
    
    dual_verts = face_centers
    return dual_verts, dual_faces


def plot_polyhedron(verts, faces, title="Polyhedron"):
    """
    Função de plotagem para visualizar o poliedro.
    Aqui usamos a função auxiliar para garantir que os três eixos tenham a mesma escala.
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plota os vértices
    ax.scatter(verts[:,0], verts[:,1], verts[:,2], color='k', s=10)
    
    # Plota as arestas de cada face
    for face in faces:
        face_points = verts[np.array(face)]
        face_points = np.vstack([face_points, face_points[0]])
        ax.plot(face_points[:,0], face_points[:,1], face_points[:,2], color='b')
    
    ax.set_title(title)
    ax.set_box_aspect((1,1,1))
    plt.show()

def generate_goldberg_polyhedron(m, n):
    """
    Gera um poliedro de Goldberg definido pelos parâmetros (m, n).
    A frequência de subdivisão é N = m+n, a malha triangular é construída
    e o dual gera o poliedro final.
    """
    n_div = m + n
    icosa_verts, icosa_faces = create_icosahedron()
    print(f"Subdividindo cada face do icosaedro em {n_div} segmentos ...")
    verts, tri_faces = subdivide_icosahedron(icosa_verts, icosa_faces, n_div)
    print(f"Esfera geodésica triangulada: {len(verts)} vértices, {len(tri_faces)} faces.")
    
    dual_verts, dual_faces = compute_dual(verts, tri_faces)
    print(f"Poliedro de Goldberg (dual): {len(dual_verts)} vértices, {len(dual_faces)} faces.")
    
    return dual_verts, dual_faces

# ----------------------------------------------------------------
# Classe que define o "mundo" para simulações sociais
# ----------------------------------------------------------------

class GoldbergWorld:
    def _init_(self, m, n):
        """
        Inicializa o mundo usando os parâmetros (m, n) do poliedro de Goldberg.
        Cada face do poliedro será uma célula com um estado (aqui, um número entre 0 e 1).
        """
        self.vertices, self.faces = generate_goldberg_polyhedron(m, n)
        self.num_faces = len(self.faces)
        
        # Inicializa os estados aleatoriamente; você pode modificar a distribuição inicial
        self.face_states = np.random.rand(self.num_faces)
        
        # Calcula as relações de vizinhança entre as faces
        self.adjacency_list = self.compute_face_adjacency()

    def compute_face_adjacency(self):
        """
        Calcula quais faces são vizinhas: duas faces são vizinhas se compartilham uma aresta.
        Cria um dicionário indexado por índices de face.
        """
        adj = {i: set() for i in range(self.num_faces)}
        edge_to_faces = {}
        for i, face in enumerate(self.faces):
            # Para cada aresta na face (fechando o polígono)
            num_vertices = len(face)
            for j in range(num_vertices):
                edge = (face[j], face[(j+1) % num_vertices])
                e_sorted = tuple(sorted(edge))
                edge_to_faces.setdefault(e_sorted, []).append(i)
        # A partir dos pares de faces que compartilham a mesma aresta:
        for faces_sharing in edge_to_faces.values():
            if len(faces_sharing) > 1:
                for i in faces_sharing:
                    for j in faces_sharing:
                        if i != j:
                            adj[i].add(j)
        return adj

    def update(self):
        """
        Atualiza o estado de cada face com base em uma regra simples:
        neste exemplo, a regra é uma espécie de difusão: o novo estado é a média
        do estado atual com os estados dos vizinhos.
        Você pode modificar esta função para inserir regras próprias da simulação.
        """
        new_states = self.face_states.copy()
        for i in range(self.num_faces):
            vizinhos = list(self.adjacency_list[i])
            if len(vizinhos) > 0:
                # Média ponderada entre o estado da célula e a dos vizinhos:
                new_states[i] = (self.face_states[i] + np.sum(self.face_states[vizinhos])) / (len(vizinhos) + 1)
        self.face_states = new_states

    def run_simulation(self, steps=10):
        """
        Executa a simulação por um número definido de passos.
        Em cada passo, atualiza o estado e plota o mundo.
        """
        for step in range(steps):
            self.update()
            self.plot(title=f"Passo da Simulação {step+1}")

    def plot(self, title="Goldberg World"):
        """
        Plota o poliedro onde cada face é colorida de acordo com seu estado.
        Utiliza um mapa de cores (aqui 'viridis') para ilustrar os diferentes estados.
        """
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

        cmap = plt.get_cmap("viridis")
        for i, face in enumerate(self.faces):
            face_points = self.vertices[np.array(face)]
            # Fecha o polígono
            face_points = np.vstack([face_points, face_points[0]])
            # Usa o valor do estado para definir a cor
            state_val = self.face_states[i]
            color = cmap(state_val)
            # Plota as arestas
            ax.plot(face_points[:,0], face_points[:,1], face_points[:,2], color='k', linewidth=0.5)
            # Preenche a face com a cor
            poly = Poly3DCollection([face_points], facecolors=color, edgecolors='k', linewidths=0.5, alpha=0.8)
            ax.add_collection3d(poly)

        ax.set_title(title)
        ax.set_box_aspect((1,1,1))
        plt.show()

# ----------------------------------------------------------------
# Exemplo de uso: criação do mundo e execução de simulação
# ----------------------------------------------------------------

if __name__ == "_main_":
    # Defina os parâmetros do poliedro de Goldberg (m, n)
    m = 2
    n = 1
    
    # Cria o mundo para a simulação
    world = GoldbergWorld(m, n)
    
    # Plota o estado inicial do mundo
    world.plot(title="Estado Inicial do Mundo")
    
    # Executa a simulação por um determinado número de passos
    world.run_simulation(steps=10)