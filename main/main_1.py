# -*- coding: utf-8 -*- # Boa prática para garantir a codificação correta de caracteres acentuados.
"""
Script para visualização 3D interativa de uma esfera triangularizada usando PyVista e VTK.

Refatorado para usar uma classe `SphereVisualizer` para encapsular estado e comportamento.

Funcionalidades Principais:
- Geração de uma esfera pseudo-geodésica.
- Destaque visual da célula sob hover e múltiplos níveis de vizinhança.
- Alternância do critério de vizinhança ('edges'/'points') com a tecla 'X'.
- Exibição/alternância de rótulos com a tecla 'L'.
- Seleção de faces por clique.
- Exibição de um objeto 'peão' sobre a célula sob hover.
- Interação padrão de câmera.
"""

import pyvista as pv
import numpy as np
import vtk  # Necessário para interações de baixo nível (vtkCellPicker, MouseMoveEvent)
import logging
from matplotlib.colors import ListedColormap
from typing import List, Dict, Optional, Any

# --- Configurações Globais de Logging (fora da classe) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Classe: SphereVisualizer
# =============================================================================

class SphereVisualizer:
    """
    Encapsula a lógica e o estado para criar e interagir com uma visualização
    de esfera triangularizada usando PyVista.
    """

    # --- Constantes da Classe (ou poderiam ser movidas para __init__) ---
    WINDOW_SIZE = [1600, 900]
    HIGHLIGHT_DEFAULT = 0
    HIGHLIGHT_EXTREME_NEIGHBOR = 1
    HIGHLIGHT_EXTENDED_NEIGHBOR = 2
    HIGHLIGHT_NEIGHBOR = 3
    HIGHLIGHT_HOVER = 4
    PICKER_TOLERANCE = 0.005

    # Mapeamento de cores (pode ser configurável no __init__ se desejado)
    DISCRETE_CMAP = ListedColormap(['red', 'orange', 'yellow', 'green', 'lightgreen'])
    # Mapeamento: 0=Red, 1=Orange, 2=Yellow, 3=Green, 4=LightGreen

    # Dimensões do peão (pode ser configurável no __init__)
    SPHERE_RADIUS_PAWN = 0.1
    CONE_RADIUS_PAWN = 0.25
    CONE_HEIGHT_PAWN = 0.5
    CYLINDER_RADIUS_PAWN = 0.25
    CYLINDER_HEIGHT_PAWN = 0.025
    PAWN_OFFSET_DISTANCE = 0.275 # Distância acima da superfície da esfera

    def __init__(self, poly_type: str = 'icosahedron', subdivisions: int = 5, radius: float = 15.0):
        """
        Inicializa o visualizador da esfera.

        Args:
            poly_type (str): Tipo de poliedro base para a esfera ('icosahedron', 'octahedron', etc.).
            subdivisions (int): Número de subdivisões para a malha da esfera.
            radius (float): Raio da esfera principal.
        """
        logger.info("Inicializando SphereVisualizer...")
        self.poly_type = poly_type
        self.subdivisions = subdivisions
        self.radius = radius

        # --- Atributos de Estado da Instância ---
        self.label_actor: Optional[pv.Actor] = None
        self.current_hovered_info: Dict[str, Any] = {
            "cellid": -1, "neighbors": [], "extended_neighbors": [], "extreme_neighbors": []
        }
        self.show_all_labels: Optional[bool] = True # Alterado para permitir None
        self.selected_cells: List[int] = []
        self.hover_picker: Optional[vtk.vtkCellPicker] = None
        self.mesh_esfera: Optional[pv.PolyData] = None
        self.plotter: Optional[pv.Plotter] = None
        self.face_centers: Optional[np.ndarray] = None
        self.face_ids: Optional[np.ndarray] = None # Adicionado para fácil acesso
        self.cell_normals: Optional[np.ndarray] = None # Adicionado para o peão
        self.connections_mode: str = "edges"
        self.pawn_mesh: Optional[pv.PolyData] = None
        self.pawn_actor: Optional[pv.Actor] = None
        self.sphere_actor: Optional[pv.Actor] = None # Referência ao ator da esfera

        # --- Executa a Configuração ---
        self._create_meshes()
        if self.mesh_esfera is None: # Aborta se a malha principal falhar
            logger.critical("Falha ao criar malha da esfera. Encerrando inicialização.")
            return

        self._setup_plotter()
        if self.plotter is None: # Aborta se o plotter falhar
            logger.critical("Falha ao criar o plotter. Encerrando inicialização.")
            return

        self._setup_picker()
        self._setup_interactions()
        self._initialize_display()

        logger.info("SphereVisualizer inicializado com sucesso.")

    # -------------------------------------------------------------------------
    # Métodos de Criação de Malha
    # -------------------------------------------------------------------------
    def _sort_cell_indices_around_vertex(
        self,
        mesh: pv.PolyData,
        point_id: int,
        cell_ids: List[int],
        face_centers: np.ndarray
    ) -> List[int]:
        """
        Sorts the indices of cells surrounding a vertex to form a correctly ordered polygon face.

        Args:
            mesh (pv.PolyData): The original mesh (subdivided icosahedron).
            point_id (int): The index of the vertex in the original mesh around which the cells are centered.
            cell_ids (List[int]): The indices of the cells (faces) sharing the vertex `point_id`.
            face_centers (np.ndarray): The pre-calculated centers of all faces in the original mesh.

        Returns:
            List[int]: The sorted list of cell indices, representing the vertices of the dual face in order.
                       Returns an empty list if sorting is not possible (e.g., < 3 cells).
        """
        if len(cell_ids) < 3:
            return [] # Cannot form a polygon

        # Coordinates of the centers of the relevant faces (these are the dual vertices)
        dual_vertex_coords = face_centers[cell_ids]

        # The coordinate of the original vertex serves as a reference point
        original_vertex_coord = mesh.points[point_id]

        # 1. Calculate the geometric center of the dual face vertices
        dual_face_center = np.mean(dual_vertex_coords, axis=0)

        # 2. Define a plane approximating the dual face.
        #    The normal can be approximated by the vector from the origin to the original vertex
        #    (assuming the mesh is roughly spherical and centered).
        plane_normal = original_vertex_coord - mesh.center # Use center in case mesh moved
        norm_mag = np.linalg.norm(plane_normal)
        if norm_mag < 1e-9: # Handle vertex at center (unlikely for sphere surface)
            plane_normal = np.array([0.0, 0.0, 1.0]) # Arbitrary fallback
        else:
            plane_normal /= norm_mag

        # 3. Create basis vectors (u, v) for the plane using the normal (w)
        #    Create a robust orthogonal basis using cross products.
        if abs(plane_normal[0]) > 0.9: # Avoid parallel with x-axis
            u_vec = np.cross(plane_normal, [0, 1, 0])
        else:
            u_vec = np.cross(plane_normal, [1, 0, 0])
        u_vec /= np.linalg.norm(u_vec)
        v_vec = np.cross(plane_normal, u_vec)
        # v_vec should be normalized if plane_normal and u_vec are

        # 4. Project dual face vertices onto the 2D plane defined by u, v
        #    and calculate angles relative to the dual face center.
        angles = []
        for i, dual_vert_coord in enumerate(dual_vertex_coords):
            vec = dual_vert_coord - dual_face_center # Vector from center to vertex
            proj_u = np.dot(vec, u_vec)
            proj_v = np.dot(vec, v_vec)
            angle = np.arctan2(proj_v, proj_u) # Angle in range [-pi, pi]
            angles.append(angle)

        # 5. Sort the original cell_ids based on the calculated angles
        sorted_indices_in_list = np.argsort(angles)
        sorted_cell_ids = [cell_ids[i] for i in sorted_indices_in_list]

        return sorted_cell_ids

    def _create_pawn_mesh(self) -> Optional[pv.PolyData]:
        """
        Cria a geometria do objeto 'peão' combinando esfera, cone e cilindro.

        Returns:
            Optional[pv.PolyData]: A malha do peão ou None em caso de erro.
        """
        try:
            logger.debug("Criando geometria do peão...")
            # Usa constantes definidas na classe
            sphere = pv.Sphere(radius=self.SPHERE_RADIUS_PAWN, phi_resolution=32, theta_resolution=32)
            cone = pv.Cone(radius=self.CONE_RADIUS_PAWN, height=self.CONE_HEIGHT_PAWN, resolution=32)
            cylinder = pv.Cylinder(radius=self.CYLINDER_RADIUS_PAWN, height=self.CYLINDER_HEIGHT_PAWN, resolution=32)

            # Posiciona as partes relativamente (orientação padrão 'para cima' no eixo Z)
                # Offset the cone and cylinder relative to the sphere
            cone_offset = [0, 0, 0]  # Move the cone below the sphere
            cylinder_offset = [-0.2625, 0, 0]  # Move the cylinder below the cone
            sphere_offset = [0.25, 0, 0]  # Move the sphere above the cone and cylinder

            # A VTK/PyVista frequentemente usa Z como eixo vertical padrão para primitivas
            sphere.translate(sphere_offset, inplace=True)
            cone.translate(cone_offset, inplace=True)
            cylinder.translate(cylinder_offset, inplace=True) # Cilindro na base z=0

            # Combina as partes
            pawn = sphere + cone + cylinder
            # Centraliza o peão na origem antes de futuras translações/rotações
            # pawn.translate(-np.array(pawn.center), inplace=True)
            logger.debug("Geometria do peão criada.")
            return pawn
        except Exception as e:
            logger.error(f"Erro ao criar geometria do peão: {e}", exc_info=True)
            return None

    def _create_triangulated_sphere(self) -> Optional[pv.PolyData]:
        """
        Cria a malha principal da esfera triangularizada.

        Baseado na função original `create_triangulated_sphere`.

        Returns:
            Optional[pv.PolyData]: A malha da esfera ou None em caso de erro.
        """
        logger.info(f"Criando esfera: base={self.poly_type}, subdiv={self.subdivisions}, raio={self.radius}")
        poly_type_lower = self.poly_type.lower()

        try:
            # 1. Cria o poliedro base
            if poly_type_lower == 'tetrahedron':
                mesh = pv.Tetrahedron()
            elif poly_type_lower == 'octahedron':
                mesh = pv.Octahedron()
            elif poly_type_lower == 'icosahedron':
                phi = (1.0 + np.sqrt(5.0)) / 2.0
                vertices = np.array([
                    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
                ]) / np.sqrt(1 + phi**2)
                faces = np.array([
                    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
                ])
                faces_pv = np.hstack((np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces)).flatten()
                mesh = pv.PolyData(vertices, faces=faces_pv)
            else:
                raise ValueError(f"poly_type inválido: {self.poly_type}")

            # 2. Subdivide
            if self.subdivisions > 0:
                mesh = mesh.subdivide(self.subdivisions, subfilter='loop')

            # 3. Normaliza e aplica raio
            vertex_vectors = mesh.points - mesh.center
            norms = np.linalg.norm(vertex_vectors, axis=1)
            norms[norms == 0] = 1.0
            mesh.points = mesh.center + (vertex_vectors / norms[:, np.newaxis]) * self.radius

            # 4. Adiciona dados às células
            mesh.cell_data['FaceID'] = np.arange(mesh.n_cells)
            mesh.cell_data['Highlight'] = np.full(mesh.n_cells, self.HIGHLIGHT_DEFAULT, dtype=int)
            mesh.set_active_scalars('Highlight')

            # 5. Calcula e armazena dados derivados importantes
            self.face_ids = mesh.cell_data['FaceID']
            self.face_centers = mesh.cell_centers().points
            mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True) # Calcula normais das células
            self.cell_normals = mesh.cell_normals

            logger.info(f"Malha esférica criada: {mesh.n_points} vértices, {mesh.n_cells} faces.")
            return mesh

        except Exception as e:
            logger.error(f"Erro ao criar malha da esfera: {e}", exc_info=True)
            return None

    def _create_goldberg_polyhedron(self) -> Optional[pv.PolyData]:
        """
        Creates a Goldberg polyhedron mesh by manually computing the dual
        of a subdivided icosahedron.

        Args:
            radius (float, optional): Final radius of the spherical polyhedron. Defaults to 1.0.
            subdivisions (int, optional): Subdivisions for the base icosahedron. Defaults to 1.

        Returns:
            Optional[pv.PolyData]: Goldberg polyhedron mesh, or None on error.
        """
        try:
            logger.info(f"Creating Goldberg Polyhedron (Manual Dual): subdivisions={self.subdivisions}, radius={self.radius}")

            # Step 1 & 2: Create and subdivide the icosahedron (Geodesic Dome)
            base_icosahedron = pv.Icosahedron()
            if self.subdivisions > 0:
                subdivided_mesh = base_icosahedron.subdivide(self.subdivisions, subfilter='loop')
            else:
                subdivided_mesh = base_icosahedron
                logger.debug(f"Base geodesic mesh created with {subdivided_mesh.n_points} points, {subdivided_mesh.n_cells} cells.")
                subdivided_mesh.clean(inplace=True) # Important for connectivity info

            # --- Step 3: Manual Dual Computation ---
            logger.debug("Starting manual dual computation...")

            # D1: Vertices of the dual mesh are the face centers of the original
            face_centers = subdivided_mesh.cell_centers().points
            n_dual_vertices = subdivided_mesh.n_cells # Number of faces = number of dual vertices
            if n_dual_vertices == 0:
                raise ValueError("Subdivided mesh has no faces to compute dual.")
            logger.debug(f"Calculated {n_dual_vertices} face centers (dual vertices).")

            # D2: Faces of the dual mesh correspond to vertices of the original
            dual_faces_list = [] # Build the VTK faces array [n_verts1, v0, v1,..., n_verts2, v0, v1,...]
            processed_dual_faces = 0
            for point_id in range(subdivided_mesh.n_points):
                # Find all cells (faces) sharing the current point (vertex)
                # Using get_point_cells might require building connectivity first
                try:
                    # Ensure connectivity is built if needed (depends on PyVista version/operations)
                    # subdivided_mesh.build_links() # Might be needed
                    cell_ids_sharing_point = subdivided_mesh.get_point_cells(point_id)
                except AttributeError:
                    # Fallback or older PyVista: Manually find cells sharing the point
                    logger.warning("Using manual search for cells sharing point (slower).")
                    cell_ids_sharing_point = [cid for cid in range(subdivided_mesh.n_cells) if point_id in subdivided_mesh.get_cell(cid).point_ids]

                num_sharing = len(cell_ids_sharing_point)

                # Expect 5 or 6 cells sharing a vertex on a closed sphere from subdivided icosahedron
                if num_sharing == 5 or num_sharing == 6:
                    # These cell IDs correspond to the indices in face_centers array
                    # Sort these indices based on their geometric arrangement around the vertex
                    sorted_dual_vertex_indices = self._sort_cell_indices_around_vertex(
                        subdivided_mesh, point_id, cell_ids_sharing_point, face_centers
                    )

                    if sorted_dual_vertex_indices:
                        # Append [n_vertices, v0_idx, v1_idx, ...] to the list
                        dual_faces_list.append(num_sharing)
                        dual_faces_list.extend(sorted_dual_vertex_indices)
                        processed_dual_faces += 1
                    else:
                        logger.warning(f"Could not sort vertices for dual face corresponding to original point {point_id}.")
                # else: # Optional: Log unexpected connectivity
                #    if num_sharing > 0: logger.warning(f"Original vertex {point_id} shared by unexpected number of cells: {num_sharing}")

            if not dual_faces_list:
                raise ValueError("Failed to construct any faces for the dual mesh.")

            logger.debug(f"Constructed {processed_dual_faces} dual faces (pentagons/hexagons).")
            # Convert the face list to a NumPy array for PyVista
            dual_faces_array = np.array(dual_faces_list, dtype=pv.ID_TYPE)

            # D3: Create the dual PolyData object
            goldberg_mesh = pv.PolyData(face_centers, faces=dual_faces_array)
            logger.debug("Initial Goldberg PolyData object created from dual.")

            # --- Step 4: Project vertices onto sphere ---
            origin = goldberg_mesh.center # Use its own center now
            vertex_vectors = goldberg_mesh.points - origin
            norms = np.linalg.norm(vertex_vectors, axis=1)
            norms[norms == 0] = 1.0
            goldberg_mesh.points = origin + (vertex_vectors / norms[:, np.newaxis]) * self.radius
            logger.debug("Goldberg vertices projected onto sphere.")

            # --- Step 5: Clean and Add Data ---
            goldberg_mesh.clean(inplace=True) # Clean after projection
            goldberg_mesh.cell_data['FaceID'] = np.arange(goldberg_mesh.n_cells)
            goldberg_mesh.cell_data['Highlight'] = np.full(goldberg_mesh.n_cells, 0, dtype=int)
            goldberg_mesh.set_active_scalars('Highlight')
            logger.debug("Final cleaning and cell data added.")

            # Final check on cell types
            cell_n_points = [goldberg_mesh.get_cell(i).n_points for i in range(goldberg_mesh.n_cells)]
            unique_counts, counts = np.unique(cell_n_points, return_counts=True)
            logger.info(f"Final cell types: {dict(zip(unique_counts, counts))}")

            # --- Step 6: Update Instance Attributes ---
            self.face_centers = goldberg_mesh.cell_centers().points
            self.face_ids = goldberg_mesh.cell_data['FaceID']
            goldberg_mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
            self.cell_normals = goldberg_mesh.cell_normals

            logger.info(f"Goldberg polyhedron created successfully: {goldberg_mesh.n_points} points, {goldberg_mesh.n_cells} cells.")
            return goldberg_mesh

        except ValueError as ve:
            logger.error(f"Value error during Goldberg creation: {ve}", exc_info=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating Goldberg polyhedron: {e}", exc_info=True)
            return None

    def _create_meshes(self):
        """Cria a malha da esfera principal e a malha do peão."""
        #self.mesh_esfera = self._create_triangulated_sphere()
        self.mesh_esfera = self._create_goldberg_polyhedron()
        self.pawn_mesh = self._create_pawn_mesh()

    # -------------------------------------------------------------------------
    # Métodos de Configuração da Visualização
    # -------------------------------------------------------------------------

    def _setup_plotter(self):
        """Configura o Plotter PyVista e adiciona os atores principais."""
        try:
            pv.set_plot_theme("document")
            self.plotter = pv.Plotter(window_size=self.WINDOW_SIZE, title="Visualizador de Esfera (Classe)")

            # Adiciona ator da esfera principal
            if self.mesh_esfera:
                self.sphere_actor = self.plotter.add_mesh(
                    self.mesh_esfera,
                    scalars='Highlight',
                    cmap=self.DISCRETE_CMAP,
                    clim=[self.HIGHLIGHT_DEFAULT, self.HIGHLIGHT_HOVER],
                    show_edges=True,
                    edge_color='black',
                    line_width=0.5,
                    lighting='light kit',
                    preference='cell',
                    pickable=True,
                    show_scalar_bar=False
                )
                logger.info("Ator da esfera principal adicionado ao plotter.")
            else:
                 logger.error("Malha da esfera não disponível para adicionar ao plotter.")
                 self.plotter = None # Invalida plotter se malha principal falhou
                 return

            # Adiciona ator do peão
            if self.pawn_mesh:
                self.pawn_actor = self.plotter.add_mesh(
                    self.pawn_mesh,
                    color='pink',
                    show_edges=True,
                    edge_color='black',
                    line_width=0.5,
                    opacity=1,
                    pickable=False # Peão não interfere na seleção
                )
                self.pawn_actor.SetVisibility(False) # Começa invisível
                logger.info("Ator do peão adicionado ao plotter.")
            else:
                logger.warning("Malha do peão não disponível, ator não adicionado.")

        except Exception as e:
            logger.critical(f"Falha ao configurar o plotter ou adicionar atores: {e}", exc_info=True)
            self.plotter = None # Invalida o plotter em caso de erro grave

    def _setup_picker(self):
        """Configura o vtkCellPicker para detecção de hover."""
        if self.plotter is None or self.sphere_actor is None:
            logger.error("Picker não pode ser configurado: plotter ou ator da esfera ausente.")
            return
        try:
            self.hover_picker = vtk.vtkCellPicker()
            self.hover_picker.SetTolerance(self.PICKER_TOLERANCE)
            self.hover_picker.AddPickList(self.sphere_actor) # Pick apenas na esfera
            self.hover_picker.PickFromListOn()
            logger.info("vtkCellPicker configurado.")
        except Exception as e:
             logger.error(f"Falha ao configurar o vtkCellPicker: {e}", exc_info=True)
             self.hover_picker = None

    def _setup_interactions(self):
        """Registra os callbacks para interações de mouse e teclado."""
        if self.plotter is None:
            logger.error("Interações não podem ser configuradas: plotter ausente.")
            return

        try:
            # a) Hover (Movimento do Mouse)
            if self.plotter.iren and self.hover_picker:
                # Passa o método da instância como callback
                self.plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, self._vtk_mouse_move_callback)
                logger.info("Callback de hover registrado.")
            else:
                logger.warning("Não foi possível registrar callback de hover (iren ou picker ausente).")

            # b) Clique na Célula
            # Passa o método da instância diretamente
            self.plotter.enable_cell_picking(
                callback=self._click_callback,
                show=False,
                show_message=False,
                use_picker=False
            )
            logger.info("Callback de clique registrado.")

            # c) Tecla 'L' (Alternar Rótulos)
            self.plotter.add_key_event("l", self._toggle_labels_callback)
            logger.info("Callback da tecla 'L' registrado.")

            # d) Tecla 'X' (Alternar Modo de Conexão)
            self.plotter.add_key_event("x", self._toggle_connections_mode)
            logger.info("Callback da tecla 'X' registrado.")

        except Exception as e:
            logger.error(f"Erro ao configurar interações: {e}", exc_info=True)

    def _initialize_display(self):
        """Chamadas iniciais para configurar a exibição."""
        if self.plotter and self.mesh_esfera and self.face_centers is not None:
            logger.info("Inicializando exibição dos rótulos...")
            self.update_labels() # Chama o método da instância
        else:
             logger.warning("Não foi possível inicializar os rótulos (plotter/mesh/centers ausentes).")

    # -------------------------------------------------------------------------
    # Métodos de Lógica de Interação (Antigas Funções)
    # -------------------------------------------------------------------------

    def find_cell_neighbors(self, cellid: int, optional_mode: Optional[str] = None) -> List[int]:
        """Encontra vizinhos da célula (agora um método)."""
        if self.mesh_esfera is None: return [] # Segurança

        mode = optional_mode if optional_mode in ["points", "edges"] else self.connections_mode
        if 0 <= cellid < self.mesh_esfera.n_cells:
            try:
                neighbors = self.mesh_esfera.cell_neighbors(cellid, connections=mode)
                return [n for n in neighbors if n >= 0]
            except Exception as e:
                logger.error(f"Erro ao encontrar vizinhos para célula {cellid} (modo '{mode}'): {e}", exc_info=True)
                return []
        return []

    def update_labels(self):
        """Atualiza os rótulos de texto (agora um método)."""
        # Usa atributos da instância (self.plotter, self.mesh_esfera, etc.)
        if not self.plotter or not self.mesh_esfera or self.face_centers is None or self.face_ids is None:
            logger.warning("Update labels ignorado: estado interno incompleto.")
            return

        # Remove ator antigo
        if self.label_actor is not None:
            try:
                self.plotter.remove_actor(self.label_actor, render=False)
            except ValueError:
                pass # Ignora erro se já removido
            finally:
                 self.label_actor = None

        if self.show_all_labels is None: return # Não mostra nada

        # Prepara novos rótulos
        labels_to_show = []
        points_to_label = []
        highlights = self.mesh_esfera.cell_data['Highlight']

        for i in range(self.mesh_esfera.n_cells):
            highlight_status = highlights[i]
            if self.show_all_labels or highlight_status != self.HIGHLIGHT_DEFAULT:
                label_text = f"ID:{self.face_ids[i]}\nH:{highlight_status}"
                labels_to_show.append(label_text)
                points_to_label.append(self.face_centers[i])

        # Adiciona novos rótulos
        if labels_to_show:
            self.label_actor = self.plotter.add_point_labels(
                np.array(points_to_label), labels_to_show,
                font_size=12, point_size=1, shape=None,
                show_points=False, pickable=False, render=False
            )

    def update_highlight(self, cellid: int, force_render: bool = False):
        """Atualiza o destaque visual (agora um método)."""
        # Usa atributos da instância (self.mesh_esfera, self.plotter, etc.)
        if self.mesh_esfera is None or self.plotter is None or self.face_centers is None or self.cell_normals is None:
            logger.warning("Update highlight ignorado: estado interno incompleto.")
            return

        needs_render = False

        if 0 <= cellid < self.mesh_esfera.n_cells:
            if cellid == self.current_hovered_info["cellid"] and not force_render:
                return

            needs_render = True
            highlights = self.mesh_esfera.cell_data['Highlight']

            # Atualiza posição/visibilidade do peão
            if self.pawn_actor is not None:
                cell_center = self.face_centers[cellid]
                cell_normal = self.cell_normals[cellid]
                pawn_position = cell_center + cell_normal * self.PAWN_OFFSET_DISTANCE

                # Cálculo da rotação (orientação Z padrão para o peão)
                default_up = np.array([1, 0, 0]) # Assume que o peão aponta para +Z por padrão
                # Normaliza o vetor normal da célula apenas para o cálculo do ângulo/eixo
                cell_normal_norm = cell_normal / np.linalg.norm(cell_normal)
                rotation_axis = np.cross(default_up, cell_normal_norm)
                dot_prod = np.dot(default_up, cell_normal_norm)
                # Corrige possível imprecisão numérica para np.arccos
                dot_prod = np.clip(dot_prod, -1.0, 1.0)
                rotation_angle = np.arccos(dot_prod) * (180.0 / np.pi)

                # Aplica a rotação se o eixo for válido
                # Resetar orientação antes de aplicar nova rotação pode ser necessário dependendo do ator VTK
                self.pawn_actor.SetOrientation(0, 0, 0) # Reset
                if np.linalg.norm(rotation_axis) > 1e-6:
                    self.pawn_actor.RotateWXYZ(rotation_angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])

                self.pawn_actor.SetPosition(pawn_position)
                self.pawn_actor.SetVisibility(True)
                #logger.debug(f"Peão movido para cell {cellid}") # Log opcional

            # Reset dos destaques anteriores
            prev_cellid = self.current_hovered_info["cellid"]
            prev_neighbors = self.current_hovered_info["neighbors"]
            prev_extended = self.current_hovered_info["extended_neighbors"]
            prev_extreme = self.current_hovered_info["extreme_neighbors"]
            cells_to_potentially_reset = set()
            if 0 <= prev_cellid < self.mesh_esfera.n_cells: cells_to_potentially_reset.add(prev_cellid)
            cells_to_potentially_reset.update(prev_neighbors)
            cells_to_potentially_reset.update(prev_extended)
            cells_to_potentially_reset.update(prev_extreme)

            for reset_id in cells_to_potentially_reset:
                 if 0 <= reset_id < self.mesh_esfera.n_cells and reset_id != cellid:
                     if highlights[reset_id] != self.HIGHLIGHT_DEFAULT:
                        highlights[reset_id] = self.HIGHLIGHT_DEFAULT

            # Cálculo e aplicação dos novos destaques (Nível 0, 1, 2, 3)
            highlights[cellid] = self.HIGHLIGHT_HOVER
            neighbors = self.find_cell_neighbors(cellid)
            neighbors = [n for n in neighbors if n != cellid]
            for neighbor_id in neighbors:
                if 0 <= neighbor_id < self.mesh_esfera.n_cells:
                    highlights[neighbor_id] = self.HIGHLIGHT_NEIGHBOR

            extended_neighbors = set()
            for neighbor_id in neighbors:
                extended = self.find_cell_neighbors(neighbor_id)
                for ext_id in extended:
                    if 0 <= ext_id < self.mesh_esfera.n_cells and ext_id != cellid and ext_id not in neighbors:
                        extended_neighbors.add(ext_id)
            for extended_neighbor_id in extended_neighbors:
                 if highlights[extended_neighbor_id] < self.HIGHLIGHT_NEIGHBOR:
                    highlights[extended_neighbor_id] = self.HIGHLIGHT_EXTENDED_NEIGHBOR

            extreme_neighbors = set()
            for extended_neighbor_id in extended_neighbors:
                extreme = self.find_cell_neighbors(extended_neighbor_id)
                for extreme_id in extreme:
                    if (0 <= extreme_id < self.mesh_esfera.n_cells and
                            extreme_id != cellid and
                            extreme_id not in neighbors and
                            extreme_id not in extended_neighbors):
                        extreme_neighbors.add(extreme_id)
            for extreme_neighbor_id in extreme_neighbors:
                if highlights[extreme_neighbor_id] < self.HIGHLIGHT_EXTENDED_NEIGHBOR:
                    highlights[extreme_neighbor_id] = self.HIGHLIGHT_EXTREME_NEIGHBOR

            # Atualização do estado global
            self.current_hovered_info["cellid"] = cellid
            self.current_hovered_info["neighbors"] = neighbors
            self.current_hovered_info["extended_neighbors"] = list(extended_neighbors)
            self.current_hovered_info["extreme_neighbors"] = list(extreme_neighbors)

        else: # Mouse fora da malha
            if self.pawn_actor is not None: self.pawn_actor.SetVisibility(False)

            prev_cellid = self.current_hovered_info["cellid"]
            if prev_cellid != -1:
                needs_render = True
                highlights = self.mesh_esfera.cell_data['Highlight']
                cells_to_reset = {prev_cellid} | set(self.current_hovered_info["neighbors"]) | \
                                 set(self.current_hovered_info["extended_neighbors"]) | \
                                 set(self.current_hovered_info["extreme_neighbors"])
                for reset_id in cells_to_reset:
                     if 0 <= reset_id < self.mesh_esfera.n_cells:
                         if highlights[reset_id] != self.HIGHLIGHT_DEFAULT:
                            highlights[reset_id] = self.HIGHLIGHT_DEFAULT

                self.current_hovered_info = {
                    "cellid": -1, "neighbors": [], "extended_neighbors": [], "extreme_neighbors": []
                }

        # Atualização Visual
        if needs_render:
            self.mesh_esfera.set_active_scalars('Highlight')
            self.update_labels()
            self.plotter.render()

    # -------------------------------------------------------------------------
    # Métodos de Callback (Registrados nas Interações)
    # -------------------------------------------------------------------------

    def _vtk_mouse_move_callback(self, vtk_interactor: vtk.vtkRenderWindowInteractor, event_name: str):
        """Callback para evento de movimento do mouse VTK (agora um método)."""
        if self.hover_picker is None or self.plotter is None or self.plotter.renderer is None:
            return
        x, y = vtk_interactor.GetEventPosition()
        self.hover_picker.Pick(x, y, 0, self.plotter.renderer)
        cellid = self.hover_picker.GetCellId()
        self.update_highlight(cellid) # Chama método da instância

    def _click_callback(self, picked_mesh: Optional[pv.PolyData], cellid: int):
        """Callback para clique em célula PyVista (agora um método)."""
        if picked_mesh is None or not (0 <= cellid < picked_mesh.n_cells) or self.face_ids is None:
            logger.info("Clique fora de uma célula válida.")
            return
        try:
            face_id = self.face_ids[cellid] # Usa face_ids da instância
            logger.info(f"Célula Clicada! Índice VTK: {cellid}, Nosso FaceID: {face_id}")
            if face_id not in self.selected_cells:
                self.selected_cells.append(face_id)
                logger.info(f"FaceID {face_id} adicionado à seleção.")
            else:
                logger.info(f"FaceID {face_id} já estava na lista.")
            logger.info(f"IDs Selecionados: {sorted(self.selected_cells)}")
        except (KeyError, IndexError) as e:
            logger.error(f"Erro ao acessar FaceID para célula {cellid}: {e}", exc_info=True)
        except Exception as e:
             logger.error(f"Erro inesperado no callback de clique: {e}", exc_info=True)

    def _toggle_labels_callback(self):
        """Callback para tecla 'L' (agora um método)."""
        if self.plotter and self.mesh_esfera and self.face_centers is not None:
            # Cicla entre True -> False -> None -> True
            if self.show_all_labels is True: self.show_all_labels = False
            elif self.show_all_labels is False: self.show_all_labels = None
            else: self.show_all_labels = True

            logger.info(f"Alternando estado de rótulos para: {self.show_all_labels}")
            self.update_labels() # Chama método da instância
            self.plotter.render()
        else:
            logger.warning("Toggle labels ignorado: estado interno incompleto.")

    def _toggle_connections_mode(self):
        """Callback para tecla 'X' (agora um método)."""
        self.connections_mode = "points" if self.connections_mode == "edges" else "edges"
        logger.info(f"Modo de conexão alternado para: '{self.connections_mode}'")
        # Força atualização do highlight para refletir novo modo de vizinhança
        if self.current_hovered_info["cellid"] >= 0:
            self.update_highlight(self.current_hovered_info["cellid"], force_render=True) # Chama método

    # -------------------------------------------------------------------------
    # Método Principal de Execução
    # -------------------------------------------------------------------------

    def run(self):
        """Inicia a visualização interativa."""
        if self.plotter is None:
             logger.critical("Visualização não pode ser iniciada: Plotter não foi criado.")
             return

        # Log das instruções
        logger.info("--- Controles ---")
        logger.info(f" - Hover Mouse: Destaca face ({self.DISCRETE_CMAP.colors[-1].upper()}) e vizinhos")
        logger.info(f"   - Nível 1: ({self.DISCRETE_CMAP.colors[self.HIGHLIGHT_NEIGHBOR].upper()})")
        logger.info(f"   - Nível 2: ({self.DISCRETE_CMAP.colors[self.HIGHLIGHT_EXTENDED_NEIGHBOR].upper()})")
        logger.info(f"   - Nível 3: ({self.DISCRETE_CMAP.colors[self.HIGHLIGHT_EXTREME_NEIGHBOR].upper()})")
        logger.info(f" - Clique Mouse: Seleciona face")
        logger.info(f" - Tecla 'L': Cicla visibilidade dos rótulos (Todos / Destacados / Nenhum)")
        logger.info(f" - Tecla 'X': Alterna critério de vizinhança (Arestas / Pontos)")
        logger.info(f" - Controles Padrão de Câmera (Mouse)")
        logger.info("-----------------")
        logger.info("Iniciando loop de eventos da janela. Feche a janela para sair.")

        # Mostra a janela e inicia o loop de eventos
        self.plotter.show()

        # Após fechar a janela
        logger.info("Janela fechada. Encerrando a aplicação.")
        logger.info(f"IDs das Faces Selecionadas finais: {sorted(self.selected_cells)}")


# =============================================================================
# Ponto de Entrada Principal do Script
# =============================================================================
if __name__ == "__main__":
    # Cria uma instância da classe visualizadora
    visualizer = SphereVisualizer(
        poly_type='icosahedron',
        subdivisions=2, # Ajuste conforme necessário
        radius=5.0     # Ajuste conforme necessário
    )
    # Inicia a aplicação
    visualizer.run()