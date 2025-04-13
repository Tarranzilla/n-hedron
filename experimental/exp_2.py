# -*- coding: utf-8 -*-
import pyvista as pv
import numpy as np
import vtk
import logging
from matplotlib.colors import ListedColormap, Colormap # Importar Colormap também
from typing import List, Dict, Optional, Any

# --- Configurações Iniciais ---

# Define DOIS mapas de cores discretos
# Mapa 1: Cores Quentes (Baseado no seu último código)
# Ordem: [Padrão(0), Hover(1), Vizinho(2)]
cmap_quente_list = ['lightcoral', 'red', 'orange'] # Ajustado Padrão para lightcoral para diferenciar
cmap_quente = ListedColormap(cmap_quente_list)
lut_quente = pv.LookupTable(cmap=cmap_quente, n_values=3, scalar_range=(0, 2)) # Criar LookupTable

# Mapa 2: Cores Frias
# Ordem: [Padrão(0), Hover(1), Vizinho(2)]
cmap_frio_list = ['lightsteelblue', 'cyan', 'deepskyblue'] # Exemplo de cores frias
cmap_frio = ListedColormap(cmap_frio_list)
lut_frio = pv.LookupTable(cmap=cmap_frio, n_values=3, scalar_range=(0, 2)) # Criar LookupTable

# Configura o sistema de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
WINDOW_SIZE = [1600, 900]
HIGHLIGHT_DEFAULT = 0
HIGHLIGHT_HOVER = 1
HIGHLIGHT_NEIGHBOR = 2
PICKER_TOLERANCE = 0.005

# --- Variáveis de Estado Globais ---
label_actor: Optional[pv.Actor] = None
current_hovered_info: Dict[str, Any] = {"cellid": -1, "neighbors": []}
show_all_labels: bool = True
selected_cells: List[int] = []
hover_picker: Optional[vtk.vtkCellPicker] = None
mesh_esfera: Optional[pv.PolyData] = None
plotter: Optional[pv.Plotter] = None
face_centers: Optional[np.ndarray] = None
actor: Optional[pv.Actor] = None # Referência ao ator da malha principal
active_lut = lut_quente # Começa com o mapa de cores quente

# =============================================================================
# Função: create_triangulated_sphere (Sem alterações aqui)
# =============================================================================
def create_triangulated_sphere(
    poly_type: str = 'icosahedron', subdivisions: int = 2, radius: float = 1.0
) -> pv.PolyData:
    """ (Docstring e código como antes) """
    logger.info(f"Criando poliedro base: {poly_type}")
    poly_type_lower = poly_type.lower()
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
        raise ValueError("poly_type must be 'tetrahedron', 'octahedron', or 'icosahedron'")

    logger.info(f"Subdividindo {subdivisions} vezes usando o esquema 'loop'...")
    if subdivisions > 0:
        mesh = mesh.subdivide(subdivisions, subfilter='loop')

    logger.info("Normalizando vértices para forma esférica e aplicando raio...")
    vertex_vectors = mesh.points - mesh.center
    norms = np.linalg.norm(vertex_vectors, axis=1)
    norms[norms == 0] = 1.0
    mesh.points = mesh.center + (vertex_vectors / norms[:, np.newaxis]) * radius

    logger.info(f"Malha final criada com {mesh.n_points} vértices e {mesh.n_cells} faces.")
    mesh.cell_data['FaceID'] = np.arange(mesh.n_cells)
    mesh.cell_data['Highlight'] = np.full(mesh.n_cells, HIGHLIGHT_DEFAULT, dtype=int)
    mesh.set_active_scalars('Highlight')
    return mesh

# =============================================================================
# Função: find_cell_neighbors (Sem alterações aqui)
# =============================================================================
def find_cell_neighbors(mesh: pv.PolyData, cellid: int) -> List[int]:
    """ (Docstring e código como antes) """
    if 0 <= cellid < mesh.n_cells:
        try:
            neighbors = mesh.cell_neighbors(cellid)
            return [n for n in neighbors if n >= 0]
        except Exception as e:
            logger.error(f"Erro ao encontrar vizinhos para célula {cellid}: {e}")
            return []
    return []

# =============================================================================
# Função: update_labels (Sem alterações na lógica principal)
# =============================================================================
def update_labels():
    """
    Atualiza os rótulos (texto) exibidos na visualização.
    Usa as variáveis globais plotter, mesh_esfera, face_centers, show_all_labels, label_actor.
    """
    global label_actor, show_all_labels, plotter, mesh_esfera, face_centers

    if not plotter or not mesh_esfera or face_centers is None: return

    if label_actor is not None:
        try:
            plotter.remove_actor(label_actor, render=False)
        except ValueError:
            pass
        label_actor = None

    labels_to_show = []
    points_to_label = []
    highlights = mesh_esfera.cell_data['Highlight']
    face_ids = mesh_esfera.cell_data['FaceID']

    for i in range(mesh_esfera.n_cells):
        highlight_status = highlights[i]
        if show_all_labels or highlight_status != HIGHLIGHT_DEFAULT:
            label_text = f"ID:{face_ids[i]}\nH:{highlight_status}"
            labels_to_show.append(label_text)
            points_to_label.append(face_centers[i])

    if labels_to_show:
        label_actor = plotter.add_point_labels(
            np.array(points_to_label), labels_to_show,
            font_size=14, point_size=5, shape=None,
            show_points=False, pickable=False, render=False
        )
    # Não precisa de render aqui, será chamado por quem invocou a atualização

# =============================================================================
# Função: update_highlight (Leve ajuste para chamar update_labels sem args)
# =============================================================================
def update_highlight(cellid: int):
    """ (Docstring como antes) """
    global current_hovered_info, mesh_esfera, plotter, face_centers

    if mesh_esfera is None or plotter is None or face_centers is None:
        logger.warning("Atualização de destaque ignorada, malha/plotter não prontos.")
        return

    if 0 <= cellid < mesh_esfera.n_cells:
        if cellid == current_hovered_info["cellid"]:
            return

        prev_cellid = current_hovered_info["cellid"]
        if 0 <= prev_cellid < mesh_esfera.n_cells:
            mesh_esfera.cell_data['Highlight'][prev_cellid] = HIGHLIGHT_DEFAULT
            for neighbor_id in current_hovered_info["neighbors"]:
                 if 0 <= neighbor_id < mesh_esfera.n_cells and neighbor_id != cellid:
                    mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_DEFAULT

        mesh_esfera.cell_data['Highlight'][cellid] = HIGHLIGHT_HOVER
        neighbors = find_cell_neighbors(mesh_esfera, cellid)
        for neighbor_id in neighbors:
             if 0 <= neighbor_id < mesh_esfera.n_cells:
                 if mesh_esfera.cell_data['Highlight'][neighbor_id] == HIGHLIGHT_DEFAULT:
                    mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_NEIGHBOR

        current_hovered_info["cellid"] = cellid
        current_hovered_info["neighbors"] = neighbors
        mesh_esfera.set_active_scalars('Highlight')
        update_labels() # Chama a função de atualizar labels (que usa globais)
        plotter.render()

    else:
        prev_cellid = current_hovered_info["cellid"]
        if 0 <= prev_cellid < mesh_esfera.n_cells:
            mesh_esfera.cell_data['Highlight'][prev_cellid] = HIGHLIGHT_DEFAULT
            for neighbor_id in current_hovered_info["neighbors"]:
                if 0 <= neighbor_id < mesh_esfera.n_cells:
                    mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_DEFAULT

            current_hovered_info = {"cellid": -1, "neighbors": []}
            mesh_esfera.set_active_scalars('Highlight')
            update_labels() # Chama a função de atualizar labels
            plotter.render()

# =============================================================================
# Função: vtk_mouse_move_callback (Sem alterações)
# =============================================================================
def vtk_mouse_move_callback(vtk_interactor, event_name):
    """ (Docstring e código como antes) """
    global hover_picker, plotter
    if hover_picker is None or plotter is None or plotter.renderer is None:
        return
    x, y = vtk_interactor.GetEventPosition()
    hover_picker.Pick(x, y, 0, plotter.renderer)
    cellid = hover_picker.GetCellId()
    update_highlight(cellid)

# =============================================================================
# Função: click_callback (Sem alterações)
# =============================================================================
def click_callback(picked_mesh: pv.PolyData, cellid: int):
    """ (Docstring e código como antes) """
    global selected_cells
    if picked_mesh is None or not (0 <= cellid < picked_mesh.n_cells):
        logger.info("Nenhuma célula válida foi clicada.")
        return
    try:
        face_id = picked_mesh.cell_data['FaceID'][cellid]
        logger.info(f"Célula Clicada (Índice): {cellid}, ID da Face: {face_id}")
        if face_id not in selected_cells:
            selected_cells.append(face_id)
        logger.info(f"IDs das Faces Selecionadas: {selected_cells}")
    except (KeyError, IndexError) as e:
        logger.error(f"Erro ao acessar FaceID para célula {cellid}: {e}")

# =============================================================================
# Função: set_label_visibility (NOVA - Lógica central para mostrar/esconder labels)
# =============================================================================
def set_label_visibility(state: bool):
    """Define o estado de visibilidade dos rótulos e atualiza a cena."""
    global show_all_labels, plotter, mesh_esfera, face_centers
    if plotter and mesh_esfera and face_centers is not None:
        if show_all_labels != state: # Só atualiza se o estado realmente mudou
            show_all_labels = state
            logger.info(f"Definindo 'Mostrar Todos os Rótulos' para: {show_all_labels}")
            update_labels() # Atualiza os rótulos com base na nova flag
            plotter.render() # Renderiza a mudança

# =============================================================================
# Função: toggle_labels_callback_key (MODIFICADA - Apenas para a tecla 'L')
# =============================================================================
def toggle_labels_callback_key():
    """Chamada pela tecla 'L' para inverter o estado de visibilidade dos rótulos."""
    global show_all_labels # Precisa saber o estado atual para inverter
    set_label_visibility(not show_all_labels) # Chama a função central com o estado invertido

# =============================================================================
# Função: toggle_colormap_callback (NOVA - Callback para o botão de colormap)
# =============================================================================
def toggle_colormap_callback(state: bool):
    """Alterna entre o mapa de cores quente e frio."""
    global active_lut, actor, plotter
    if actor is None or plotter is None:
        return

    if state: # Se o botão está marcado (vamos associar True com 'quente')
        active_lut = lut_quente
        logger.info("Mapa de cores alterado para: Quente")
    else: # Se o botão não está marcado (associar False com 'frio')
        active_lut = lut_frio
        logger.info("Mapa de cores alterado para: Frio")

    # Aplica o LookupTable (LUT) selecionado ao mapper do ator
    actor.mapper.lookup_table = active_lut
    plotter.render() # Renderiza para mostrar a mudança de cor

# =============================================================================
# Função: main
# Objetivo: Função principal que orquestra a criação e exibição da cena.
# =============================================================================
def main():
    """Função principal que configura e executa a visualização interativa."""
    # Disponibiliza variáveis globais
    global selected_cells, hover_picker, mesh_esfera, plotter, face_centers, actor, active_lut

    # --- 1. Configuração da Esfera ---
    poly_base = 'icosahedron'
    num_subdivisions = 3 # Aumentado para melhor visualização
    esfera_radius = 5.0

    mesh_esfera = create_triangulated_sphere(
        poly_type=poly_base,
        subdivisions=num_subdivisions,
        radius=esfera_radius
    )
    face_centers = mesh_esfera.cell_centers().points

    # --- 2. Configuração do Plotter ---
    pv.set_plot_theme("document")
    plotter = pv.Plotter(window_size=WINDOW_SIZE)

    # Adiciona a malha ao plotter - IMPORTANTE: Usar o active_lut inicial
    # E guardar a referência ao ator na variável global 'actor'
    actor = plotter.add_mesh(
        mesh_esfera,
        scalars='Highlight',
        # cmap=discrete_cmap, # Não usamos mais cmap diretamente aqui
        lookup_table=active_lut, # Usa o LookupTable ativo
        clim=[HIGHLIGHT_DEFAULT, HIGHLIGHT_NEIGHBOR], # Range ainda é 0-2
        show_edges=True, edge_color='gray', line_width=0.5,
        lighting='light kit', preference='cell',
        pickable=True,
        show_scalar_bar=False
    )

    # --- 3. Configuração do Picker VTK para o Hover ---
    hover_picker = vtk.vtkCellPicker()
    hover_picker.SetTolerance(PICKER_TOLERANCE)
    hover_picker.AddPickList(actor)
    hover_picker.PickFromListOn()

    # --- 4. Habilita Interações ---

    # a) Hover VTK (sem mudanças)
    if plotter.iren:
        plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, vtk_mouse_move_callback)
        logger.info("Destaque por hover habilitado (usando Observador VTK MouseMove).")
    else:
        logger.error("Não foi possível obter vtkRenderWindowInteractor (plotter.iren) para adicionar observador.")

    # b) Clique PyVista (sem mudanças)
    plotter.enable_cell_picking(
        callback=lambda mesh, idx: click_callback(mesh, idx),
        show=False, show_message=False
    )
    logger.info("Clique em células habilitado (usando PyVista enable_cell_picking).")

    # c) Tecla 'L' - Chama a função específica da tecla
    plotter.add_key_event("l", toggle_labels_callback_key)
    logger.info("Pressione 'L' para alternar a visibilidade dos rótulos.")

    # --- 5. Adiciona Botões (Widgets) ---

    # a) Botão para alternar Mapa de Cores
    # Posição (x, y) - canto inferior esquerdo, um pouco acima
    # Tamanho (width, height)
    # Borda, Cor, etc.
    # value=True indica que começa marcado (associado ao estado 'quente')
    plotter.add_checkbox_button_widget(
        callback=toggle_colormap_callback,
        value=True, # Começa True (Quente)
        position=(0.02, 0.02), # Posição (x=2%, y=2% a partir do canto inferior esquerdo)
        size=30, # Tamanho do botão
        border_size=1,
        color_on='tomato', # Cor quando marcado (Quente)
        color_off='lightblue', # Cor quando desmarcado (Frio)
        background_color='grey'
    )
    logger.info("Botão para alternar mapa de cores adicionado (inferior esquerdo). Marcado=Quente, Desmarcado=Frio.")


    # b) Botão para alternar Rótulos
    # Coloca um pouco acima do outro botão
    plotter.add_checkbox_button_widget(
        callback=set_label_visibility, # Chama a função que DEFINE o estado
        value=show_all_labels, # Estado inicial baseado na variável global
        position=(0.02, 0.08), # Posição (x=2%, y=8% - acima do outro botão)
        size=30,
        border_size=1,
        color_on='lime', # Cor quando marcado (Mostrar Todos)
        color_off='silver', # Cor quando desmarcado (Mostrar Apenas Destacados)
        background_color='grey'
    )
    logger.info("Botão para alternar rótulos adicionado (inferior esquerdo). Marcado=Todos, Desmarcado=Destacados.")


    # --- 6. Inicializa os Rótulos ---
    update_labels() # Chama a função para mostrar os rótulos iniciais

    # --- 7. Exibe Instruções e a Janela ---
    logger.info("--- Controles ---")
    logger.info(" - Passar Mouse (Hover): Destaca face e vizinhas (cores dependem do modo Quente/Frio)")
    logger.info(" - Clicar Mouse: Seleciona face (registra ID no console)")
    logger.info(" - Tecla 'L' / Botão Inferior (On/Off): Alterna visibilidade dos rótulos")
    logger.info(" - Botão Superior (Vermelho/Azul): Alterna mapa de cores (Quente/Frio)")
    # ... (outras instruções)

    plotter.show()

    logger.info(f"Saindo. IDs das Faces Selecionadas finais: {selected_cells}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    main()