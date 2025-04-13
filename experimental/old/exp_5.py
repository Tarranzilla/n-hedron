# -*- coding: utf-8 -*-
"""
Visualização 3D interativa de uma esfera triangularizada usando PyVista e VTK.

Funcionalidades:
- Criação de uma esfera pseudo-geodésica a partir de um icosaedro subdividido.
- Destaque da face (célula) sob o cursor do mouse (hover).
- Destaque das faces vizinhas à face sob hover.
- Exibição de rótulos com ID da face e status de destaque.
- Alternância da visibilidade de todos os rótulos (tecla 'L').
- Seleção de faces por clique (registra o ID no log).
"""

import pyvista as pv
import numpy as np
import vtk  # Necessário para vtkCellPicker e eventos VTK (MouseMoveEvent)
import logging
from matplotlib.colors import ListedColormap
from typing import List, Dict, Optional, Any

# --- Configurações Iniciais ---

# Define um mapa de cores discreto para os valores de Destaque (Highlight)
# Mapeamento: 0 -> Padrão (Vermelho), 1 -> Hover (Amarelo), 2 -> Vizinho (Laranja)
# Ajuste as cores conforme desejado.
discrete_cmap = ListedColormap(['red', 'yellow', 'orange'])

# Configura o sistema de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Constantes ---
WINDOW_SIZE = [1600, 900]  # Tamanho da janela de visualização em pixels
HIGHLIGHT_DEFAULT = 0     # Valor para estado normal/padrão da célula
HIGHLIGHT_HOVER = 1       # Valor para célula sob o mouse (hover)
HIGHLIGHT_NEIGHBOR = 2    # Valor para célula vizinha da célula sob hover
PICKER_TOLERANCE = 0.005  # Tolerância para vtkCellPicker (sensibilidade do hover)

# --- Variáveis de Estado Globais ---
# Usadas para compartilhar informações entre callbacks.
# Alternativa para aplicações maiores: encapsular em uma classe.
label_actor: Optional[pv.Actor] = None # Ator dos rótulos (para remoção/atualização)
current_hovered_info: Dict[str, Any] = {"cellid": -1, "neighbors": []} # Info da célula atual sob hover
show_all_labels: bool = True      # Controla visibilidade dos rótulos
selected_cells: List[int] = []    # Lista de FaceIDs das células clicadas
hover_picker: Optional[vtk.vtkCellPicker] = None # Picker VTK para detectar hover
mesh_esfera: Optional[pv.PolyData] = None       # Malha da esfera
plotter: Optional[pv.Plotter] = None          # Objeto Plotter do PyVista
face_centers: Optional[np.ndarray] = None     # Coordenadas do centro das faces
face_ids: Optional[np.ndarray] = None         # IDs únicos das faces (para referência)

# =============================================================================
# Função: create_triangulated_sphere
# =============================================================================
def create_triangulated_sphere(
    poly_type: str = 'icosahedron', subdivisions: int = 2, radius: float = 1.0
) -> pv.PolyData:
    """Cria uma malha pseudo-esférica triangularizada.

    Inicia com um poliedro base, o subdivide para aumentar a resolução
    e normaliza os vértices para formar uma esfera com o raio especificado.

    Args:
        poly_type (str): Tipo do poliedro base ('tetrahedron', 'octahedron', 'icosahedron').
                         Default: 'icosahedron'.
        subdivisions (int): Número de subdivisões ('loop' filter) para refinar a malha.
                            Mais subdivisões resultam em uma esfera mais suave. Default: 2.
        radius (float): Raio final da esfera gerada. Default: 1.0.

    Returns:
        pv.PolyData: O objeto de malha (mesh) da esfera criada, com dados
                     'FaceID' e 'Highlight' adicionados às células.

    Raises:
        ValueError: Se `poly_type` for inválido.
    """
    logger.info(f"Criando poliedro base: {poly_type}")
    poly_type_lower = poly_type.lower()

    # 1. Cria o poliedro base
    if poly_type_lower == 'tetrahedron':
        mesh = pv.Tetrahedron()
    elif poly_type_lower == 'octahedron':
        mesh = pv.Octahedron()
    elif poly_type_lower == 'icosahedron':
        # Definição manual do icosaedro (alternativa: pv.Icosahedron())
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]) / np.sqrt(1 + phi**2) # Normaliza para raio 1 inicial
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        # Formato PyVista: [num_pontos, p1, p2, p3, ...]
        faces_pv = np.hstack((np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces)).flatten()
        mesh = pv.PolyData(vertices, faces=faces_pv)
    else:
        raise ValueError("poly_type deve ser 'tetrahedron', 'octahedron', ou 'icosahedron'")

    # 2. Subdivide a malha base (esquema 'loop' para triângulos)
    if subdivisions > 0:
        logger.info(f"Subdividindo {subdivisions} vezes usando o esquema 'loop'...")
        mesh = mesh.subdivide(subdivisions, subfilter='loop')

    # 3. Normaliza os vértices para formar esfera e aplica raio
    logger.info("Normalizando vértices para forma esférica e aplicando raio...")
    vertex_vectors = mesh.points - mesh.center
    norms = np.linalg.norm(vertex_vectors, axis=1)
    # Previne divisão por zero se algum ponto estiver exatamente no centro
    norms[norms == 0] = 1.0
    # Normaliza para raio 1 e então escala para o raio desejado
    mesh.points = mesh.center + (vertex_vectors / norms[:, np.newaxis]) * radius

    logger.info(f"Malha final criada com {mesh.n_points} vértices e {mesh.n_cells} faces.")

    # 4. Adiciona dados às células (faces)
    global face_ids # Armazena globalmente para fácil acesso
    mesh.cell_data['FaceID'] = np.arange(mesh.n_cells)
    mesh.cell_data['Highlight'] = np.full(mesh.n_cells, HIGHLIGHT_DEFAULT, dtype=int)
    mesh.set_active_scalars('Highlight') # Define 'Highlight' para coloração

    face_ids = mesh.cell_data['FaceID'] # Guarda os IDs

    return mesh

# =============================================================================
# Função: find_cell_neighbors
# =============================================================================
def find_cell_neighbors(mesh: pv.PolyData, cellid: int) -> List[int]:
    """Encontra os IDs das células vizinhas de uma célula específica.

    Vizinhas são células que compartilham pelo menos uma aresta.
    Utiliza o método otimizado `mesh.cell_neighbors()`.

    Args:
        mesh (pv.PolyData): A malha onde procurar os vizinhos.
        cellid (int): O índice (ID) da célula para a qual encontrar vizinhos.

    Returns:
        List[int]: Lista com os IDs das células vizinhas. Retorna lista vazia
                   se o cellid for inválido, não houver vizinhos ou ocorrer erro.
    """
    if not (0 <= cellid < mesh.n_cells):
        logger.debug(f"Tentativa de encontrar vizinhos para cellid inválido: {cellid}")
        return []
    try:
        # mesh.cell_neighbors pode retornar -1 para faces de borda, mas não em esfera fechada.
        neighbors = mesh.cell_neighbors(cellid)
        # Filtra por segurança, garantindo que apenas IDs válidos sejam retornados.
        valid_neighbors = [n for n in neighbors if n >= 0]
        # logger.debug(f"Vizinhos da célula {cellid}: {valid_neighbors}") # Log detalhado opcional
        return valid_neighbors
    except Exception as e:
        logger.error(f"Erro ao encontrar vizinhos para célula {cellid}: {e}", exc_info=True)
        return []

# =============================================================================
# Função: update_labels
# =============================================================================
def update_labels():
    """Atualiza os rótulos de texto exibidos na visualização.

    Remove os rótulos antigos e adiciona novos com base no estado de destaque
    atual e na flag `show_all_labels`. Acessa variáveis globais
    `plotter`, `mesh_esfera`, `face_centers`, `label_actor`, `show_all_labels`.
    """
    global label_actor, show_all_labels, plotter, mesh_esfera, face_centers, face_ids

    if not plotter or not mesh_esfera or face_centers is None or face_ids is None:
        logger.warning("Update labels ignorado: plotter/mesh/centers/face_ids não prontos.")
        return

    # 1. Remove ator de rótulos antigo, se existir
    if label_actor is not None:
        try:
            plotter.remove_actor(label_actor, render=False)
        except ValueError:
            # Actor já pode ter sido removido ou nunca adicionado
            logger.debug("Ator de rótulo não encontrado para remoção (pode já ter sido removido).")
        finally:
             label_actor = None # Garante que a referência seja limpa

    # 2. Prepara dados para novos rótulos
    labels_to_show = []
    points_to_label = []
    highlights = mesh_esfera.cell_data['Highlight'] # Pega o array de destaques atual

    # 3. Decide quais rótulos mostrar
    for i in range(mesh_esfera.n_cells):
        highlight_status = highlights[i]
        # Mostra se show_all_labels=True ou se a célula está destacada
        if show_all_labels or highlight_status != HIGHLIGHT_DEFAULT:
            label_text = f"ID:{face_ids[i]}\nH:{highlight_status}" # Usa face_ids global
            labels_to_show.append(label_text)
            points_to_label.append(face_centers[i])

    # 4. Adiciona novos rótulos, se houver
    if labels_to_show:
        logger.debug(f"Adicionando {len(labels_to_show)} rótulos.")
        label_actor = plotter.add_point_labels(
            np.array(points_to_label),
            labels_to_show,
            font_size=12, # Ajuste o tamanho conforme necessário
            point_size=5, # Ponto invisível de ancoragem
            shape=None,
            show_points=False,
            pickable=False, # Rótulos não são clicáveis
            render=False    # Renderização será feita após outras atualizações
        )
    # else: # Não precisa remover explicitamente aqui, já foi feito no início
    #    logger.debug("Nenhum rótulo para exibir.")


# =============================================================================
# Função: update_highlight
# =============================================================================
def update_highlight(cellid: int):
    """Atualiza o estado de destaque (hover e vizinhos) de forma eficiente.

    Limpa o destaque anterior e aplica o novo baseado no `cellid` sob o cursor.
    Chamada pelo `vtk_mouse_move_callback`. Acessa variáveis globais.

    Args:
        cellid (int): O índice (ID) da célula sob o cursor (-1 se nenhuma).
    """
    global current_hovered_info, mesh_esfera, plotter

    # Verificação de segurança essencial
    if mesh_esfera is None or plotter is None:
        # Não usar face_centers aqui, pois update_labels será chamado
        logger.warning("Update highlight ignorado: mesh/plotter não prontos.")
        return

    needs_render = False # Flag para controlar se a renderização é necessária

    # --- Caso 1: Cursor sobre uma célula válida ---
    if 0 <= cellid < mesh_esfera.n_cells:
        # Otimização: Sai se o mouse ainda está sobre a mesma célula
        if cellid == current_hovered_info["cellid"]:
            return

        logger.debug(f"Hover entrou na célula {cellid}")
        needs_render = True # Haverá mudança visual
        highlights = mesh_esfera.cell_data['Highlight'] # Acesso direto ao array numpy

        # --- Limpa Destaque Anterior ---
        prev_cellid = current_hovered_info["cellid"]
        prev_neighbors = current_hovered_info["neighbors"]

        cells_to_reset = set()
        if 0 <= prev_cellid < mesh_esfera.n_cells:
             cells_to_reset.add(prev_cellid)
        cells_to_reset.update(n for n in prev_neighbors if 0 <= n < mesh_esfera.n_cells)

        # Reseta todos os anteriores que NÃO serão destacados novamente agora
        new_hover_and_neighbors = {cellid} | set(find_cell_neighbors(mesh_esfera, cellid))
        for reset_id in cells_to_reset:
            if reset_id not in new_hover_and_neighbors:
                 highlights[reset_id] = HIGHLIGHT_DEFAULT


        # --- Aplica Novo Destaque ---
        new_neighbors = find_cell_neighbors(mesh_esfera, cellid)
        highlights[cellid] = HIGHLIGHT_HOVER # Marca célula atual
        for neighbor_id in new_neighbors:
            if 0 <= neighbor_id < mesh_esfera.n_cells:
                 # Só marca como vizinho se não for a própria célula sob hover
                 if highlights[neighbor_id] != HIGHLIGHT_HOVER:
                      highlights[neighbor_id] = HIGHLIGHT_NEIGHBOR

        # --- Atualiza Estado Global ---
        current_hovered_info["cellid"] = cellid
        current_hovered_info["neighbors"] = new_neighbors

    # --- Caso 2: Cursor fora da malha ---
    else:
        prev_cellid = current_hovered_info["cellid"]
        # Só precisa limpar se antes havia algo destacado
        if prev_cellid != -1:
            logger.debug(f"Hover saiu da malha (anterior: {prev_cellid})")
            needs_render = True # Haverá mudança visual
            highlights = mesh_esfera.cell_data['Highlight']

            # Limpa célula anterior e seus vizinhos
            cells_to_reset = {prev_cellid} | set(current_hovered_info["neighbors"])
            for reset_id in cells_to_reset:
                 if 0 <= reset_id < mesh_esfera.n_cells:
                     highlights[reset_id] = HIGHLIGHT_DEFAULT

            # Reseta estado global
            current_hovered_info = {"cellid": -1, "neighbors": []}

    # --- Atualiza Visualização (se necessário) ---
    if needs_render:
        #logger.debug("Atualizando escalares e rótulos...")
        mesh_esfera.set_active_scalars('Highlight') # Garante que os dados atualizados sejam usados
        update_labels() # Atualiza os rótulos baseado no novo estado 'Highlight'
        plotter.render() # Redesenha a cena

# =============================================================================
# Função: vtk_mouse_move_callback
# =============================================================================
def vtk_mouse_move_callback(vtk_interactor: vtk.vtkRenderWindowInteractor, event_name: str):
    """Callback do VTK para o evento de movimento do mouse.

    Usa `vtkCellPicker` para encontrar a célula sob o cursor e chama
    `update_highlight` para atualizar o estado visual.

    Args:
        vtk_interactor: O interator da janela VTK que disparou o evento.
        event_name (str): O nome do evento (ex: "MouseMoveEvent").
    """
    global hover_picker, plotter

    # Verificação de segurança crucial
    if hover_picker is None or plotter is None or plotter.renderer is None:
        # logger.warning("Callback de mouse move ignorado: picker/plotter/renderer não prontos.")
        return

    # 1. Obtém posição do mouse na janela
    x, y = vtk_interactor.GetEventPosition()

    # 2. Usa o picker para encontrar a célula na posição (x, y)
    # O picker considera apenas o ator 'mesh_esfera' devido à configuração em main()
    hover_picker.Pick(x, y, 0, plotter.renderer)

    # 3. Obtém o ID da célula atingida (-1 se nenhuma)
    cellid = hover_picker.GetCellId()

    # 4. Delega a lógica de atualização para update_highlight
    # Não é necessário verificar se a célula é válida aqui, update_highlight faz isso.
    update_highlight(cellid)

# =============================================================================
# Função: click_callback
# =============================================================================
def click_callback(picked_mesh: Optional[pv.PolyData], cellid: int):
    """Callback do PyVista para o evento de clique em uma célula.

    Registra o FaceID da célula clicada na lista `selected_cells`.

    Args:
        picked_mesh (Optional[pv.PolyData]): A malha que foi clicada.
                                             Pode ser None se o clique for fora.
        cellid (int): O índice (ID) da célula clicada.
    """
    global selected_cells, face_ids # Acessa globais

    # Verifica se o clique foi válido em uma célula da nossa malha
    if picked_mesh is None or not (0 <= cellid < picked_mesh.n_cells) or face_ids is None:
        logger.info("Clique fora de uma célula válida da esfera.")
        return

    try:
        # Usa o array global face_ids (assumindo que corresponde à picked_mesh)
        face_id = face_ids[cellid]
        logger.info(f"Célula Clicada (Índice VTK): {cellid}, ID da Face: {face_id}")

        # Adiciona à lista de selecionados se ainda não estiver lá
        if face_id not in selected_cells:
            selected_cells.append(face_id)
            logger.info(f"Face {face_id} adicionada à seleção.")
            # --- Ponto de Extensão para Feedback Visual ---
            # Ex: Mudar permanentemente a cor da célula clicada
            # picked_mesh.cell_data['Highlight'][cellid] = ALGUMA_COR_SELECAO
            # mesh_esfera.set_active_scalars('Highlight') # Se mudar dados
            # plotter.render()
        else:
             logger.info(f"Face {face_id} já estava selecionada.")
             # --- Ponto de Extensão para Desseleção ---
             # selected_cells.remove(face_id)
             # logger.info(f"Face {face_id} removida da seleção.")
             # picked_mesh.cell_data['Highlight'][cellid] = HIGHLIGHT_DEFAULT # Resetar cor
             # plotter.render()


        logger.info(f"IDs das Faces Selecionadas: {sorted(selected_cells)}") # Mostra ordenado

    except IndexError:
        logger.error(f"Erro de índice ao tentar obter FaceID para célula {cellid}.", exc_info=True)
    except Exception as e:
         logger.error(f"Erro inesperado no callback de clique para célula {cellid}: {e}", exc_info=True)


# =============================================================================
# Função: toggle_labels_callback
# =============================================================================
def toggle_labels_callback():
    """Callback para evento de tecla ('L'). Alterna a visibilidade dos rótulos."""
    global show_all_labels, plotter, mesh_esfera, face_centers # Acessa globais

    if plotter and mesh_esfera and face_centers is not None:
        show_all_labels = not show_all_labels # Inverte o booleano
        logger.info(f"Visibilidade de todos os rótulos alterada para: {show_all_labels}")
        update_labels() # Atualiza os rótulos com base na nova flag
        plotter.render() # Redesenha
    else:
        logger.warning("Toggle labels ignorado: estado da aplicação incompleto.")


# =============================================================================
# Função: main
# =============================================================================
def main():
    """Função principal: configura a cena, interações e inicia a visualização."""
    # Torna globais acessíveis para modificação/setup inicial
    global selected_cells, hover_picker, mesh_esfera, plotter, face_centers

    # --- 1. Configuração da Esfera ---
    poly_base = 'icosahedron'
    num_subdivisions = 3 # Aumentei para teste, ajuste conforme necessário
    esfera_radius = 5.0

    try:
        mesh_esfera = create_triangulated_sphere(
            poly_type=poly_base,
            subdivisions=num_subdivisions,
            radius=esfera_radius
        )
        # Centros das faces são calculados e armazenados dentro de create_triangulated_sphere
        face_centers = mesh_esfera.cell_centers().points
    except Exception as e:
        logger.critical(f"Falha ao criar a malha da esfera: {e}", exc_info=True)
        return # Aborta se a malha não puder ser criada

    # --- 2. Configuração do Plotter ---
    pv.set_plot_theme("document") # Ou "dark", "paraview", etc.
    plotter = pv.Plotter(window_size=WINDOW_SIZE, title="Visualizador de Esfera Interativa")

    # Adiciona a malha à cena
    try:
        actor = plotter.add_mesh(
            mesh_esfera,
            scalars='Highlight', # Colorir por dados de destaque
            cmap=discrete_cmap,  # Mapa de cores definido no início
            clim=[HIGHLIGHT_DEFAULT, HIGHLIGHT_NEIGHBOR], # Range das cores (0, 1, 2)
            show_edges=True,
            edge_color='dimgray', # Cor mais escura para arestas
            line_width=0.5,
            lighting='light kit', # Esquema de iluminação padrão
            preference='cell',    # Escalares são por célula
            pickable=True,        # Malha pode ser selecionada por clique/hover
            show_scalar_bar=False # Não mostrar barra de cores
        )
    except Exception as e:
         logger.critical(f"Falha ao adicionar a malha ao plotter: {e}", exc_info=True)
         return

    # --- 3. Configuração do Picker VTK para Hover ---
    # Usamos vtkCellPicker para ter controle fino sobre a detecção contínua (hover)
    hover_picker = vtk.vtkCellPicker()
    hover_picker.SetTolerance(PICKER_TOLERANCE) # Sensibilidade
    # Diz ao picker para considerar APENAS o ator da nossa esfera. Melhora performance.
    hover_picker.AddPickList(actor)
    hover_picker.PickFromListOn()

    # --- 4. Habilita Interações ---

    # a) Hover (Movimento do Mouse) - Observador VTK
    # plotter.iren é o vtkRenderWindowInteractor subjacente
    if plotter.iren:
        # Conecta o evento de movimento do mouse ao nosso callback VTK
        plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, vtk_mouse_move_callback)
        logger.info("Destaque por hover habilitado (Observador VTK MouseMoveEvent).")
    else:
        # Situação incomum, mas impede erro fatal.
        logger.error("vtkRenderWindowInteractor (plotter.iren) não encontrado. Hover não funcionará.")

    # b) Clique na Célula - Abstração PyVista
    # Usa a função de conveniência do PyVista para cliques discretos
    plotter.enable_cell_picking(
        callback=click_callback, # Nossa função será chamada no clique
        show=False,              # Não usar o destaque visual padrão do PyVista
        show_message=False,      # Não mostrar mensagem padrão no console
        use_picker=False         # Importante: Usa o picker interno do PyVista para clique, não o nosso 'hover_picker'
    )
    logger.info("Seleção por clique habilitada (PyVista enable_cell_picking).")

    # c) Teclado (Alternar Rótulos) - Abstração PyVista
    plotter.add_key_event("l", toggle_labels_callback) # 'l' minúsculo
    logger.info("Pressione 'L' para alternar a visibilidade dos rótulos.")

    # --- 5. Inicializa os Rótulos ---
    logger.info("Inicializando rótulos...")
    update_labels() # Exibe o estado inicial dos rótulos

    # --- 6. Exibe Instruções e Inicia a Janela ---
    logger.info("--- Controles ---")
    logger.info(" * Hover Mouse: Destaca face (Amarelo) e vizinhas (Laranja)")
    logger.info(" * Clique Mouse: Seleciona face (registra ID no log - Vermelho por padrão)")
    logger.info(" * Tecla 'L': Alterna visibilidade dos rótulos (Todos vs. Destaque)")
    logger.info(" * Botão Esquerdo + Arrastar: Rotacionar Câmera")
    logger.info(" * Botão Direito + Arrastar / Roda Mouse: Zoom Câmera")
    logger.info(" * Botão Meio + Arrastar: Mover Câmera (Pan)")
    logger.info("-----------------")

    # Adiciona uma fonte de luz extra se desejar
    # light = pv.Light(position=(0, 10, 10), light_type='scene light')
    # plotter.add_light(light)

    # Define a posição inicial da câmera (opcional)
    # plotter.camera_position = 'xy' # Vista de cima
    # plotter.camera.azimuth = 30
    # plotter.camera.elevation = 30

    # Abre a janela interativa e bloqueia até ser fechada
    plotter.show()

    # Código executado após fechar a janela
    logger.info("Janela fechada.")
    logger.info(f"IDs das Faces Selecionadas finais: {sorted(selected_cells)}")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()