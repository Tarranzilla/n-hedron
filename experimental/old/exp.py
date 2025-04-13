# -*- coding: utf-8 -*- # Boa prática para caracteres acentuados
import pyvista as pv
import numpy as np
import vtk # Necessário para o picker e eventos VTK (como movimento do mouse)
import logging
from matplotlib.colors import ListedColormap
from typing import List, Dict, Optional, Any

# --- Configurações Iniciais ---

# Define um mapa de cores discreto para os valores de Destaque (Highlight)
# 0: Cor Padrão (red), 1: Cor Hover (green), 2: Cor Vizinho (yellow), 3: Cor Vizinho 2 (orange)
discrete_cmap = ListedColormap(['red', 'orange', 'yellow', 'green', 'cyan'])

# Configura o sistema de logging para exibir informações durante a execução
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes ---
WINDOW_SIZE = [1600, 900] # Tamanho da janela de visualização em pixels
HIGHLIGHT_DEFAULT = 0    # Valor numérico para indicar o estado normal/padrão de uma célula
HIGHLIGHT_EXTREME_NEIGHBOR = 1 # Valor numérico para indicar que uma célula é vizinha de uma vizinha de uma vizinha
HIGHLIGHT_EXTENDED_NEIGHBOR = 2 # Valor numérico para indicar que uma célula é vizinha de uma vizinha
HIGHLIGHT_NEIGHBOR = 3   # Valor numérico para indicar que uma célula é vizinha da célula sob hover
HIGHLIGHT_HOVER = 4      # Valor numérico para indicar que uma célula está sob o mouse (hover)
PICKER_TOLERANCE = 0.005 # Tolerância para o vtkCellPicker ao detectar a célula sob o mouse

# --- Variáveis de Estado Globais ---
# Usadas para compartilhar informações entre callbacks (funções chamadas por eventos)
label_actor: Optional[pv.Actor] = None # Armazena o ator (objeto gráfico) dos rótulos para poder removê-lo/atualizá-lo
current_hovered_info: Dict[str, Any] = {
    "cellid": -1,
    "neighbors": [],
    "extended_neighbors": [],
    "extreme_neighbors": []
} # Guarda o ID da célula atualmente sob hover e seus vizinhos
show_all_labels: bool = True # Controla se todos os rótulos devem ser exibidos ou apenas os destacados
selected_cells: List[int] = [] # Lista para armazenar os FaceIDs das células que foram clicadas
hover_picker: Optional[vtk.vtkCellPicker] = None # Objeto do VTK para detectar qual célula está sob o cursor do mouse
mesh_esfera: Optional[pv.PolyData] = None        # Armazena a malha da esfera para acesso global nos callbacks
plotter: Optional[pv.Plotter] = None             # Armazena o objeto Plotter do PyVista para acesso global nos callbacks
face_centers: Optional[np.ndarray] = None        # Armazena as coordenadas do centro de cada face para posicionar os rótulos
connections_mode = "edges"                       # Default mode is "edges"
# =============================================================================
# Função: create_triangulated_sphere
# Objetivo: Gerar a malha 3D da esfera triangularizada.
# =============================================================================
def create_triangulated_sphere(
    poly_type: str = 'icosahedron', subdivisions: int = 2, radius: float = 1.0
) -> pv.PolyData:
    """
    Cria uma malha pseudo-esférica triangularizada a partir de um poliedro base.

    Args:
        poly_type (str): Tipo do poliedro base ('tetrahedron', 'octahedron', 'icosahedron').
        subdivisions (int): Número de subdivisões para refinar a malha (mais subdivisões = esfera mais suave).
        radius (float): Raio final da esfera gerada.

    Returns:
        pv.PolyData: O objeto de malha (mesh) da esfera criada.
    """
    logger.info(f"Criando poliedro base: {poly_type}")
    poly_type_lower = poly_type.lower()

    # 1. Cria o poliedro base escolhido usando PyVista ou definição manual (icosaedro)
    if poly_type_lower == 'tetrahedron':
        mesh = pv.Tetrahedron()
    elif poly_type_lower == 'octahedron':
        mesh = pv.Octahedron()
    elif poly_type_lower == 'icosahedron':
        # Definição manual dos vértices e faces de um icosaedro normalizado
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
        # Formata as faces para o padrão do PyVista: [num_pontos, p1, p2, p3, ...]
        faces_pv = np.hstack((np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces)).flatten()
        mesh = pv.PolyData(vertices, faces=faces_pv)
    else:
        raise ValueError("poly_type deve ser 'tetrahedron', 'octahedron', ou 'icosahedron'")

    # 2. Subdivide a malha base para aumentar a resolução (mais triângulos)
    logger.info(f"Subdividindo {subdivisions} vezes usando o esquema 'loop'...")
    if subdivisions > 0:
        # O método 'loop' é adequado para malhas triangulares, suavizando a superfície
        mesh = mesh.subdivide(subdivisions, subfilter='loop')

    # 3. Normaliza os vértices para formar uma esfera perfeita e aplica o raio desejado
    logger.info("Normalizando vértices para forma esférica e aplicando raio...")
    # Calcula vetores do centro da malha para cada vértice
    vertex_vectors = mesh.points - mesh.center
    # Calcula a distância de cada vértice ao centro (norma do vetor)
    norms = np.linalg.norm(vertex_vectors, axis=1)
    # Evita divisão por zero caso algum vértice esteja exatamente no centro
    norms[norms == 0] = 1.0
    # Ajusta a posição de cada vértice: (vetor / norma) dá um vetor unitário (na esfera de raio 1)
    # Multiplica pelo raio desejado e adiciona de volta ao centro da malha
    mesh.points = mesh.center + (vertex_vectors / norms[:, np.newaxis]) * radius

    logger.info(f"Malha final criada com {mesh.n_points} vértices e {mesh.n_cells} faces.")

    # 4. Adiciona dados às células (faces) da malha
    # 'FaceID': Um identificador único para cada face (0, 1, 2, ...)
    mesh.cell_data['FaceID'] = np.arange(mesh.n_cells)
    # 'Highlight': Inicializa o estado de destaque de todas as faces como padrão (0)
    mesh.cell_data['Highlight'] = np.full(mesh.n_cells, HIGHLIGHT_DEFAULT, dtype=int)
    # Define 'Highlight' como o escalar ativo, usado para colorir a malha
    mesh.set_active_scalars('Highlight')

    return mesh # Retorna a malha da esfera pronta

# =============================================================================
# Função: toggle_connections_mode
# Objetivo: Controlar o estado e modo de conexão (pontos ou arestas) para a malha.
# =============================================================================

def toggle_connections_mode():
    """
    Toggle between 'points' and 'edges' for the connections mode.
    """
    global connections_mode
    connections_mode = "points" if connections_mode == "edges" else "edges"
    logger.info(f"Modo de conexão alternado para: {connections_mode}")

# =============================================================================
# Função: find_cell_neighbors
# Objetivo: Encontrar as células (faces) vizinhas a uma célula específica.
# =============================================================================
def find_cell_neighbors(mesh: pv.PolyData, cellid: int, optional_mode: Optional[str] = None) -> List[int]:
    """
    Encontra as células vizinhas de uma dada célula usando o método eficiente do PyVista.
    Vizinhas são células que compartilham pelo menos uma aresta ou ponto, dependendo do modo.

    Args:
        mesh (pv.PolyData): A malha onde procurar os vizinhos.
        cellid (int): O índice (ID) da célula para a qual encontrar vizinhos.
        optional_mode (Optional[str]): Modo de conexão ('points' ou 'edges') para sobrescrever o global.

    Returns:
        List[int]: Uma lista com os índices (IDs) das células vizinhas.
                   Retorna lista vazia se o cellid for inválido ou não tiver vizinhos.
    """
    # Determina o modo de conexão a ser usado
    mode = optional_mode if optional_mode in ["points", "edges"] else connections_mode

    # Verifica se o ID da célula é válido
    if 0 <= cellid < mesh.n_cells:
        try:
            # Usa a função otimizada do PyVista para obter os vizinhos
            neighbors = mesh.cell_neighbors(cellid, connections=mode)
            # Filtra IDs válidos (>= 0)
            return [n for n in neighbors if n >= 0]
        except Exception as e:
            logger.error(f"Erro ao encontrar vizinhos para célula {cellid}: {e}")
            return []
    return []  # Retorna lista vazia se o cellid inicial for inválido

# =============================================================================
# Função: update_labels
# Objetivo: Atualizar os rótulos de texto exibidos sobre as faces da esfera.
# =============================================================================
def update_labels(plotter_ref: pv.Plotter, mesh_ref: pv.PolyData, centers_ref: np.ndarray):
    """
    Atualiza os rótulos (texto) exibidos na visualização, baseado no estado
    atual de destaque e na flag `show_all_labels`.

    Args:
        plotter_ref (pv.Plotter): Referência ao objeto plotter principal.
        mesh_ref (pv.PolyData): Referência à malha da esfera.
        centers_ref (np.ndarray): Coordenadas do centro de cada face.
    """
    global label_actor, show_all_labels # Acessa/Modifica variáveis globais

    # Verificação de segurança: não faz nada se os objetos não estiverem prontos
    if not plotter_ref or not mesh_ref or centers_ref is None: return

    # 1. Remove os rótulos antigos (se existirem)
    if label_actor is not None:
        try:
             # Tenta remover o ator dos rótulos da cena
             plotter_ref.remove_actor(label_actor, render=False) # render=False para não redesenhar a cena ainda
        except ValueError:
             # Ignora o erro se o ator já tiver sido removido por algum motivo
             pass
        label_actor = None # Garante que a referência ao ator antigo seja limpa

    # 2. Prepara listas para os novos rótulos e suas posições
    labels_to_show = []
    points_to_label = []

    # Acessa os dados de destaque e IDs das faces
    highlights = mesh_ref.cell_data['Highlight']
    face_ids = mesh_ref.cell_data['FaceID']

    # 3. Itera sobre todas as faces para decidir quais rótulos mostrar
    for i in range(mesh_ref.n_cells):
        highlight_status = highlights[i]
        # Condição para mostrar o rótulo:
        # - Se show_all_labels for True, OU
        # - Se a célula estiver destacada (hover ou vizinho)
        if show_all_labels or highlight_status != HIGHLIGHT_DEFAULT:
            # Formata o texto do rótulo (ID da Face e Status de Destaque)
            label_text = f"ID:{face_ids[i]}\nH:{highlight_status}"
            labels_to_show.append(label_text)
            # Adiciona a posição central da face correspondente
            points_to_label.append(centers_ref[i])

    # 4. Adiciona os novos rótulos à cena (se houver algum para mostrar)
    if labels_to_show:
        # Cria um novo ator de rótulos
        label_actor = plotter_ref.add_point_labels(
            np.array(points_to_label), # Posições onde os rótulos aparecerão
            labels_to_show,           # Textos dos rótulos
            font_size=14,             # Tamanho da fonte
            point_size=5,             # Tamanho do "ponto invisível" que ancora o rótulo
            shape=None,               # Nenhuma forma geométrica para o ponto de ancoragem
            show_points=False,        # Não mostrar o ponto de ancoragem
            pickable=False,           # Rótulos não precisam ser clicáveis
            render=False              # Adia a renderização final
        )

# =============================================================================
# Função: update_highlight
# Objetivo: Lógica principal para atualizar o estado visual de destaque das faces.
# =============================================================================
def update_highlight(cellid: int):
    global current_hovered_info, mesh_esfera, plotter, face_centers

    if mesh_esfera is None or plotter is None or face_centers is None:
        logger.warning("Atualização de destaque ignorada, malha/plotter não prontos.")
        return

    if 0 <= cellid < mesh_esfera.n_cells:
        if cellid == current_hovered_info["cellid"]:
            return

        # Reset previous highlights
        prev_cellid = current_hovered_info["cellid"]
        if 0 <= prev_cellid < mesh_esfera.n_cells:
            mesh_esfera.cell_data['Highlight'][prev_cellid] = HIGHLIGHT_DEFAULT
            for neighbor_id in current_hovered_info["neighbors"]:
                mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_DEFAULT
            for extended_neighbor_id in current_hovered_info["extended_neighbors"]:
                mesh_esfera.cell_data['Highlight'][extended_neighbor_id] = HIGHLIGHT_DEFAULT
            for extreme_neighbor_id in current_hovered_info["extreme_neighbors"]:
                mesh_esfera.cell_data['Highlight'][extreme_neighbor_id] = HIGHLIGHT_DEFAULT

        # Update highlights
        mesh_esfera.cell_data['Highlight'][cellid] = HIGHLIGHT_HOVER
        neighbors = find_cell_neighbors(mesh_esfera, cellid)
        neighbors = [n for n in neighbors if n != cellid]
        for neighbor_id in neighbors:
            mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_NEIGHBOR

        extended_neighbors = set()
        for neighbor_id in neighbors:
            extended = find_cell_neighbors(mesh_esfera, neighbor_id)
            for ext_id in extended:
                if ext_id != cellid and ext_id not in neighbors:
                    extended_neighbors.add(ext_id)
        for extended_neighbor_id in extended_neighbors:
            mesh_esfera.cell_data['Highlight'][extended_neighbor_id] = HIGHLIGHT_EXTENDED_NEIGHBOR

        extreme_neighbors = set()
        for extended_neighbor_id in extended_neighbors:
            extreme = find_cell_neighbors(mesh_esfera, extended_neighbor_id)
            for extreme_id in extreme:
                if (
                    extreme_id != cellid
                    and extreme_id not in neighbors
                    and extreme_id not in extended_neighbors
                ):
                    extreme_neighbors.add(extreme_id)
        for extreme_neighbor_id in extreme_neighbors:
            mesh_esfera.cell_data['Highlight'][extreme_neighbor_id] = HIGHLIGHT_EXTREME_NEIGHBOR

        # Update global state
        current_hovered_info["cellid"] = cellid
        current_hovered_info["neighbors"] = neighbors
        current_hovered_info["extended_neighbors"] = list(extended_neighbors)
        current_hovered_info["extreme_neighbors"] = list(extreme_neighbors)

        # Update visualization
        mesh_esfera.set_active_scalars('Highlight')
        update_labels(plotter, mesh_esfera, face_centers)
        plotter.render()

    else:
        # Reset highlights if no cell is hovered
        prev_cellid = current_hovered_info["cellid"]
        if 0 <= prev_cellid < mesh_esfera.n_cells:
            mesh_esfera.cell_data['Highlight'][prev_cellid] = HIGHLIGHT_DEFAULT
            for neighbor_id in current_hovered_info["neighbors"]:
                mesh_esfera.cell_data['Highlight'][neighbor_id] = HIGHLIGHT_DEFAULT
            for extended_neighbor_id in current_hovered_info["extended_neighbors"]:
                mesh_esfera.cell_data['Highlight'][extended_neighbor_id] = HIGHLIGHT_DEFAULT
            for extreme_neighbor_id in current_hovered_info["extreme_neighbors"]:
                mesh_esfera.cell_data['Highlight'][extreme_neighbor_id] = HIGHLIGHT_DEFAULT

        current_hovered_info = {
            "cellid": -1,
            "neighbors": [],
            "extended_neighbors": [],
            "extreme_neighbors": []
        }
        mesh_esfera.set_active_scalars('Highlight')
        update_labels(plotter, mesh_esfera, face_centers)
        plotter.render()

# =============================================================================
# Função: vtk_mouse_move_callback
# Objetivo: Ser chamada pelo sistema VTK sempre que o mouse se mover na janela.
# =============================================================================
def vtk_mouse_move_callback(vtk_interactor, event_name):
    """
    Função de callback chamada pelo VTK quando o mouse se move.
    Usa o vtkCellPicker para identificar a célula sob o cursor e chama update_highlight.

    Args:
        vtk_interactor: O objeto interator da janela VTK.
        event_name: O nome do evento (geralmente "MouseMoveEvent").
    """
    global hover_picker, plotter # Acessa os objetos globais necessários

    # Verificação de segurança: não faz nada se o picker ou plotter não estiverem prontos
    if hover_picker is None or plotter is None or plotter.renderer is None:
        return

    # 1. Obtém as coordenadas (x, y) da posição atual do mouse na janela
    x, y = vtk_interactor.GetEventPosition()

    # 2. Usa o vtkCellPicker para "lançar um raio" da câmera através do pixel (x,y)
    #    e descobrir qual célula (se alguma) foi atingida.
    hover_picker.Pick(x, y, 0, plotter.renderer) # O 0 significa z=0 (profundidade da tela)

    # 3. Obtém o ID da célula atingida (-1 se nenhuma foi atingida)
    cellid = hover_picker.GetCellId()

    # 4. Chama a função principal de atualização do destaque, passando o ID encontrado
    update_highlight(cellid)

# =============================================================================
# Função: click_callback
# Objetivo: Ser chamada pelo PyVista quando uma célula é clicada.
# =============================================================================
def click_callback(picked_mesh: pv.PolyData, cellid: int):
    """
    Função de callback chamada quando uma célula da malha é clicada.
    Registra o ID da face clicada.

    Args:
        picked_mesh (pv.PolyData): A malha que foi clicada (deve ser a nossa esfera).
        cellid (int): O índice (ID) da célula que foi clicada.
    """
    global selected_cells # Acessa/Modifica a lista global de células selecionadas

    # Verifica se o clique foi válido (em uma célula existente da malha)
    if picked_mesh is None or not (0 <= cellid < picked_mesh.n_cells):
        logger.info("Nenhuma célula válida foi clicada.")
        return

    try:
        # Obtém o 'FaceID' (nosso identificador único) associado à célula clicada
        face_id = picked_mesh.cell_data['FaceID'][cellid]
        logger.info(f"Célula Clicada (Índice): {cellid}, ID da Face: {face_id}")

        # Adiciona o FaceID à lista de selecionados (se ainda não estiver lá)
        if face_id not in selected_cells:
            selected_cells.append(face_id)
        logger.info(f"IDs das Faces Selecionadas: {selected_cells}")

        # --- Ponto de Extensão ---
        # Aqui você poderia adicionar mais lógica, como:
        # - Mudar a cor da célula clicada permanentemente.
        # - Exibir informações adicionais sobre a face.
        # - Iniciar alguma ação de jogo relacionada àquela face.

    except (KeyError, IndexError) as e:
        # Trata erros caso 'FaceID' não exista ou o índice seja inválido
        logger.error(f"Erro ao acessar FaceID para célula {cellid}: {e}")

# =============================================================================
# Função: toggle_labels_callback
# Objetivo: Ser chamada quando a tecla 'L' for pressionada.
# =============================================================================
def toggle_labels_callback():
    """
    Alterna o modo de exibição dos rótulos (todos ou apenas destacados)
    quando a tecla 'L' é pressionada.
    """
    global show_all_labels, plotter, mesh_esfera, face_centers # Acessa globais

    # Verifica se os objetos necessários estão prontos
    if plotter and mesh_esfera and face_centers is not None:
        # Inverte o valor booleano (True vira False, False vira True)
        show_all_labels = not show_all_labels
        logger.info(f"Alternou 'Mostrar Todos os Rótulos' para: {show_all_labels}")
        # Chama a função para atualizar os rótulos com base no novo valor da flag
        update_labels(plotter, mesh_esfera, face_centers)
        # Redesenha a cena para mostrar a mudança
        plotter.render()

# =============================================================================
# Função: main
# Objetivo: Função principal que orquestra a criação e exibição da cena.
# =============================================================================
def main():
    """Função principal que configura e executa a visualização interativa."""
    # Disponibiliza variáveis globais para serem modificadas dentro desta função
    # e para serem acessadas pelos callbacks definidos fora dela.
    global selected_cells, hover_picker, mesh_esfera, plotter, face_centers

    # --- 1. Configuração da Esfera ---
    poly_base = 'icosahedron'       # Poliedro base para iniciar a esfera
    num_subdivisions = 3          # Nível de detalhe da esfera (mais subdivisões = mais suave)
    esfera_radius = 5.0             # Raio da esfera final

    # Cria a malha da esfera usando a função auxiliar
    mesh_esfera = create_triangulated_sphere(
        poly_type=poly_base,
        subdivisions=num_subdivisions,
        radius=esfera_radius
    )
    # Calcula e armazena as posições centrais de todas as faces (para os rótulos)
    face_centers = mesh_esfera.cell_centers().points

    # --- 2. Configuração do Plotter (Janela de Visualização) ---
    pv.set_plot_theme("document") # Define um tema visual para a janela
    plotter = pv.Plotter(window_size=WINDOW_SIZE) # Cria o objeto Plotter

    # Adiciona a malha da esfera à cena do plotter
    actor = plotter.add_mesh( # Guarda a referência ao 'ator' da malha
        mesh_esfera,                     # A malha a ser adicionada
        scalars='Highlight',             # Usa os dados 'Highlight' para colorir as faces
        cmap=discrete_cmap,              # Mapa de cores a ser usado (lightblue, orange, yellow)
        clim=[HIGHLIGHT_DEFAULT, HIGHLIGHT_HOVER], # Define o range do cmap (0 a total de cores)
        show_edges=True,                 # Mostra as arestas dos triângulos
        edge_color='gray',               # Cor das arestas
        line_width=0.5,                  # Espessura das arestas
        lighting='light kit',            # Esquema de iluminação da cena
        preference='cell',               # Indica que os escalares ('Highlight') são por célula (face)
        pickable=True,                   # Permite que esta malha seja detectada por 'picking' (hover/clique)
        show_scalar_bar=False            # Não mostra a legenda de cores (scalar bar)
    )

    # --- 3. Configuração do Picker VTK para o Hover ---
    hover_picker = vtk.vtkCellPicker()          # Cria o objeto picker
    hover_picker.SetTolerance(PICKER_TOLERANCE) # Define a tolerância (quão perto o mouse precisa estar)
    hover_picker.AddPickList(actor)             # Diz ao picker para considerar APENAS este ator (a esfera)
    hover_picker.PickFromListOn()               # Ativa o uso da lista definida acima

    # --- 4. Habilita Interações ---

    # a) Interação de Hover (passar o mouse) - Usando o observador VTK
    # plotter.iren é o interator da janela VTK (vtkRenderWindowInteractor)
    if plotter.iren:
        # Adiciona um "ouvinte" para o evento de movimento do mouse.
        # Quando o mouse se mover, a função vtk_mouse_move_callback será chamada.
        plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, vtk_mouse_move_callback)
        logger.info("Destaque por hover habilitado (usando Observador VTK MouseMove).")
    else:
        # Caso raro onde o interator não está disponível
        logger.error("Não foi possível obter vtkRenderWindowInteractor (plotter.iren) para adicionar observador.")

    # b) Interação de Clique - Usando a função do PyVista
    plotter.enable_cell_picking(
        callback=lambda mesh, idx: click_callback(mesh, idx), # Função a ser chamada no clique
        show=False,        # Não mostra a seleção padrão do PyVista (bloco colorido)
        show_message=False # Não mostra a mensagem padrão "Picked cell #..."
    )
    logger.info("Clique em células habilitado (usando PyVista enable_cell_picking).")

    # c) Interação por Teclado - Alternar Rótulos
    # Associa a tecla 'l' (minúsculo) à função toggle_labels_callback
    plotter.add_key_event("l", toggle_labels_callback)
    logger.info("Pressione 'L' para alternar a visibilidade dos rótulos.")

    # d) Alternar Modo de Conexão (pontos ou arestas)
    # Associa a tecla 'x' (minúsculo) à função toggle_connections_mode
    # (que alterna entre 'points' e 'edges')
    plotter.add_key_event("x", toggle_connections_mode)
    logger.info("Pressione 'X' para alternar entre 'points' e 'edges' para vizinhos.")

    # --- 5. Inicializa os Rótulos ---
    # Chama a função uma vez no início para exibir os rótulos conforme a configuração inicial
    update_labels(plotter, mesh_esfera, face_centers)

    # --- 6. Exibe Instruções e a Janela ---
    logger.info("--- Controles ---")
    logger.info(" - Passar Mouse (Hover): Destaca face (laranja) e vizinhas (amarelo)")
    logger.info(" - Clicar Mouse: Seleciona face (registra ID no console)")
    logger.info(" - Tecla 'L': Alterna visibilidade dos rótulos")
    logger.info(" - Botão Esquerdo + Arrastar: Rotacionar")
    logger.info(" - Botão Direito + Arrastar / Roda Mouse: Zoom")
    logger.info(" - Botão Meio + Arrastar: Mover (Pan)")

    # Abre a janela interativa e inicia o loop de eventos VTK
    plotter.show()

    # Código abaixo só executa após a janela ser fechada
    logger.info(f"Saindo. IDs das Faces Selecionadas finais: {selected_cells}")

# --- Ponto de Entrada do Script ---
# Garante que a função main() só seja executada quando o script for rodado diretamente
if __name__ == "__main__":
    main()