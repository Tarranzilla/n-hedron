# -*- coding: utf-8 -*- # Boa prática para garantir a codificação correta de caracteres acentuados.
"""
Script para visualização 3D interativa de uma esfera triangularizada usando PyVista e VTK.

Funcionalidades Principais:
- Geração de uma esfera pseudo-geodésica a partir de um poliedro base (ex: icosaedro).
- Destaque visual da célula (face) atualmente sob o cursor do mouse (hover).
- Destaque visual de múltiplos níveis de vizinhança em relação à célula sob hover:
    - Vizinhos diretos (nível 1).
    - Vizinhos dos vizinhos (nível 2, excluindo os já destacados).
    - Vizinhos de nível 3 (excluindo os já destacados).
- Alternância do critério de vizinhança (compartilhando arestas ou pontos) com a tecla 'X'.
- Exibição opcional de rótulos sobre as faces com seu ID e status de destaque.
- Alternância da visibilidade dos rótulos com a tecla 'L'.
- Seleção de faces por clique, registrando seus IDs.
- Interação padrão de câmera (rotação, zoom, pan).
"""

import pyvista as pv
import numpy as np
import vtk  # Necessário para interações de baixo nível (vtkCellPicker, MouseMoveEvent)
import logging
from matplotlib.colors import ListedColormap
from typing import List, Dict, Optional, Any

# --- Configurações Iniciais ---

# Define um mapa de cores discreto para os diferentes estados de destaque (Highlight)
# Mapeamento dos valores de HIGHLIGHT para as cores:
# 0 (DEFAULT): Vermelho ('red') - Estado padrão da célula.
# 1 (EXTREME): Laranja ('orange') - Vizinho de nível 3.
# 2 (EXTENDED): Amarelo ('yellow') - Vizinho de nível 2 (vizinho de vizinho).
# 3 (NEIGHBOR): Verde ('green') - Vizinho direto (nível 1).
# 4 (HOVER): Azul ('blue') - Célula diretamente sob o cursor do mouse.
discrete_cmap = ListedColormap(['red', 'orange', 'yellow', 'green', 'lightgreen'])

# Configura o sistema de logging para exibir mensagens informativas durante a execução.
# Use level=logging.DEBUG para informações mais detalhadas de depuração.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s', # Adicionado nome da função ao log
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Cria uma instância do logger para este módulo.

# --- Constantes ---
WINDOW_SIZE = [1600, 900]  # Dimensões da janela de visualização em pixels [largura, altura].

# Constantes numéricas para representar os diferentes estados de destaque de uma célula.
# Estes valores são usados no array 'Highlight' da malha e mapeados pelo 'discrete_cmap'.
HIGHLIGHT_DEFAULT = 0     # Estado padrão, sem destaque.
HIGHLIGHT_EXTREME_NEIGHBOR = 1 # Estado para vizinho de nível 3 (vizinho de um vizinho de um vizinho).
HIGHLIGHT_EXTENDED_NEIGHBOR = 2 # Estado para vizinho de nível 2 (vizinho de um vizinho).
HIGHLIGHT_NEIGHBOR = 3    # Estado para vizinho direto (nível 1) da célula sob hover.
HIGHLIGHT_HOVER = 4       # Estado para a célula diretamente sob o cursor do mouse.

PICKER_TOLERANCE = 0.005 # Tolerância de seleção para o vtkCellPicker.
                         # Define quão perto o raio do mouse precisa estar de uma célula
                         # para detectá-la. Valores menores exigem mais precisão.

# --- Variáveis de Estado Globais ---
# Armazenam o estado atual da aplicação, compartilhado entre as funções (callbacks).
# Em aplicações maiores, encapsular este estado em uma classe seria mais robusto.

# Ator (objeto gráfico) que contém os rótulos de texto exibidos na cena.
# Armazenado globalmente para poder ser removido e recriado quando os rótulos mudam.
label_actor: Optional[pv.Actor] = None

# Dicionário que armazena informações sobre a última célula que esteve sob o cursor.
# Inclui o ID da célula e as listas de IDs de seus vizinhos em diferentes níveis.
# 'cellid' é -1 quando o cursor não está sobre nenhuma célula.
current_hovered_info: Dict[str, Any] = {
    "cellid": -1,               # ID da célula atualmente sob hover (-1 se nenhuma).
    "neighbors": [],            # Lista de IDs dos vizinhos diretos (nível 1).
    "extended_neighbors": [],   # Lista de IDs dos vizinhos estendidos (nível 2).
    "extreme_neighbors": []     # Lista de IDs dos vizinhos extremos (nível 3).
}

# Flag booleana que controla se todos os rótulos de células devem ser exibidos
# ou apenas os das células que estão atualmente destacadas (hover ou vizinhas).
# Alternado pela tecla 'L'.
show_all_labels: bool = True

# Lista que armazena os 'FaceID' (identificadores únicos definidos por nós)
# das células que foram clicadas pelo usuário.
selected_cells: List[int] = []

# Instância do vtkCellPicker, usado para detectar qual célula da malha
# está sob o cursor do mouse durante o evento de movimento (hover).
hover_picker: Optional[vtk.vtkCellPicker] = None

# Referência global ao objeto PolyData do PyVista que representa a malha da esfera.
# Permite que os callbacks acessem e modifiquem os dados da malha (ex: 'Highlight').
mesh_esfera: Optional[pv.PolyData] = None

# Referência global ao objeto Plotter principal do PyVista, que gerencia a janela
# e a cena 3D. Necessário para adicionar/remover atores e renderizar atualizações.
plotter: Optional[pv.Plotter] = None

# Array NumPy armazenando as coordenadas (x, y, z) do centro de cada face (célula) da malha.
# Usado para posicionar corretamente os rótulos sobre as faces. Calculado uma vez após criar a esfera.
face_centers: Optional[np.ndarray] = None

# String que define o modo atual de busca por vizinhos:
# 'edges': Células que compartilham pelo menos uma aresta são vizinhas.
# 'points': Células que compartilham pelo menos um ponto (vértice) são vizinhas.
# Alternado pela tecla 'X'.
connections_mode: str = "edges" # Modo inicial padrão é por arestas.

sphere_radius = 0.1  # Raio da esfera base (pode ser ajustado conforme necessário).
cone_radius = 0.25  # Raio do cone (parte superior do peão).
cone_height = 0.5  # Altura do cone (parte superior do peão).
cylinder_radius = 0.25  # Raio do cilindro (parte inferior do peão).
cylinder_height = 0.025  # Altura do cilindro (parte inferior do peão).

floating_sphere_actor: Optional[pv.Actor] = None  # Actor for the floating sphere
floating_sphere = pv.Sphere(radius=sphere_radius, direction=(1,0,0), phi_resolution=32, theta_resolution=32)  # Adjust the radius as needed
floating_cone = pv.Cone(radius=cone_radius, height=cone_height, resolution=32)  # Adjust the radius and height as needed
floating_base_cylinder = pv.Cylinder(radius=cylinder_radius, height=cylinder_height, resolution=32)  # Adjust the radius and height as needed
pawn_height = sphere_radius + cone_height + cylinder_height  # Total height of the pawn (sphere + cone + cylinder)

# =============================================================================
# Função: create_pawn
# =============================================================================

def create_pawn():
    """
    Creates a pawn-shaped object using a sphere, cone, and cylinder.
    Returns the combined PyVista mesh.
    """
    global floating_sphere, floating_cone, floating_base_cylinder

    # Offset the cone and cylinder relative to the sphere
    cone_offset = [0, 0, 0]  # Move the cone below the sphere
    cylinder_offset = [-0.2625, 0, 0]  # Move the cylinder below the cone
    sphere_offset = [0.25, 0, 0]  # Move the sphere above the cone and cylinder
    
    # Translate the sphere
    sphere = floating_sphere.translate(sphere_offset, inplace=False)
    # Rotate the sphere 90 degrees around the Y-axis to align its poles upward

    # Translate the cone and cylinder
    cone = floating_cone.translate(cone_offset, inplace=False)
    cylinder = floating_base_cylinder.translate(cylinder_offset, inplace=False)

    # Combine the sphere, cone, and cylinder into a single mesh
    pawn = sphere + cone + cylinder

    return pawn

# =============================================================================
# Função: create_triangulated_sphere
# =============================================================================
def create_triangulated_sphere(
    poly_type: str = 'icosahedron', subdivisions: int = 2, radius: float = 1.0
) -> pv.PolyData:
    """
    Cria uma malha 3D pseudo-esférica com faces triangulares.

    O processo envolve:
    1. Criar um poliedro regular base (ex: Icosaedro).
    2. Subdividir suas faces repetidamente para aumentar a resolução e suavidade.
    3. Normalizar as posições dos vértices para que todos fiquem à mesma distância
       do centro, formando uma esfera perfeita.
    4. Escalar a esfera para o raio desejado.
    5. Adicionar dados iniciais às células (faces), como 'FaceID' e 'Highlight'.

    Args:
        poly_type (str, optional): O tipo de poliedro platônico usado como base.
                                   Opções: 'tetrahedron', 'octahedron', 'icosahedron'.
                                   Default: 'icosahedron'.
        subdivisions (int, optional): O número de vezes que a malha base será subdividida
                                     usando o algoritmo 'loop' (adequado para triângulos).
                                     Mais subdivisões geram uma esfera mais suave e detalhada.
                                     Default: 2.
        radius (float, optional): O raio final da esfera gerada. Default: 1.0.

    Returns:
        pv.PolyData: Um objeto PyVista PolyData contendo a geometria e topologia
                     da esfera triangularizada, com os campos 'FaceID' e 'Highlight'
                     adicionados aos dados da célula (`cell_data`).

    Raises:
        ValueError: Se o `poly_type` fornecido não for um dos valores válidos.
    """
    logger.info(f"Iniciando criação da esfera: base={poly_type}, subdivisões={subdivisions}, raio={radius}")
    poly_type_lower = poly_type.lower()

    # 1. Cria o poliedro base
    # Seleciona a geometria inicial com base no argumento poly_type.
    # Para o icosaedro, usamos uma definição manual de vértices e faces
    # (alternativa seria usar pv.Icosahedron() e ajustar).
    if poly_type_lower == 'tetrahedron':
        mesh = pv.Tetrahedron()
    elif poly_type_lower == 'octahedron':
        mesh = pv.Octahedron()
    elif poly_type_lower == 'icosahedron':
        logger.debug("Usando definição manual para icosaedro base.")
        phi = (1.0 + np.sqrt(5.0)) / 2.0 # Proporção áurea
        # Vértices de um icosaedro regular inscrito em uma esfera de raio ~1
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]) / np.sqrt(1 + phi**2) # Normaliza para raio exato 1
        # Índices dos vértices que formam as 20 faces triangulares do icosaedro
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])
        # PyVista requer um formato específico para faces: [N_pontos_face1, p1_idx, p2_idx, ..., N_pontos_face2, ...]
        # Como todas as faces são triângulos (N=3), criamos este array achatado.
        faces_pv = np.hstack((np.full((faces.shape[0], 1), 3, dtype=faces.dtype), faces)).flatten()
        mesh = pv.PolyData(vertices, faces=faces_pv) # Cria o objeto PolyData
    else:
        raise ValueError("poly_type deve ser 'tetrahedron', 'octahedron', ou 'icosahedron'")

    # 2. Subdivide a malha base recursivamente
    # O filtro 'loop' é apropriado para malhas triangulares, suavizando a superfície.
    if subdivisions > 0:
        logger.info(f"Aplicando {subdivisions} subdivisões ('loop' filter)...")
        mesh = mesh.subdivide(subdivisions, subfilter='loop')

    # 3. Projeta os vértices na esfera (normalização) e aplica o raio
    # Isso garante que, independentemente das subdivisões, a forma final seja esférica.
    logger.info("Normalizando vértices e aplicando raio final...")
    # Vetores do centro da malha para cada vértice
    vertex_vectors = mesh.points - mesh.center
    # Distância de cada vértice ao centro (norma L2)
    norms = np.linalg.norm(vertex_vectors, axis=1)
    # Evita divisão por zero para vértices que possam estar no centro (improvável aqui)
    norms[norms == 0] = 1.0
    # Calcula a nova posição: (vetor / norma) dá um vetor unitário na direção do vértice.
    # Multiplica pelo raio desejado e adiciona de volta à posição central da malha.
    mesh.points = mesh.center + (vertex_vectors / norms[:, np.newaxis]) * radius

    logger.info(f"Malha esférica final criada: {mesh.n_points} vértices, {mesh.n_cells} faces.")

    # 4. Adiciona dados personalizados às células (faces)
    # 'FaceID': Um identificador sequencial único (0, 1, 2, ...) para cada face. Útil para referência externa.
    mesh.cell_data['FaceID'] = np.arange(mesh.n_cells)
    # 'Highlight': Um campo numérico para armazenar o estado de destaque de cada face.
    # Inicializa todas as faces com o estado padrão (HIGHLIGHT_DEFAULT = 0).
    mesh.cell_data['Highlight'] = np.full(mesh.n_cells, HIGHLIGHT_DEFAULT, dtype=int)
    # Define 'Highlight' como o array de escalares ativo. O PyVista usará este array
    # para colorir a malha quando um colormap for especificado.
    mesh.set_active_scalars('Highlight')

    return mesh # Retorna o objeto PolyData da esfera pronta.

# =============================================================================
# Função: toggle_connections_mode
# =============================================================================
def toggle_connections_mode():
    """
    Alterna o modo global de definição de vizinhança entre 'points' e 'edges'.

    Esta função é chamada quando a tecla 'X' é pressionada.
    O modo selecionado (`connections_mode`) é usado pela função `find_cell_neighbors`.
    - 'edges': Duas células são vizinhas se compartilharem uma aresta.
    - 'points': Duas células são vizinhas se compartilharem pelo menos um vértice.
    """
    global connections_mode, current_hovered_info
    # Alterna o valor da variável global
    connections_mode = "points" if connections_mode == "edges" else "edges"
    logger.info(f"Modo de conexão para busca de vizinhos alternado para: '{connections_mode}'")
    # Nota: A atualização visual dos vizinhos ocorrerá no próximo movimento do mouse

    # Trigger an update to reflect the new connection mode
    if current_hovered_info["cellid"] >= 0:  # Ensure a valid cell is hovered
        update_highlight(current_hovered_info["cellid"], force_render=True)
# =============================================================================
# Função: find_cell_neighbors
# =============================================================================
def find_cell_neighbors(mesh: pv.PolyData, cellid: int, optional_mode: Optional[str] = None) -> List[int]:
    """
    Encontra os IDs das células vizinhas de uma célula específica na malha.

    A definição de vizinhança depende do modo de conexão ('edges' ou 'points').
    Utiliza a função otimizada `mesh.cell_neighbors()` do PyVista.

    Args:
        mesh (pv.PolyData): A malha (nossa esfera) onde a busca será realizada.
        cellid (int): O índice (ID da célula no PyVista/VTK) da célula central
                      para a qual queremos encontrar os vizinhos.
        optional_mode (Optional[str], optional): Permite sobrescrever temporariamente
                                                o modo de conexão global (`connections_mode`)
                                                apenas para esta chamada. Útil se diferentes
                                                partes do código precisarem de diferentes
                                                critérios de vizinhança. Default: None (usa o modo global).

    Returns:
        List[int]: Uma lista contendo os IDs das células vizinhas válidas (>= 0).
                   Retorna uma lista vazia se o `cellid` de entrada for inválido,
                   se a célula não tiver vizinhos, ou se ocorrer um erro interno.
    """
    # Determina qual modo de conexão usar: o modo opcional, se fornecido e válido, ou o modo global.
    mode = optional_mode if optional_mode in ["points", "edges"] else connections_mode
    # logger.debug(f"Buscando vizinhos para cellid {cellid} usando modo '{mode}'") # Log opcional

    # Verifica se o ID da célula fornecido está dentro dos limites válidos da malha.
    if 0 <= cellid < mesh.n_cells:
        try:
            # Chama a função principal do PyVista para encontrar vizinhos.
            # O parâmetro 'connections' determina se a vizinhança é por aresta ou ponto.
            neighbors = mesh.cell_neighbors(cellid, connections=mode)
            # A função pode retornar IDs negativos em casos de borda, embora não esperado para uma esfera fechada.
            # Filtramos para garantir que apenas IDs de células válidos (>= 0) sejam retornados.
            valid_neighbors = [n for n in neighbors if n >= 0]
            # logger.debug(f"Vizinhos encontrados para {cellid} (modo '{mode}'): {valid_neighbors}") # Log opcional
            return valid_neighbors
        except Exception as e:
            # Captura e loga qualquer erro inesperado durante a busca de vizinhos.
            logger.error(f"Erro ao encontrar vizinhos para célula {cellid} (modo '{mode}'): {e}", exc_info=True)
            return [] # Retorna lista vazia em caso de erro.
    else:
        # Loga um aviso se um ID inválido for passado para a função.
        logger.warning(f"Tentativa de buscar vizinhos para cellid inválido: {cellid}")
        return [] # Retorna lista vazia se o cellid inicial for inválido.

# =============================================================================
# Função: update_labels
# =============================================================================
def update_labels(plotter_ref: pv.Plotter, mesh_ref: pv.PolyData, centers_ref: np.ndarray):
    """
    Atualiza os rótulos de texto exibidos sobre as faces da esfera na cena.

    Esta função:
    1. Remove o ator de rótulos anterior (se existir) da cena.
    2. Determina quais rótulos devem ser exibidos com base no estado de destaque
       atual das células (`mesh_ref.cell_data['Highlight']`) e na flag global
       `show_all_labels`.
    3. Coleta as posições (centros das faces) e os textos para os rótulos a serem exibidos.
    4. Cria e adiciona um novo ator de rótulos à cena com as informações coletadas.

    Args:
        plotter_ref (pv.Plotter): Referência ao objeto Plotter principal onde os rótulos
                                  serão adicionados/removidos.
        mesh_ref (pv.PolyData): Referência à malha da esfera, usada para acessar os dados
                                de 'Highlight' e 'FaceID'.
        centers_ref (np.ndarray): Array com as coordenadas do centro de cada face,
                                  usado para posicionar os rótulos.
    """
    global label_actor, show_all_labels # Acessa as variáveis globais necessárias.

    # Verificação de segurança: Sai da função se algum dos objetos essenciais não estiver pronto.
    if not plotter_ref or not mesh_ref or centers_ref is None:
        logger.warning("Atualização de rótulos abortada: Referências de plotter, malha ou centros ausentes.")
        return

    # 1. Remove o ator de rótulos anterior para evitar duplicação.
    if label_actor is not None:
        try:
            # Remove o ator da cena. `render=False` adia a renderização,
            # o que é mais eficiente se outras atualizações ocorrerão em seguida.
            plotter_ref.remove_actor(label_actor, render=False)
            # logger.debug("Ator de rótulo anterior removido.") # Log opcional
        except ValueError:
            # Pode acontecer se o ator já foi removido por algum motivo. Ignora o erro.
            logger.debug("Ator de rótulo não encontrado para remoção (pode já ter sido removido).")
        finally:
             # Garante que a referência global seja limpa, mesmo se a remoção falhar.
             label_actor = None

        # Se o estado for None, não exibe nenhum rótulo.
    if show_all_labels is None:
        logger.debug("Nenhum rótulo será exibido (estado: None).")
        return

    # 2. Prepara as listas para os novos rótulos.
    labels_to_show = [] # Lista para armazenar os textos dos rótulos.
    points_to_label = [] # Lista para armazenar as posições (coordenadas) dos rótulos.

    # Acessa os dados relevantes da malha de forma eficiente.
    try:
        highlights = mesh_ref.cell_data['Highlight'] # Array com o estado de destaque de cada célula.
        face_ids = mesh_ref.cell_data['FaceID']     # Array com o ID único de cada célula.
    except KeyError as e:
        logger.error(f"Erro ao acessar dados da célula ('Highlight' ou 'FaceID'): {e}. Rótulos não serão atualizados.")
        return

    # 3. Itera sobre todas as células para decidir quais rotular.
    # logger.debug(f"Verificando rótulos para {mesh_ref.n_cells} células. show_all_labels={show_all_labels}") # Log opcional
    for i in range(mesh_ref.n_cells):
        highlight_status = highlights[i] # Obtém o estado de destaque da célula atual.

        # Condição para exibir o rótulo desta célula:
        # - A flag `show_all_labels` está ativa, OU
        # - A célula não está no estado padrão (ou seja, está destacada de alguma forma).
        if show_all_labels or highlight_status != HIGHLIGHT_DEFAULT:
            # Formata o texto do rótulo incluindo o FaceID e o valor numérico do Highlight.
            label_text = f"ID:{face_ids[i]}\nH:{highlight_status}"
            labels_to_show.append(label_text)
            # Adiciona a coordenada do centro da face correspondente à lista de posições.
            points_to_label.append(centers_ref[i])

    # 4. Adiciona os novos rótulos à cena, se houver algum para mostrar.
    if labels_to_show:
        # logger.debug(f"Adicionando {len(labels_to_show)} rótulos à cena.") # Log opcional
        # Usa a função `add_point_labels` do PyVista para criar o ator de rótulos.
        label_actor = plotter_ref.add_point_labels(
            np.array(points_to_label), # Posições dos rótulos (precisa ser um array NumPy).
            labels_to_show,            # Textos dos rótulos.
            font_size=12,              # Tamanho da fonte (ajuste conforme necessário).
            point_size=1,              # Tamanho do ponto de ancoragem (quase invisível).
            shape=None,                # Não desenha forma geométrica no ponto de ancoragem.
            show_points=False,         # Não torna o ponto de ancoragem visível.
            pickable=False,            # Rótulos não precisam ser clicáveis/selecionáveis.
            render=False               # Adia a renderização para o final do ciclo de atualização.
        )
    # else:
        # logger.debug("Nenhum rótulo para exibir desta vez.") # Log opcional

# =============================================================================
# Função: update_highlight
# =============================================================================
def update_highlight(cellid: int, force_render: bool = False):
    """
    Atualiza o estado de destaque visual das células da esfera.

    Esta é a função central da lógica de hover. Ela é chamada sempre que o mouse
    se move sobre a janela (`vtk_mouse_move_callback`).

    O processo realizado é:
    1. Verifica se o `cellid` recebido (célula sob o cursor) é válido e diferente
       do anterior para evitar processamento redundante.
    2. **Reset:** Limpa (volta para `HIGHLIGHT_DEFAULT`) o estado de destaque da
       célula que *estava* anteriormente sob hover e de todos os seus níveis de
       vizinhos armazenados em `current_hovered_info`.
    3. **Cálculo de Novos Vizinhos:**
        - Encontra os vizinhos diretos (nível 1) da nova `cellid`.
        - Encontra os vizinhos dos vizinhos (nível 2), excluindo a `cellid` e os de nível 1.
        - Encontra os vizinhos dos vizinhos de nível 2 (nível 3), excluindo a `cellid` e os de nível 1 e 2.
    4. **Aplicação de Novos Destaques:** Define o estado `Highlight` apropriado para a
       nova `cellid` (HOVER) e para cada nível de vizinho encontrado (NEIGHBOR,
       EXTENDED_NEIGHBOR, EXTREME_NEIGHBOR). A prioridade é dada aos níveis mais
       próximos (ex: uma célula que é vizinha direta não será marcada como vizinha estendida).
    5. **Atualização do Estado Global:** Armazena as informações da nova célula
       sob hover e suas listas de vizinhos em `current_hovered_info`.
    6. **Atualização Visual:** Informa ao PyVista para usar os dados 'Highlight'
       atualizados, chama `update_labels` para refletir mudanças nos rótulos (se houver)
       e finalmente renderiza (redesenha) a cena.
    7. **Caso de Saída:** Se `cellid` for -1 (mouse fora da malha), apenas a etapa de
       reset é executada para limpar qualquer destaque anterior.

    Args:
        cellid (int): O índice (ID VTK/PyVista) da célula que está atualmente
                      sob o cursor do mouse. Será -1 se o cursor estiver fora da malha.
        force_render (bool): Se True, força a renderização mesmo que o cellid não tenha mudado.
    """
    global current_hovered_info, mesh_esfera, plotter, face_centers, floating_sphere_actor, pawn_height # Acessa globais

    # Verificação de segurança essencial
    if mesh_esfera is None or plotter is None or face_centers is None:
        logger.warning("Atualização de destaque ignorada: malha, plotter ou centros de face não estão prontos.")
        return

    needs_render = False # Flag para indicar se a cena precisa ser redesenhada no final

    # --- Parte Principal: Mouse está sobre uma célula válida ---
    if 0 <= cellid < mesh_esfera.n_cells:
        # Otimização: Se o mouse ainda está sobre a mesma célula, não faz nada.
        if cellid == current_hovered_info["cellid"] and not force_render:
            return

        #logger.debug(f"Hover ENTER: cellid={cellid}") # Log opcional
        needs_render = True # Indica que haverá mudança visual
        highlights = mesh_esfera.cell_data['Highlight'] # Acesso direto ao array de dados

         # Update the floating sphere's position
        if floating_sphere_actor is not None:
            # Get the center of the hovered cell
            cell_center = face_centers[cellid]

            # Get the normal vector of the hovered cell
            cell_normals = mesh_esfera.cell_normals  # Ensure normals are precomputed
            cell_normal = cell_normals[cellid]

            # Offset the pawn position slightly above the surface
            offset_distance = 0.275  # Adjust this value as needed
            pawn_position = cell_center + cell_normal * offset_distance

            # Calculate the rotation to align the pawn with the cell normal
            default_up = np.array([1, 0, 0])  # Default "up" direction of the pawn
            rotation_axis = np.cross(default_up, cell_normal)  # Axis of rotation
            rotation_angle = np.arccos(np.dot(default_up, cell_normal)) * (180 / np.pi)  # Angle in degrees

            # Apply the rotation to the pawn
            if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid division by zero
                floating_sphere_actor.SetOrientation(0, 0, 0)  # Reset orientation
                floating_sphere_actor.RotateWXYZ(rotation_angle, *rotation_axis)

            # Update the pawn's position
            floating_sphere_actor.SetVisibility(True)
            floating_sphere_actor.SetPosition(pawn_position)
            logger.debug(f"Pawn moved to cell {cellid} at position {pawn_position} with rotation {rotation_angle}° around {rotation_axis}")

        # --- 1. Reset dos Destaques Anteriores ---
        # Pega informações da célula que *estava* sob hover antes.
        prev_cellid = current_hovered_info["cellid"]
        prev_neighbors = current_hovered_info["neighbors"]
        prev_extended = current_hovered_info["extended_neighbors"]
        prev_extreme = current_hovered_info["extreme_neighbors"]

        # Cria um conjunto com todos os IDs que estavam destacados anteriormente.
        cells_to_potentially_reset = set()
        if 0 <= prev_cellid < mesh_esfera.n_cells:
            cells_to_potentially_reset.add(prev_cellid)
        cells_to_potentially_reset.update(prev_neighbors)
        cells_to_potentially_reset.update(prev_extended)
        cells_to_potentially_reset.update(prev_extreme)

        #logger.debug(f"  Células a verificar para reset: {cells_to_potentially_reset}") # Log opcional

        # Reseta o destaque APENAS se a célula não for ser destacada novamente agora.
        # (Não precisamos calcular os novos vizinhos ainda para esta etapa)
        reset_count = 0
        for reset_id in cells_to_potentially_reset:
             # Verifica se o ID é válido e se não é a *nova* célula sob hover
             # (A verificação completa se ele será vizinho etc. é implícita ao setar os novos highlights depois)
             if 0 <= reset_id < mesh_esfera.n_cells and reset_id != cellid:
                 if highlights[reset_id] != HIGHLIGHT_DEFAULT:
                    highlights[reset_id] = HIGHLIGHT_DEFAULT
                    reset_count += 1
        #logger.debug(f"  {reset_count} células tiveram highlight resetado (excluindo a nova hover).") # Log opcional


        # --- 2. Cálculo e Aplicação dos Novos Destaques ---

        # --- Nível 0: Hover ---
        highlights[cellid] = HIGHLIGHT_HOVER
        #logger.debug(f"  Aplicado HOVER   ({HIGHLIGHT_HOVER}) para cellid {cellid}") # Log opcional

        # --- Nível 1: Vizinhos Diretos ---
        neighbors = find_cell_neighbors(mesh_esfera, cellid) # Usa modo global ('edges' ou 'points')
        neighbors = [n for n in neighbors if n != cellid] # Garante que a própria célula não é sua vizinha
        neighbor_count = 0
        for neighbor_id in neighbors:
            if 0 <= neighbor_id < mesh_esfera.n_cells:
                highlights[neighbor_id] = HIGHLIGHT_NEIGHBOR
                neighbor_count += 1
        #logger.debug(f"  Aplicado NEIGHBOR ({HIGHLIGHT_NEIGHBOR}) para {neighbor_count} vizinhos: {neighbors}") # Log opcional

        # --- Nível 2: Vizinhos Estendidos ---
        extended_neighbors = set()
        for neighbor_id in neighbors:
            # Encontra os vizinhos de cada vizinho direto
            extended = find_cell_neighbors(mesh_esfera, neighbor_id)
            for ext_id in extended:
                # Adiciona à lista se for válido e NÃO for a célula central (hover) nem um vizinho direto (nível 1)
                if 0 <= ext_id < mesh_esfera.n_cells and ext_id != cellid and ext_id not in neighbors:
                    extended_neighbors.add(ext_id)
        extended_count = 0
        for extended_neighbor_id in extended_neighbors:
             # Aplica o highlight APENAS se não estiver já marcado como HOVER ou NEIGHBOR (prioridade maior)
             if highlights[extended_neighbor_id] < HIGHLIGHT_NEIGHBOR:
                highlights[extended_neighbor_id] = HIGHLIGHT_EXTENDED_NEIGHBOR
                extended_count +=1
        #logger.debug(f"  Aplicado EXTENDED ({HIGHLIGHT_EXTENDED_NEIGHBOR}) para {extended_count} vizinhos: {list(extended_neighbors)}") # Log opcional

        # --- Nível 3: Vizinhos Extremos ---
        extreme_neighbors = set()
        for extended_neighbor_id in extended_neighbors:
             # Encontra os vizinhos de cada vizinho estendido (nível 2)
            extreme = find_cell_neighbors(mesh_esfera, extended_neighbor_id)
            for extreme_id in extreme:
                # Adiciona se for válido e NÃO for a célula central, nem vizinho direto, nem vizinho estendido
                if (0 <= extreme_id < mesh_esfera.n_cells and
                        extreme_id != cellid and
                        extreme_id not in neighbors and
                        extreme_id not in extended_neighbors):
                    extreme_neighbors.add(extreme_id)
        extreme_count = 0
        for extreme_neighbor_id in extreme_neighbors:
            # Aplica o highlight APENAS se não estiver já marcado com prioridade maior
            if highlights[extreme_neighbor_id] < HIGHLIGHT_EXTENDED_NEIGHBOR:
                highlights[extreme_neighbor_id] = HIGHLIGHT_EXTREME_NEIGHBOR
                extreme_count += 1
        #logger.debug(f"  Aplicado EXTREME ({HIGHLIGHT_EXTREME_NEIGHBOR}) para {extreme_count} vizinhos: {list(extreme_neighbors)}") # Log opcional


        # --- 3. Atualização do Estado Global ---
        # Guarda as informações da célula e vizinhos recém-calculados.
        current_hovered_info["cellid"] = cellid
        current_hovered_info["neighbors"] = neighbors # Lista
        current_hovered_info["extended_neighbors"] = list(extended_neighbors) # Converte set para lista
        current_hovered_info["extreme_neighbors"] = list(extreme_neighbors) # Converte set para lista


    # --- Parte Secundária: Mouse está fora da malha (cellid < 0) ---
    else:
        # Hide the floating sphere when no cell is hovered
        if floating_sphere_actor is not None:
            floating_sphere_actor.SetVisibility(False)

        # Verifica se havia alguma célula destacada anteriormente para limpar.
        prev_cellid = current_hovered_info["cellid"]
        if prev_cellid != -1: # Só faz algo se antes estava sobre uma célula
            #logger.debug(f"Hover EXIT: cellid={prev_cellid}") # Log opcional
            needs_render = True # Precisa redesenhar para remover os destaques
            highlights = mesh_esfera.cell_data['Highlight']

            # Reseta o destaque da célula anterior e todos os seus níveis de vizinhos.
            cells_to_reset = {prev_cellid} | set(current_hovered_info["neighbors"]) | \
                             set(current_hovered_info["extended_neighbors"]) | \
                             set(current_hovered_info["extreme_neighbors"])

            reset_count = 0
            for reset_id in cells_to_reset:
                 if 0 <= reset_id < mesh_esfera.n_cells:
                     if highlights[reset_id] != HIGHLIGHT_DEFAULT:
                        highlights[reset_id] = HIGHLIGHT_DEFAULT
                        reset_count += 1
            #logger.debug(f"  {reset_count} células tiveram highlight resetado ao sair da malha.") # Log opcional

            # Limpa completamente o estado global de hover.
            current_hovered_info = {
                "cellid": -1,
                "neighbors": [],
                "extended_neighbors": [],
                "extreme_neighbors": []
            }

    # --- 4. Atualização Visual (Comum a ambos os casos, se necessário) ---
    if needs_render:
        # Informa ao PyVista para usar o array 'Highlight' modificado para colorir.
        mesh_esfera.set_active_scalars('Highlight')
        # Atualiza os rótulos (eles podem precisar mudar com base nos novos highlights).
        update_labels(plotter, mesh_esfera, face_centers)
        # Solicita que o PyVista redesenhe a cena com todas as atualizações.
        plotter.render()

# =============================================================================
# Função: vtk_mouse_move_callback
# =============================================================================
def vtk_mouse_move_callback(vtk_interactor: vtk.vtkRenderWindowInteractor, event_name: str):
    """
    Função de callback executada pelo sistema VTK sempre que o mouse se move na janela.

    Esta função atua como uma ponte entre o evento de baixo nível do VTK e a
    lógica de atualização de destaque da nossa aplicação.

    Usa um `vtkCellPicker` pré-configurado (`hover_picker`) para determinar eficientemente
    qual célula da malha `mesh_esfera` está sob as coordenadas atuais do mouse.
    Em seguida, chama `update_highlight` passando o ID da célula encontrada (ou -1).

    Args:
        vtk_interactor (vtk.vtkRenderWindowInteractor): O objeto interator da janela VTK
                                                      que originou o evento. Fornece
                                                      métodos para obter informações do evento,
                                                      como a posição do mouse.
        event_name (str): O nome do evento VTK que disparou o callback
                         (geralmente "MouseMoveEvent").
    """
    global hover_picker, plotter # Acessa os objetos globais necessários.

    # Verificação de segurança: Garante que os objetos essenciais (picker, plotter, renderer)
    # foram inicializados antes de prosseguir.
    if hover_picker is None or plotter is None or plotter.renderer is None:
        # logger.warning("vtk_mouse_move_callback ignorado: objetos essenciais não prontos.") # Log opcional
        return

    # 1. Obtém as coordenadas (x, y) da posição atual do cursor do mouse na janela 2D.
    x, y = vtk_interactor.GetEventPosition()

    # 2. Realiza a operação de "picking" (seleção).
    # O vtkCellPicker "dispara um raio" da posição da câmera através do pixel (x, y)
    # na tela e verifica se ele intersecta alguma célula do(s) ator(es)
    # configurado(s) na sua PickList (neste caso, apenas o ator da esfera).
    # O terceiro argumento (0) representa a coordenada Z (geralmente ignorada para pick 2D).
    # O quarto argumento é o renderer onde a seleção deve ocorrer.
    hover_picker.Pick(x, y, 0, plotter.renderer)

    # 3. Obtém o resultado do picking.
    # GetCellId() retorna o índice (ID) da célula que foi atingida pelo raio.
    # Retorna -1 se nenhuma célula na PickList foi atingida naquela posição.
    cellid = hover_picker.GetCellId()
    # logger.debug(f"Mouse at ({x},{y}), Picked cellid: {cellid}") # Log opcional

    # 4. Chama a função principal de atualização do destaque visual.
    # Delega toda a lógica de como tratar o hover (seja sobre uma célula ou fora)
    # para a função `update_highlight`.
    update_highlight(cellid)

# =============================================================================
# Função: click_callback
# =============================================================================
def click_callback(picked_mesh: Optional[pv.PolyData], cellid: int):
    """
    Função de callback executada pelo PyVista quando uma célula da malha é clicada.

    Esta função é registrada usando `plotter.enable_cell_picking()`.
    Quando um clique válido ocorre em uma célula:
    1. Obtém o `FaceID` (nosso identificador único) correspondente ao `cellid` clicado.
    2. Loga a informação do clique (índice VTK e FaceID).
    3. Adiciona o `FaceID` à lista global `selected_cells`, se ainda não estiver presente.
    4. Loga a lista atualizada de células selecionadas.

    Args:
        picked_mesh (Optional[pv.PolyData]): A malha que foi efetivamente clicada.
                                             Será `None` se o clique ocorrer fora de
                                             qualquer malha configurada para picking.
        cellid (int): O índice (ID VTK/PyVista) da célula específica que foi clicada
                     dentro da `picked_mesh`.
    """
    global selected_cells # Permite modificar a lista global de seleções.

    # Verifica se o clique foi realmente sobre uma célula válida da nossa malha esfera.
    # `picked_mesh` deve ser o nosso `mesh_esfera` (ou pelo menos conter 'FaceID').
    # `cellid` deve ser um índice válido dentro dessa malha.
    if picked_mesh is None or not (0 <= cellid < picked_mesh.n_cells):
        logger.info("Clique detectado fora de uma célula válida.")
        return

    try:
        # Acessa o dado 'FaceID' que adicionamos à malha.
        # Usa o `cellid` (índice da célula clicada) para obter o `FaceID` correspondente.
        face_id = picked_mesh.cell_data['FaceID'][cellid]
        logger.info(f"Célula Clicada! Índice VTK: {cellid}, Nosso FaceID: {face_id}")

        # Adiciona o FaceID à lista de selecionados, evitando duplicatas.
        if face_id not in selected_cells:
            selected_cells.append(face_id)
            logger.info(f"FaceID {face_id} adicionado à seleção.")
            # --- Ponto de Extensão ---
            # Aqui seria um bom lugar para adicionar feedback visual para a seleção, por exemplo:
            # 1. Definir um novo estado HIGHLIGHT_SELECTED = 5
            # 2. Atualizar o colormap e clim para incluir essa nova cor.
            # 3. Fazer: mesh_esfera.cell_data['Highlight'][cellid] = HIGHLIGHT_SELECTED
            # 4. Chamar plotter.render()
            # Ou implementar lógica de deseleção se clicar novamente.
        else:
            logger.info(f"FaceID {face_id} já estava na lista de selecionados.")
            # --- Ponto de Extensão (Deseleção) ---
            # selected_cells.remove(face_id)
            # logger.info(f"FaceID {face_id} removido da seleção.")
            # Resetar highlight: mesh_esfera.cell_data['Highlight'][cellid] = HIGHLIGHT_DEFAULT (ou HOVER se ainda estiver sob mouse)
            # plotter.render()

        # Loga a lista completa de IDs selecionados (ordenada para facilitar a leitura).
        logger.info(f"IDs das Faces Selecionadas atuais: {sorted(selected_cells)}")

    except (KeyError, IndexError) as e:
        # Trata erros que podem ocorrer se 'FaceID' não existir na malha clicada
        # ou se o `cellid` for inválido por algum motivo inesperado.
        logger.error(f"Erro ao acessar FaceID para célula {cellid}: {e}", exc_info=True)
    except Exception as e:
         # Captura genérica para outros erros inesperados.
         logger.error(f"Erro inesperado no callback de clique para célula {cellid}: {e}", exc_info=True)

# =============================================================================
# Função: toggle_labels_callback
# =============================================================================
def toggle_labels_callback():
    """
    Callback executado quando a tecla 'L' é pressionada.

    Alterna entre os três estados de exibição de rótulos:
    - True: Exibe todos os rótulos.
    - False: Exibe apenas os rótulos das células destacadas.
    - None: Não exibe nenhum rótulo.
    """
    global show_all_labels, plotter, mesh_esfera, face_centers

    # Verifica se os componentes essenciais para atualizar os rótulos estão prontos.
    if plotter and mesh_esfera and face_centers is not None:
        # Cicla entre os estados: True -> False -> None -> True
        if show_all_labels is True:
            show_all_labels = False
        elif show_all_labels is False:
            show_all_labels = None
        else:
            show_all_labels = True

        logger.info(f"Alternando estado de exibição de rótulos para: {show_all_labels}")

        # Atualiza os rótulos na tela com base no novo estado.
        update_labels(plotter, mesh_esfera, face_centers)

        # Redesenha a cena para refletir as mudanças.
        plotter.render()
    else:
        logger.warning("Toggle labels ignorado: estado da aplicação incompleto.")

# =============================================================================
# Função: main
# Objetivo: Orquestrar a criação e exibição da cena interativa.
# =============================================================================
def main():
    """
    Função principal que configura e executa a aplicação de visualização interativa.

    Responsabilidades:
    1. Define os parâmetros para a criação da esfera.
    2. Chama `create_triangulated_sphere` para gerar a malha.
    3. Calcula e armazena os centros das faces para posicionamento de rótulos.
    4. Configura o objeto `Plotter` do PyVista (janela, tema).
    5. Adiciona a malha da esfera à cena do plotter, configurando sua aparência
       (cores baseadas em 'Highlight', mapa de cores, arestas, iluminação, etc.).
    6. Configura o `vtkCellPicker` para a detecção eficiente do hover do mouse.
    7. Habilita e configura as interações do usuário:
        - Hover do mouse (usando observador VTK).
        - Clique do mouse (usando `enable_cell_picking` do PyVista).
        - Tecla 'L' para alternar rótulos.
        - Tecla 'X' para alternar modo de vizinhança.
    8. Inicializa a exibição dos rótulos chamando `update_labels` uma vez.
    9. Loga as instruções de controle para o usuário.
    10. Inicia o loop de eventos da janela (`plotter.show()`), que mantém a
        aplicação rodando e respondendo às interações até ser fechada.
    11. Loga as células selecionadas ao final da execução.
    """
    # Permite que esta função modifique as variáveis globais que serão usadas
    # pelos callbacks e outras partes da aplicação.
    global selected_cells, hover_picker, mesh_esfera, plotter, face_centers
    global floating_sphere_actor, floating_sphere # Acessa globais

    # --- 1. Configuração e Criação da Esfera ---
    poly_base = 'icosahedron'    # Poliedro base (mais uniforme para esfera geodésica).
    num_subdivisions = 5         # Nível de detalhe (resolução) da esfera. Ajuste conforme necessário.
    esfera_radius = 15.0          # Raio da esfera no espaço 3D.

    logger.info("Iniciando configuração da cena...")
    try:
        # Cria a malha da esfera usando a função auxiliar.
        mesh_esfera = create_triangulated_sphere(
            poly_type=poly_base,
            subdivisions=num_subdivisions,
            radius=esfera_radius
        )
        # Calcula e armazena globalmente os centros das faces.
        face_centers = mesh_esfera.cell_centers().points
        logger.info("Malha da esfera e centros das faces criados com sucesso.")
    except Exception as e:
        logger.critical(f"Falha crítica ao criar a malha da esfera: {e}", exc_info=True)
        return # Aborta a execução se a esfera não puder ser criada.

    # --- 2. Configuração do Plotter (Janela e Cena) ---
    pv.set_plot_theme("document") # Define um tema visual (cores de fundo, texto, etc.). Outros: "dark", "paraview".
    plotter = pv.Plotter(window_size=WINDOW_SIZE, title="Visualizador de Esfera Interativa com Hover") # Cria a janela.

    # Adiciona a malha da esfera como um ator na cena do plotter.
    try:
        # Guarda a referência ao ator adicionado, necessária para o picker.
        actor = plotter.add_mesh(
            mesh_esfera,                # O objeto PolyData da esfera.
            scalars='Highlight',        # Nome do array nos dados da célula ('cell_data') usado para colorir.
            cmap=discrete_cmap,         # O mapa de cores customizado definido anteriormente.
            # clim=[min_val, max_val]: Define o intervalo de valores escalares que serão mapeados para as cores do cmap.
            # Como temos 5 estados (0 a 4), o intervalo [0, 4] garante que todos sejam mapeados corretamente
            # para as 5 cores do 'discrete_cmap'.
            clim=[HIGHLIGHT_DEFAULT, HIGHLIGHT_HOVER], # Mapeia valores de 0 a 4.
            show_edges=True,            # Exibe as arestas dos triângulos que formam as faces.
            edge_color='black',          # Cor das arestas.
            line_width=0.5,             # Espessura das linhas das arestas.
            lighting='light kit',       # Usa um conjunto padrão de luzes para iluminar a cena.
            preference='cell',          # Indica que os dados escalares ('Highlight') são associados às células (faces), não aos pontos (vértices).
            pickable=True,              # Permite que este ator seja detectado por operações de picking (hover, clique).
            show_scalar_bar=False       # Não exibe a legenda de cores (barra escalar) na tela.
        )
        logger.info("Malha adicionada ao plotter com sucesso.")
    except Exception as e:
         logger.critical(f"Falha ao adicionar a malha ao plotter: {e}", exc_info=True)
         return # Aborta se não conseguir adicionar o ator principal.

    #Adiciona uma esfera acima da célula selecionada (opcional)
    # Create a small blue sphere to float over the hovered cell
    try:
        # Create the pawn-shaped object
        pawn_mesh = create_pawn()

        # Add the pawn to the scene
        floating_sphere_actor = plotter.add_mesh(
            pawn_mesh,
            color='pink',  # Color of the pawn
            show_edges=True,  # No edges for the pawn
            edge_color='black',  # Color of the edges
            line_width=0.5,  # Thickness of the edges
            opacity=1,  # Slight transparency for better visualization
            pickable=False  # The pawn should not interfere with picking
        )
        floating_sphere_actor.SetVisibility(False)  # Initially hide the pawn
    except Exception as e:
        logger.error(f"Erro ao adicionar objeto em forma de peão: {e}", exc_info=True)
        floating_sphere_actor = None
        logger.info("Objeto em forma de peão não criado.")

        return # Aborta se não conseguir adicionar a esfera flutuante.

    # --- 3. Configuração do Picker VTK para Detecção de Hover ---
    # Usamos um vtkCellPicker diretamente para ter controle sobre a detecção
    # contínua do mouse sobre as células (hover).
    hover_picker = vtk.vtkCellPicker()
    hover_picker.SetTolerance(PICKER_TOLERANCE) # Define a sensibilidade do picker.
    # Configura o picker para considerar APENAS o ator da nossa esfera.
    # Isso melhora o desempenho, pois ele não precisa testar outros atores na cena.
    hover_picker.AddPickList(actor)
    hover_picker.PickFromListOn() # Ativa o uso exclusivo da PickList definida.
    logger.info("vtkCellPicker configurado para detecção de hover na esfera.")

    # --- 4. Configuração das Interações do Usuário ---

    # a) Interação de Hover (Movimento do Mouse) via Observador VTK
    # Acessa o interator da janela VTK subjacente ao plotter PyVista.
    if plotter.iren: # Verifica se o interator está disponível.
        # Adiciona um "observador" que chamará nossa função `vtk_mouse_move_callback`
        # sempre que o evento `MouseMoveEvent` do VTK ocorrer na janela.
        plotter.iren.add_observer(vtk.vtkCommand.MouseMoveEvent, vtk_mouse_move_callback)
        logger.info("Callback para hover (MouseMoveEvent) registrado.")
    else:
        # Situação incomum, mas registra um erro se não for possível configurar o hover.
        logger.error("vtkRenderWindowInteractor (plotter.iren) não encontrado. Hover não funcionará.")

    # b) Interação de Clique na Célula via PyVista
    # Usa a função de conveniência do PyVista para lidar com cliques discretos.
    plotter.enable_cell_picking(
        callback=click_callback, # Associa o clique à nossa função `click_callback`.
                                 # A lambda `lambda mesh, idx: click_callback(mesh, idx)` é implícita aqui.
        show=False,              # Desativa o destaque visual padrão do PyVista ao clicar.
        show_message=False,      # Desativa a mensagem de log padrão do PyVista ao clicar.
        use_picker=False         # Importante: Usa o picker interno do PyVista para cliques, não o nosso `hover_picker`.
    )
    logger.info("Callback para clique em células (enable_cell_picking) registrado.")

    # c) Interação por Teclado - Alternar Visibilidade dos Rótulos
    # Associa a pressão da tecla 'l' (minúscula) à nossa função `toggle_labels_callback`.
    plotter.add_key_event("l", toggle_labels_callback)
    logger.info("Callback para tecla 'L' (alternar rótulos) registrado.")

    # d) Interação por Teclado - Alternar Modo de Conexão para Vizinhos
    # Associa a pressão da tecla 'x' (minúscula) à nossa função `toggle_connections_mode`.
    plotter.add_key_event("x", toggle_connections_mode)
    logger.info("Callback para tecla 'X' (alternar modo de vizinhança 'edges'/'points') registrado.")

    # --- 5. Inicialização dos Rótulos ---
    # Chama a função de atualização dos rótulos uma vez no início para exibir
    # o estado inicial (provavelmente todos os rótulos, se show_all_labels=True).
    logger.info("Inicializando exibição dos rótulos...")
    update_labels(plotter, mesh_esfera, face_centers)

    # --- 6. Exibição de Instruções e Início da Janela ---
    logger.info("--- Controles ---")
    logger.info(f" - Hover Mouse: Destaca face ({discrete_cmap.colors[-1].upper()}) e vizinhos")
    logger.info(f"   - Nível 1: ({discrete_cmap.colors[HIGHLIGHT_NEIGHBOR].upper()})")
    logger.info(f"   - Nível 2: ({discrete_cmap.colors[HIGHLIGHT_EXTENDED_NEIGHBOR].upper()})")
    logger.info(f"   - Nível 3: ({discrete_cmap.colors[HIGHLIGHT_EXTREME_NEIGHBOR].upper()})")
    logger.info(f" - Clique Mouse: Seleciona face (registra ID no console)")
    logger.info(f" - Tecla 'L': Alterna visibilidade dos rótulos (Todos / Destacados)")
    logger.info(f" - Tecla 'X': Alterna critério de vizinhança (Arestas / Pontos)")
    logger.info(f" - Controles Padrão de Câmera:")
    logger.info(f"   - Botão Esquerdo + Arrastar: Rotacionar")
    logger.info(f"   - Botão Direito + Arrastar / Roda Mouse: Zoom")
    logger.info(f"   - Botão Meio + Arrastar: Mover (Pan)")
    logger.info("-----------------")
    logger.info("Iniciando loop de eventos da janela. Feche a janela para sair.")

    # Abre a janela interativa e inicia o loop de eventos VTK/PyVista.
    # O script ficará bloqueado aqui até que a janela seja fechada pelo usuário.
    plotter.show()

    # --- Código Pós-Execução ---
    # Será executado apenas após o fechamento da janela.
    logger.info("Janela fechada. Encerrando a aplicação.")
    logger.info(f"IDs das Faces Selecionadas durante a execução: {sorted(selected_cells)}")

# --- Ponto de Entrada do Script ---
# A construção `if __name__ == "__main__":` garante que a função `main()`
# só seja executada quando o script é rodado diretamente (não quando importado como módulo).
if __name__ == "__main__":
    main()