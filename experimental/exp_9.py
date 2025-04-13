# -*- coding: utf-8 -*-
import pyvista as pv
import numpy as np
import vtk
import logging
from typing import Optional, List, Tuple

# --- Basic Logger Setup ---

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# ----------------------------------------------------

def sort_cell_indices_around_vertex(
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


def create_goldberg_polyhedron(radius: float = 1.0, subdivisions: int = 1) -> Optional[pv.PolyData]:
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
        logger.info(f"Creating Goldberg Polyhedron (Manual Dual): subdivisions={subdivisions}, radius={radius}")

        # Step 1 & 2: Create and subdivide the icosahedron (Geodesic Dome)
        base_icosahedron = pv.Icosahedron()
        if subdivisions > 0:
            subdivided_mesh = base_icosahedron.subdivide(subdivisions, subfilter='loop')
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
                sorted_dual_vertex_indices = sort_cell_indices_around_vertex(
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
        goldberg_mesh.points = origin + (vertex_vectors / norms[:, np.newaxis]) * radius
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


        logger.info(f"Goldberg polyhedron created successfully: {goldberg_mesh.n_points} points, {goldberg_mesh.n_cells} cells.")
        return goldberg_mesh

    except ValueError as ve:
         logger.error(f"Value error during Goldberg creation: {ve}", exc_info=False)
         return None
    except Exception as e:
        logger.error(f"Unexpected error creating Goldberg polyhedron: {e}", exc_info=True)
        return None

# --- Example Usage ---
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG) # Use DEBUG to see detailed steps

    # Create Goldberg polyhedron (more subdivisions = more faces)
    # subdivisions=1 -> GP(1,1) class? (Check Goldberg theory)
    # subdivisions=2 -> GP(2,0)/GP(1,2)/GP(2,1) class? (Depends on mapping details)
    gp_mesh = create_goldberg_polyhedron(subdivisions=2, radius=5.0)

    if gp_mesh:
        print(f"\nSuccessfully created Goldberg Polyhedron:")
        print(f" - Number of Points (Vertices): {gp_mesh.n_points}")
        print(f" - Number of Cells (Faces): {gp_mesh.n_cells}")

        # Verify cell types again
        cell_n_points = [gp_mesh.get_cell(i).n_points for i in range(gp_mesh.n_cells)]
        unique_counts, counts = np.unique(cell_n_points, return_counts=True)
        print(f" - Cell Types Counts: {dict(zip(unique_counts, counts))}")  # Should show 5s and 6s

        plotter = pv.Plotter(window_size=[800, 800])
        plotter.add_mesh(gp_mesh, show_edges=True, edge_color='black', color='goldenrod')
        plotter.add_text("Goldberg Polyhedron (Manual Dual)", font_size=10)
        print("\nDisplaying plot...")
        plotter.show()
    else:
        print("\nFailed to create Goldberg Polyhedron.")