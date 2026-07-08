# Methods for automated compilation of a .tex file report on a pinned group

from pathlib import Path
import sympy as sp
from root_system import build_directed_dynkin_graph

groups_tex_path = Path("./groups_tex")
template_file = groups_tex_path / "template.tex"
        
def pinned_group_to_tex(pinned_group):
    # Build .tex document section from a pinned_group object
    
    import re
    
    # 1. Basic group attributes
    #       a. (TO DO: EQUATIONS, EPSILON AS SYMBOL) Summary table - name string, matrix size, rank, defining equations
    #       b. (COMPLETE) Bilinear form info (if there is one) - summary table (dimension, Witt index, primitive elemtn, epsilon), matrix
    #       c. (TO DO: EQUATIONS) Lie algebra info - defining equations, generic Lie algebra element
    #       d. (COMPLETE) Torus info - generic torus element, trivial characters
    # 2. Root system
    #       a. (COMPLETE) Summary table - Dynkin type, irreducible?, reduced?, simply laced?, number of roots
    #       b. (TO DO: BETTER CENTERING) Dynkin diagram
    #       c. (COMPLETE) Cartan matrix
    #       d. (COMPLETE) Table of roots with info on simple, positive/negative, multipliable
    #       e. (COMPLETE) List of root, coroot pairs
    #       f. (TO DO: SORT BY NUMBER OF PAIRS) Table of linear combinations of roots
    # 3. Root spaces
    #       a. Table with root, root space dimension
    #       b. Table with root, generic root space element
    # 4. Root subgroups
    #       a. Table with root, generic root subgroup element
    #       b. Table with root, homomorphism defect coefficient
    #       c. Explicit equations for homomorphism/pseudo-homomorphism property
    # 5. Commutators
    #       a. Table of linear combinations of roots
    #       b. Table with root1, root2, commutator coefficient
    #       c. Explicit equations for commutator relation
    # 6. Weyl group
    #       a. Table of root, Weyl group element
    #       b. Table of root1, root2, Weyl group conjugation coefficient
    #       c. Explicit equations for Weyl group conjugation relation
    
    # Read the template
    if not template_file.exists():
        raise FileNotFoundError(f"Template not found at {template_file}. Please create it first.")
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    G = pinned_group
    Phi = G.root_system
    
    # Group basics, Lie algebra, torus
    replacements = {
        "GroupNamePlaceholder": f"{G.name_string}",
        "MatrixSizePlaceholder": f"{G.matrix_size}",
        "RankPlaceholder": f"{G.rank}",
        "GenericLieAlgebraElementPlaceholder": sp.latex(G.generic_lie_algebra_element(letter = 'x')),
        "GenericTorusElementPlaceholder": sp.latex(G.generic_torus_element(letter = 't')),
        "TrivialCharactersPlaceholder": sp.latex(sp.Matrix(Phi.lattice_matrix).T)
    }
    
    # Bilinear form
    if G.form is not None:
        replacements["BilinearFormPlaceholder"] = nondegenerate_isotropic_form_to_tex(G.form)
    else:
        replacements["BilinearFormPlaceholder"] = ""
    
    # Root system
    replacements.update({
        "DynkinTypePlaceholder": '$' + re.sub(r'(\d+)', r'_\1', Phi.name_string).replace('_x_', ' \\times ') + '$',
        "IrreduciblePlaceholder": f"{Phi.is_irreducible}",
        "ReducedPlaceholder": f"{Phi.is_reduced}",
        "SimplyLacedPlaceholder": f"{Phi.is_simply_laced}",
        "NumberOfRootsPlaceholder": len(Phi.root_list),
        "RootTablePlaceholder": build_root_table(Phi),
        "CorootTablePlaceholder": build_coroot_table(Phi),
        "LinearCombinationsPlaceholder": build_root_linear_combos_table(Phi)
    })
    if Phi.is_irreducible:
        # For irreducible root systems, the Cartan matrix and Dynkin diagram are simpler to handle
        replacements['rootsystemirreduciblefalse'] = 'rootsystemirreducibletrue'
        replacements['CartanMatrixPlaceholder'] = sp.latex(sp.Matrix(Phi.cartan_matrix))
        replacements['DynkinDiagramPlaceholder'] = build_dynkin_tex(Phi)
    else:
        # For a reducible root system, each connected component has
        # its own Cartan matrix and Dynkin diagram
        replacements['rootsystemirreducibletrue'] = 'rootsystemirreduciblefalse'
        replacements['RootSystemComponentsPlaceholder'] = f'{len(Phi.components)}'
        cartan_tex = ''
        dynkin_diagram_tex = ''
        for comp in Phi.components:
            cartan_tex += " " + sp.latex(sp.Matrix(comp.cartan_matrix))
            dynkin_diagram_tex += " " + build_dynkin_tex(comp)
        replacements['CartanMatrixPlaceholder'] = cartan_tex
        replacements['DynkinDiagramPlaceholder'] = dynkin_diagram_tex
    
    # Make substitutions
    for placeholder, value in replacements.items():
        if value is None:
            # Fallback to an explicit string token so .replace doesn't crash
            value = "None"
        content = content.replace(placeholder, str(value))
    
    return content

def root_system_to_tex(root_system):
    # Build .tex document section on root system
    
    # Table with basic summary: Dynkin type, irreducible?, reduced?, simply laced?, number of roots
    # Dynkin diagram
    # Cartan matrix
    # Table of roots with info on simple, positive/negative, multipliable
    # List of root, coroot pairs
    # Table on linear combinations of roots
    
    return ""

def nondegenerate_isotropic_form_to_tex(form):
    # Build .tex document section on the form
    # Note that form may be None, in which case this should return ""
    
    # Table with basic summary: dimension, Witt index, primitive element, epsilon
    # Matrix

    # If there is no matrix or name, it's a trivial or non-existent form
    if form.matrix is None or not form.name_string:
        return "No invariant bilinear form defined for this group configuration."
        
    tex = f"\\subsection{{{form.name_string.capitalize()} form}}\n"
    
    # Format optional/none fields safely for text mode
    prim_elem = f"${sp.latex(form.primitive_element)}$" if form.primitive_element is not None else "N/A"
    eps_val = f"${form.epsilon}$" if form.epsilon is not None else "N/A"

    # Using raw strings r"..." removes escaping head-scratchers entirely
    tex += r"\begin{tabular}{|l|l|}" + "\n"
    tex += r"    \hline" + "\n" 
    tex += f"    Dimension & {form.dimension} \\\\\n"
    tex += r"    \hline" + "\n"  
    tex += f"    Witt index & {form.witt_index} \\\\\n"
    tex += r"    \hline" + "\n"  
    tex += f"    Primitive element & {prim_elem} \\\\\n"
    tex += r"    \hline" + "\n"  
    tex += f"    Epsilon & {eps_val} \\\\\n"
    tex += r"    \hline" + "\n"  
    tex += r"\end{tabular}" + "\n\n"
    
    tex += "\\bigskip\n"  # Added a newline for clean LaTeX output
    tex += f"\\noindent Matrix: \\(\n{sp.latex(form.matrix)}\n\\)\n\n"
    
    return tex


def build_dynkin_tex(Phi):
    """
    Builds the TikZ code for a single irreducible root system component.
    
    Accepts an irreducible root system (or root system component) 'Phi', 
    extracts/builds its internal subgraph representation, calculates the geometric 
    layout, and generates a standalone TikZ picture string.
    """
    
    assert(Phi.is_irreducible)
    
    # 1. Cleanly pull the subgraph dictionary out of the component
    if hasattr(Phi, 'dynkin_graph'):
        subgraph = Phi.dynkin_graph
    else:
        # Fallback if the component hasn't computed its graph map yet
        subgraph = build_directed_dynkin_graph(Phi.simple_roots)
        
    # Assert that we are truly parsing an irreducible graph slice
    degrees = {v: len(neighbors) for v, neighbors in subgraph.items()}
    fork_nodes = [v for v, deg in degrees.items() if deg >= 3]
    
    assert len(fork_nodes) <= 1, "Passed component is not irreducible (multiple branching junctions found)"
    
    node_positions = {}

    # --- Branched component layout (D, E) ---
    if fork_nodes:
        fork = fork_nodes[0]
        
        def get_path_from_neighbor(start_node):
            path = [start_node]
            visited = {fork, start_node}
            curr = start_node
            while True:
                next_nodes = [n for n in subgraph[curr] if n not in visited]
                if not next_nodes: 
                    break
                curr = next_nodes[0]
                path.append(curr)
                visited.add(curr)
            return path

        branches = [get_path_from_neighbor(n) for n in subgraph[fork].keys()]
        branches.sort(key=len)
        
        vertical_branch = branches[0]
        left_branch = branches[1]
        right_branch = branches[2]
        left_branch.reverse()
        
        backbone = left_branch + [fork] + right_branch
        
        for x_idx, node in enumerate(backbone):
            node_positions[node] = (x_idx * 1.5, 0.0)
                
        fork_x, _ = node_positions[fork]
        for y_idx, node in enumerate(vertical_branch):
            node_positions[node] = (fork_x, (y_idx + 1) * 1.2)

    # --- Linear component layout (A, B, C, F, G, or isolated points) ---
    else:
        leaf_nodes = [v for v, deg in degrees.items() if deg == 1]
        start = min(leaf_nodes) if leaf_nodes else min(subgraph) if subgraph else None
        
        if start is not None:
            visited = {start}
            order = [start]
            current = start
            while True:
                unvisited = [u for u in subgraph[current] if u not in visited]
                if not unvisited: 
                    break
                next_node = unvisited[0]
                order.append(next_node)
                visited.add(next_node)
                current = next_node
                
            for x_idx, node in enumerate(order):
                node_positions[node] = (x_idx * 1.5, 0.0)

    # --- BUILD ENVIRONMENT STRING ---
    comp_tex = []
    comp_tex.append(r"\begin{tikzpicture}[")
    comp_tex.append(r"    node/.style={circle, draw, fill=black!5, minimum size=6mm, inner sep=0pt},")
    comp_tex.append(r"    arrow/.style={-latex, thick},")
    comp_tex.append(r"    doublearrow/.style={double, double distance=2pt, -latex, thick},")
    comp_tex.append(r"    triplearrow/.style={double, double distance=4pt, -latex, thick},")
    comp_tex.append(r"    plain/.style={thick},")
    comp_tex.append(r"    baseline=(current bounding box.center)")
    comp_tex.append(r"]")

    # Render nodes
    for node, (x, y) in node_positions.items():
        comp_tex.append(f"    \\node[node] ({node}) at ({x:.2f}, {y:.2f}) {{${node}$}};")

    # Render edges
    drawn_edges = set()
    for u in subgraph:
        for v in subgraph[u]:
            if (u, v) in drawn_edges or (v, u) in drawn_edges:
                continue
            
            mult_u_to_v = subgraph[u].get(v, 1)
            mult_v_to_u = subgraph[v].get(u, 1)
            
            if mult_u_to_v > mult_v_to_u:
                if mult_u_to_v == 2:
                    comp_tex.append(f"    \\draw[doublearrow] ({u}) -- ({v});")
                elif mult_u_to_v == 3:
                    comp_tex.append(f"    \\draw[plain] ({u}) -- ({v});")
                    comp_tex.append(f"    \\draw[doublearrow] ({u}) -- ({v});")
                else:
                    comp_tex.append(f"    \\draw[arrow] ({u}) -- node[above] {{{mult_u_to_v}}} ({v});")
            elif mult_u_to_v < mult_v_to_u:
                if mult_v_to_u == 2:
                    comp_tex.append(f"    \\draw[doublearrow] ({v}) -- ({u});")
                elif mult_v_to_u == 3:
                    comp_tex.append(f"    \\draw[plain] ({v}) -- ({u});")
                    comp_tex.append(f"    \\draw[doublearrow] ({v}) -- ({u});")
                else:
                    comp_tex.append(f"    \\draw[arrow] ({v}) -- node[above] {{{mult_v_to_u}}} ({u});")
            else:
                comp_tex.append(f"    \\draw[plain] ({u}) -- ({v});")
            
            drawn_edges.add((u, v))

    comp_tex.append(r"\end{tikzpicture}")
    return "\n".join(comp_tex)

def build_root_table(Phi):
    tex = ""
    tex += r"\begin{tabular}{|l|c|c|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Simple} & \\textbf{Sign} & \\textbf{Multipliable} \\\\\n"
    tex += r"    \hline" + "\n"
    
    for alpha in Phi.root_list:
        alpha_latex = f"${sp.latex(alpha)}$"
        
        if Phi.is_irreducible:
            target_simple = Phi.simple_roots
            target_positive = Phi.positive_roots
        else:
            comp = next((c for c in Phi.components if any(alpha.equals(r) for r in c.root_list)), None)
            target_simple = comp.simple_roots if comp else []
            target_positive = comp.positive_roots if comp else []

        is_simple = any(alpha.equals(s) for s in target_simple)
        is_positive = any(alpha.equals(p) for p in target_positive)
        is_multipliable = Phi.is_multipliable_root(alpha)
        
        simple_str = "Yes" if is_simple else "No"
        sign_str = "$+$" if is_positive else "$-$"
        multipliable_str = "Yes" if is_multipliable else "No"
        
        tex += f"    {alpha_latex} & {simple_str} & {sign_str} & {multipliable_str} \\\\\n"
        tex += r"    \hline" + "\n"
        
    tex += "\\end{tabular}\n"
    return tex

def build_coroot_table(Phi):
    tex = ""
    tex += r"\begin{tabular}{|l|l|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Coroot} \\\\\n"
    tex += r"    \hline" + "\n"
    for alpha in Phi.root_list:
        alpha_coroot = Phi.coroot_dict[alpha]
        alpha_latex = f"${sp.latex(alpha)}$"
        coroot_latex = f"${sp.latex(alpha_coroot)}$"
        tex += f"    {alpha_latex} & {coroot_latex} \\\\\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{tabular}\n"
    return tex

def build_root_linear_combos_table(Phi):
    pair_rows = []
    for idx, alpha in enumerate(Phi.root_list):
        for beta in Phi.root_list[idx + 1:]:
            combos = Phi.integer_linear_combos(alpha, beta)
            if combos:
                alpha_lat = sp.latex(alpha)
                beta_lat = sp.latex(beta)
                sorted_pairs = sorted(combos.keys())
                pairs_str = ", ".join(f"({i},{j})" for (i, j) in sorted_pairs)
                pair_rows.append(f"    ${alpha_lat}$ & ${beta_lat}$ & ${pairs_str}$ \\\\\n")

    combo_table = ""
    if pair_rows:
        combo_table += r"\begin{tabular}{|l|l|c|}" + "\n"
        combo_table += r"    \hline" + "\n"
        combo_table += "    $\\alpha$ & $\\beta$ & Pairs $(i,j)$ \\\\\n"
        combo_table += r"    \hline" + "\n"
        for row in pair_rows:
            combo_table += row
            combo_table += r"    \hline" + "\n"
        combo_table += "\\end{tabular}\n"
    else:
        combo_table += "\\noindent No pairs of roots generate positive integer linear combinations.\n"

    return combo_table
