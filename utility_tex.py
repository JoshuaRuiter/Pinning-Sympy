# Methods for automated compilation of a .tex file report on a pinned group

from pathlib import Path
import sympy as sp
from utility_general import vector_variable

groups_tex_path = Path("./groups_tex")
template_file = groups_tex_path / "template.tex"
        
def pinned_group_to_tex(pinned_group):
    # Build .tex document section from a pinned_group object
    
    import re
    
    # 1. Basic group attributes
    #       a. Summary table - name string, matrix size, rank, defining equations
    #       b. Bilinear form info (if there is one) - 
    #                       summary table (dimension, Witt index, primitive elemtn, epsilon), matrix
    #       c. Lie algebra info - defining equations, generic Lie algebra element
    #       d. Torus info - generic torus element, trivial characters
    # 2. Root system
    #       a. Summary table - Dynkin type, irreducible?, reduced?, simply laced?, number of roots
    #       b. Dynkin diagram
    #       c. Cartan matrix
    #       d. Table of roots with info on simple, norm^2, positive/negative, multipliable
    #       e. List of root, coroot pairs
    #       f. Table of linear combinations of roots
    # 3. Root spaces
    #       a. Table with root, root space dimension
    #       b. Table with root, generic root space element
    # 4. Root subgroups
    #       a. Table with root, generic root subgroup element
    #       b. Table with root, homomorphism defect coefficient
    #       c. Symbolic and matrix pseudo-homomorphism equations
    # 5. Commutators
    #       a. Table of linear combinations of roots
    #       b. Table with root1, root2, commutator coefficient
    #       c. Symbolic and matrix commutator equations
    # 6. Weyl group
    #       a. Table of root, Weyl group element
    #       b. Table of root1, root2, Weyl group conjugation coefficient
    #       c. Symbolic and matrix Weyl conjugation equations
    
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
        "DefiningEquationPlaceholder" : G.group_constraints_string,
        "LieAlgebraEquationPlaceholder" : G.lie_algebra_constraints_string,
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
        "CorootTablePlaceholder": build_coroot_table(Phi)
    })
    if Phi.is_irreducible:
        # For irreducible root systems, the Cartan matrix and Dynkin diagram are simpler to handle
        replacements['rootsystemirreduciblefalse'] = 'rootsystemirreducibletrue'
        replacements['CartanMatrixPlaceholder'] = sp.latex(sp.Matrix(Phi.cartan_matrix))
        replacements['DynkinDiagramPlaceholder'] = build_dynkin_tex(Phi)
    else:
        # For a reducible root system, each connected component has its own Cartan matrix and Dynkin diagram
        replacements['rootsystemirreducibletrue'] = 'rootsystemirreduciblefalse'
        replacements['RootSystemComponentsPlaceholder'] = f'{len(Phi.components)}'
        cartan_tex = ''
        dynkin_diagram_tex = ''
        for comp in Phi.components:
            cartan_tex += " " + sp.latex(sp.Matrix(comp.cartan_matrix))
            dynkin_diagram_tex += " " + build_dynkin_tex(comp)
        replacements['CartanMatrixPlaceholder'] = cartan_tex
        replacements['DynkinDiagramPlaceholder'] = dynkin_diagram_tex
    
    # Root spaces, subgroups, Weyl group
    replacements.update({
        "RootSpaceDimensionPlaceholder": build_dimension_table(G),
        "RootSpaceTablePlaceholder" : build_root_space_table(G),
        "RootSubgroupTablePlaceholder": build_root_subgroup_table(G),
        "WeylGroupPlaceholder" : build_weyl_element_table(G),
        "WeylConjugationCoefficientPlaceholder" : build_weyl_conjugation_table(G),
        "WeylConjugationEquationsPlaceholder" : build_weyl_conjugation_equations(G)
    })
    
    # Homomorphism defect coefficients (only for non-reduced case)
    if Phi.is_reduced:
        replacements['rootsystemreducedfalse'] = 'rootsystemreducedtrue'
    else:
        replacements.update({
            "rootsystemreducedtrue" : 'rootsystemreducedfalse',
            "HomomorphismDefectTablePlaceholder": build_hom_defect_table(G),
            "HomomorphismDefectEquationsPlaceholder" : build_hom_defect_equations(G)
        })
    
    # Commutators
    if len(Phi.summable_non_proportional_pairs) > 0:
        replacements.update({
            "summablepairsfalse" : 'summablepairstrue',
            "LinearCombinationsPlaceholder": build_root_linear_combos_table(Phi),
            "CommutatorCoefficientTablePlaceholder" : build_commutator_table(G),
            "CommutatorEquationsPlaceholder" : build_commutator_equations(G)
        })
    else:
        replacements.update({
            "summablepairstrue" : 'summablepairsfalse',
            "LinearCombinationsPlaceholder": "",
            "CommutatorCoefficientTablePlaceholder" : "",
            "CommutatorgEquationsPlaceholder" : "",
            "CommutatorSymbolicEquationsPlaceholder" : ""
        })
    
    # Make substitutions
    for placeholder, value in replacements.items():
        if value is None:
            # Fallback to an explicit string token so .replace doesn't crash
            value = "None"
        content = content.replace(placeholder, str(value))
    
    return content

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
    # Build a TikZ-based LaTeX diagram for a single irreducible root system component
    assert Phi.is_irreducible
    
    family = Phi.dynkin_type
    rank = int(Phi.rank)
    
    # Standard configuration parameters for uniform rendering
    r = 0.12        # Node radius
    step = 0.8      # Distance between nodes
    
    # Boilerplate setup for the TikZ picture
    tikz_lines = [
        r"\begin{center}",
        r"\begin{tikzpicture}[scale=1, baseline=(current bounding box.center),",
        r"    node/.style={circle, draw, minimum size=" + f"{2*r}cm" + r", inner sep=0pt, thick, fill=white},",
        r"    filled/.style={circle, draw, minimum size=" + f"{2*r}cm" + r", inner sep=0pt, thick, fill=black}",
        r"]"
    ]

    if family == 'A':
        # Simple path graph: circles connected by single segments
        for i in range(1, rank + 1):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")

    elif family == 'B':
        # Path graph with a double bond at the end pointing to the last short root (filled or double arrow)
        for i in range(1, rank + 1):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        # Double bond between the last two nodes
        if rank >= 2:
            tikz_lines.append(f"  \\draw[thick, double, double distance=2pt] (v{rank-1}) -- (v{rank});")
            mid_x = (rank - 1.5) * step
            tikz_lines.append(f"  \\draw[thick] ({mid_x + 0.05}, 0.1) -- ({mid_x - 0.05}, 0) -- ({mid_x + 0.05}, -0.1);")

    elif family == 'C':
        # Path graph with a double bond at the end pointing towards the long root
        for i in range(1, rank + 1):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        if rank >= 2:
            tikz_lines.append(f"  \\draw[thick, double, double distance=2pt] (v{rank-1}) -- (v{rank});")
            mid_x = (rank - 1.5) * step
            tikz_lines.append(f"  \\draw[thick] ({mid_x - 0.05}, 0.1) -- ({mid_x + 0.05}, 0) -- ({mid_x - 0.05}, -0.1);")

    elif family == 'BC':
        # Path graph with a double bond pointing to the end, with the last node filled black
        for i in range(1, rank + 1):
            if i == rank:
                tikz_lines.append(f"  \\node[filled] (v{i}) at ({(i-1)*step}, 0) {{}};")
            else:
                tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        if rank >= 2:
            tikz_lines.append(f"  \\draw[thick, double, double distance=2pt] (v{rank-1}) -- (v{rank});")
            # Arrow pointing to the right towards the filled root (v_n)
            mid_x = (rank - 1.5) * step
            tikz_lines.append(f"  \\draw[thick] ({mid_x + 0.05}, 0.1) -- ({mid_x - 0.05}, 0) -- ({mid_x + 0.05}, -0.1);")

    elif family == 'D':
        # Path graph with a fork on the right side
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 2):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        
        # Forked nodes at the end
        if rank >= 3:
            spine_end_x = (rank - 3) * step
            tikz_lines.append(f"  \\node[node] (v{rank-1}) at ({spine_end_x + step}, {0.4*step}) {{}};")
            tikz_lines.append(f"  \\node[node] (v{rank}) at ({spine_end_x + step}, {-0.4*step}) {{}};")
            tikz_lines.append(f"  \\draw[thick] (v{rank-2}) -- (v{rank-1});")
            tikz_lines.append(f"  \\draw[thick] (v{rank-2}) -- (v{rank});")

    elif family == 'E':
        # Straight path of length rank - 1, with an offshoot branch on node 3
        for i in range(1, rank):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        # Long branch element down or up from node 3
        tikz_lines.append(f"  \\node[node] (v{rank}) at ({2*step}, {step}) {{}};")
        tikz_lines.append(f"  \\draw[thick] (v3) -- (v{rank});")

    elif family == 'F' and rank == 4:
        # F4: Four nodes, central double bond pointing to short roots (right)
        for i in range(1, 5):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        tikz_lines.append(r"  \\draw[thick] (v1) -- (v2);")
        tikz_lines.append(r"  \\draw[thick, double, double distance=2pt] (v2) -- (v3);")
        tikz_lines.append(r"  \\draw[thick] (v3) -- (v4);")
        mid_x = 1.5 * step
        tikz_lines.append(f"  \\draw[thick] ({mid_x + 0.05}, 0.1) -- ({mid_x - 0.05}, 0) -- ({mid_x + 0.05}, -0.1);")

    elif family == 'G' and rank == 2:
        # G2: Two nodes with a triple bond pointing to short root (right)
        tikz_lines.append(f"  \\node[node] (v1) at (0, 0) {{}};")
        tikz_lines.append(f"  \\node[node] (v2) at ({step}, 0) {{}};")
        tikz_lines.append(r"  \\draw[thick] (v1) -- (v2);")
        tikz_lines.append(r"  \\draw[thick, transform canvas={yshift=2pt}] (v1) -- (v2);")
        tikz_lines.append(r"  \\draw[thick, transform canvas={yshift=-2pt}] (v1) -- (v2);")
        mid_x = 0.5 * step
        tikz_lines.append(f"  \\draw[thick] ({mid_x + 0.05}, 0.1) -- ({mid_x - 0.05}, 0) -- ({mid_x + 0.05}, -0.1);")

    tikz_lines.append(r"\end{tikzpicture}")
    tikz_lines.append(r"\end{center}")
    
    return "\n".join(tikz_lines)

def build_root_table(Phi):
    # Buidl a table with list of roots and basic info (squared norm, simple, positive/negative, multipliable)
    row_data = []
    for alpha in Phi.root_list:
        alpha_latex = f"${sp.latex(alpha)}$"
        short_alpha = Phi.short_form_roots[alpha]
        short_alpha_latex = f"${sp.latex(short_alpha)}$"
        
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
        
        sq_norm = alpha.dot(alpha)
        sq_norm_latex = f"${sp.latex(sq_norm)}$"
        
        simple_str = "Yes" if is_simple else "No"
        sign_str = "$+$" if is_positive else "$-$"
        multipliable_str = "Yes" if is_multipliable else "No"
        
        # Included the short_alpha_latex as the second column
        row_text = f"    {alpha_latex} & {short_alpha_latex} & {sq_norm_latex} & {simple_str} & {sign_str} & {multipliable_str} \\\\\n"
        
        # Sort key:
        # 1. sq_norm ascending
        # 2. is_simple descending (True/0 before False/1)
        # 3. is_positive descending (True/0 before False/1)
        sort_key = (sq_norm, 0 if is_simple else 1, 0 if is_positive else 1)
        row_data.append((sort_key, row_text))

    # Sort based on the multi-level key
    row_data.sort(key=lambda x: x[0])

    # Construct the LaTeX table
    table_lines = [
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"    \hline",
        r"    \textbf{Root} & \textbf{Short Form} & \textbf{Norm$^2$} & \textbf{Simple} & \textbf{Sign} & \textbf{Multipliable} \\",
        r"    \hline"
    ]
    
    for _, row in row_data:
        table_lines.append(row.rstrip())
        table_lines.append(r"    \hline")
        
    table_lines.append(r"\end{tabular}" + "\n")
    return "\n".join(table_lines)

def build_coroot_table(Phi):
    # Build a Latex table with root, coroot pairs
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
    # Build a table which stores all information about valid pairs of positive integer
    # linear combinations of roots that are also roots
    
    # Fail immediately if no non-proportional pairs exist
    assert len(Phi.summable_non_proportional_pairs) > 0, "Root system has no summable non-proportional pairs."
    
    pair_rows = []
    root_indices = {root: idx for idx, root in enumerate(Phi.root_list)}
    
    # Filter for unique unordered pairs
    unique_pairs = (
        (alpha, beta) for alpha, beta in Phi.summable_non_proportional_pairs
        if root_indices[alpha] < root_indices[beta]
    )
    
    for alpha, beta in unique_pairs:
        combos = Phi.integer_linear_combos(alpha, beta)
        if combos:
            # Use the short-form view mapping for both roots
            short_alpha = Phi.short_form_roots[alpha]
            short_beta = Phi.short_form_roots[beta]
            
            alpha_lat = sp.latex(short_alpha)
            beta_lat = sp.latex(short_beta)
            
            sorted_pairs = sorted(combos.keys())
            pairs_str = ", ".join(f"({i},{j})" for (i, j) in sorted_pairs)
            row_text = f"    ${alpha_lat}$ & ${beta_lat}$ & {pairs_str} \\\\"
            
            pair_rows.append((len(sorted_pairs), row_text))

    # Sort rows by the number of linear combinations
    pair_rows.sort(key=lambda x: x[0])

    # Construct the LaTeX table using longtable
    table_lines = [
        r"\noindent \begin{longtable}{|l|l|c|}",
        r"    \hline",
        r"    $\alpha$ & $\beta$ & Pairs $(i,j)$ \\",
        r"    \hline",
        r"    \endhead"
    ]
    
    for _, row in pair_rows:
        table_lines.append(row.rstrip())
        table_lines.append(r"    \hline")
        
    table_lines.append(r"\end{longtable}" + "\n")
    return "\n".join(table_lines)

def build_commutator_table(group):
    # Builde a table of all commutator coefficients
    
    Phi = group.root_system
    assert len(group.commutator_coefficient_dict) > 0

    # Gather blocks of rows grouped by (alpha, beta) so we can sort by total number of combinations
    table_blocks = []

    for (alpha, beta) in Phi.summable_non_proportional_pairs:
        d_alpha = group.root_space_dimension(alpha)
        d_beta = group.root_space_dimension(beta)
        u = vector_variable(letter='u', length=d_alpha)
        v = vector_variable(letter='v', length=d_beta)
        
        linear_combos = Phi.integer_linear_combos(alpha, beta)
        if not linear_combos:
            continue
            
        # Retrieve the shortened forms of roots for formatting
        short_alpha = Phi.short_form_roots[alpha]
        short_beta = Phi.short_form_roots[beta]
        
        alpha_lat = sp.latex(short_alpha)
        beta_lat = sp.latex(short_beta)
        
        block_rows = []
        # Sort keys to ensure deterministic table ordering per pair
        for key in sorted(linear_combos.keys()):
            i, j = key[0], key[1]
            combo = linear_combos[key]
            
            coeff = group.commutator_coefficient_map(alpha, beta, i, j, u, v)
            raw_expr = coeff[0] if len(coeff) == 1 else coeff.T
            
            # Use the shortened representation for the linear combination as well
            short_combo = Phi.short_form_roots[combo]
            
            combo_lat = sp.latex(short_combo)
            expr_lat = sp.latex(raw_expr)
            
            # Format single row for this specific coefficient entry
            row_text = (
                f"    ${alpha_lat}$ & ${beta_lat}$ & {i} & {j} & "
                f"${combo_lat}$ & \\adjustbox{{max width=0.5\\textwidth}}{{${expr_lat}$}} \\\\\n"
            )
            block_rows.append(row_text)
            
        if block_rows:
            # Store the count of entries for this root pair, and the accumulated rows
            table_blocks.append((len(block_rows), block_rows))

    # Sort blocks by the quantity of entries (index 0 of the tuple)
    table_blocks.sort(key=lambda x: x[0])

    combo_table = ""
    if table_blocks:
        # 6 columns matching the elements extracted per row, using longtable
        combo_table += r"\noindent \begin{longtable}{|l|l|c|c|l|c|}" + "\n"
        combo_table += r"    \hline" + "\n"
        combo_table += (
            "    $\\alpha$ & $\\beta$ & $i$ & $j$ & "
            " $i\\alpha + j\\beta$ & $N_{ij}^{\\alpha\\beta}(u,v)$ \\\\\n"
        )
        combo_table += r"    \hline" + "\n"
        combo_table += r"    \endhead" + "\n"
        
        # Unpack the sorted blocks and write rows sequentially
        for _, rows in table_blocks:
            for row in rows:
                combo_table += row
                combo_table += r"    \hline" + "\n"
                
        combo_table += "\\end{longtable}\n"
    else:
        combo_table += "\\noindent No pairs of roots generate positive integer linear combinations.\n"
        
    return combo_table

def _unwrap_scalar(val):
    # Helper function to extract a scalar value from a 1-dimensional wrapper like a
    # Sympy matrix, list, tuple, etc.
    
    if val is None:
        return val

    # Handle any SymPy Matrix type (Dense, Sparse, Immutable, etc.)
    if hasattr(val, 'shape') and hasattr(val, '__getitem__'):
        # A matrix is 1D if it is 1x1, or a flat vector of length 1
        if val.shape == (1, 1) or val.shape == (1,) or val.shape == (1, 1, 1) or len(val) == 1:
            return val[0]
            
    # Handle standard python sequences (list, tuple, etc.)
    # We exclude strings and dicts since they behave differently
    elif isinstance(val, (list, tuple)) and len(val) == 1:
        return val[0]
        
    return val

def build_commutator_equations(group):
    # Build a LaTeX string containing interleaved commutator equations.
    # For each summable, non-proportional root pair, it outputs:
    #  1. The root parameter prefix (\alpha, \beta) on its own line.
    #  2. The symbolic commutator relation on its own line (scaled if necessary).
    #  3. The concrete matrix counterpart equation on its own line (scaled if necessary).
    
    # Each equation is typeset in its own display block to prevent TeX memory exhaustion,
    # and scaled down if it exceeds \\textwidth using \\scaletoalign.
    
    equations = []
    
    for (alpha, beta) in group.root_system.summable_non_proportional_pairs:
        assert group.root_system.is_root(alpha + beta)
        
        # --- 1. Set Up Shared Symbolic Variables ---
        d_alpha = group.root_space_dimension(alpha)
        u = vector_variable(letter='u', length=d_alpha)
        
        d_beta = group.root_space_dimension(beta)
        v = vector_variable(letter='v', length=d_beta)
        
        u_elem = _unwrap_scalar(u)
        v_elem = _unwrap_scalar(v)
        
        u_latex = sp.latex(u_elem)
        v_latex = sp.latex(v_elem)
        
        # --- 2. Format Prefix Line ---
        alpha_latex = sp.latex(alpha)
        beta_latex = sp.latex(beta)
        
        prefix_line = (
            "\\[\n"
            rf"\alpha = {alpha_latex}, \quad \beta = {beta_latex}"
            "\n\\]"
        )
        equations.append(prefix_line)
        
        # --- 3. Format Symbolic Equation ---
        lhs_symbolic = (
            rf"\left[ X_{{\alpha}}\left({u_latex}\right), "
            rf"X_{{\beta}}\left({v_latex}\right) \right]"
        )
        
        linear_combos = group.root_system.integer_linear_combos(alpha, beta)
        rhs_symbolic_terms = []
        
        for key in linear_combos:
            i = key[0]
            j = key[1]
            coeff = group.commutator_coefficient_map(alpha, beta, i, j, u, v)
            
            coeff_elem = _unwrap_scalar(coeff)
            coeff_latex = sp.latex(coeff_elem)
            
            # Format the subscript label (e.g., "i\alpha + j\beta")
            subscript_parts = []
            if i != 0:
                coeff_i = "" if i == 1 else str(i)
                subscript_parts.append(rf"{coeff_i}\alpha")
            if j != 0:
                coeff_j = "" if j == 1 else str(j)
                subscript_parts.append(rf"{coeff_j}\beta")
            
            subscript_latex = " + ".join(subscript_parts)
            term_latex = rf"X_{{{subscript_latex}}}\left({coeff_latex}\right)"
            rhs_symbolic_terms.append(term_latex)
            
        rhs_symbolic = " ".join(rhs_symbolic_terms)
        
        symbolic_eq_line = (
            "\\[\n"
            "\\scaletoalign{\\textwidth}{"
            f"{lhs_symbolic} = {rhs_symbolic}"
            "}\n"
            "\\]"
        )
        equations.append(symbolic_eq_line)
        
        # --- 4. Format Matrix Counterpart Equation ---
        x_alpha_u = group.root_subgroup_map(alpha, u)
        x_beta_v = group.root_subgroup_map(beta, v)
        
        x_alpha_u_inv = x_alpha_u**(-1)
        x_beta_v_inv = x_beta_v**(-1)
        
        # Multiply out the matrix RHS
        RHS_matrix = sp.eye(group.matrix_size)
        for key in linear_combos:
            i = key[0]
            j = key[1]
            root = linear_combos[key]
            
            coeff = group.commutator_coefficient_map(alpha, beta, i, j, u, v)
            new_factor = group.root_subgroup_map(root, coeff)
            RHS_matrix = RHS_matrix * new_factor
            
        X_alpha_latex = f"{{{sp.latex(x_alpha_u)}}}"
        X_beta_latex = f"{{{sp.latex(x_beta_v)}}}"
        X_alpha_inv_latex = f"{{{sp.latex(x_alpha_u_inv)}}}"
        X_beta_inv_latex = f"{{{sp.latex(x_beta_v_inv)}}}"
        RHS_matrix_latex = f"{{{sp.latex(RHS_matrix)}}}"
        
        matrix_eq_line = (
            "\\[\n"
            "\\scaletoalign{\\textwidth}{"
            f"{X_alpha_latex} {X_beta_latex} {X_alpha_inv_latex} {X_beta_inv_latex} = {RHS_matrix_latex}"
            "}\n"
            "\\]"
        )
        equations.append(matrix_eq_line)
        
    if equations:
        macro_def = (
            "\\providecommand{\\scaletoalign}[2]{%\n"
            "  \\sbox0{$#2$}% \n"
            "  \\ifdim\\wd0>#1\\relax \n"
            "    \\resizebox{#1}{!}{\\usebox0}% \n"
            "  \\else \n"
            "    \\usebox0% \n"
            "  \\fi \n"
            "}\n"
        )
        tex = macro_def + "\n".join(equations) + "\n"
    else:
        tex = ""
        
    return tex

def build_dimension_table(group):
    # Build a table of root, root space dimension
    
    Phi = group.root_system
    tex = r"\noindent \begin{tabular}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    \textbf{$\alpha$} & \textbf{$d_{\alpha}$} \\" + "\n"
    tex += r"    \hline" + "\n"
    
    # 1. Gather roots and pre-calculate their dimensions
    table_data = []
    for alpha in Phi.root_list:
        dim = group.root_space_dimension(alpha)
        table_data.append((alpha, dim))
        
    # 2. Sort the data by the dimension (the second element in the tuple) in ascending order
    table_data.sort(key=lambda item: item[1])
    
    # 3. Build the LaTeX table from the sorted data
    for alpha, dim in table_data:
        alpha_latex = f"${sp.latex(alpha)}$"
        tex += f"    {alpha_latex} & {dim} \\\\\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{tabular}\n"
    return tex

def build_root_space_table(group):
    # Build a table of root spaces
    Phi = group.root_system
    tex = r"\noindent \begin{longtable}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Generic element of root space} \\\\\n"
    tex += r"    \hline" + "\n"
    tex += r"    \endhead" + "\n" 
    for alpha in Phi.root_list:
        d_alpha = group.root_space_dimension(alpha)
        u = vector_variable(letter = 'u', length = d_alpha)
        X_alpha_u = group.root_space_map(alpha, u)
        alpha_latex = f"{sp.latex(alpha)}"
        X_alpha_u_latex = f"{{{sp.latex(X_alpha_u)}}}"
        tex += f"    ${alpha_latex}$ & ${X_alpha_u_latex}$ \\\\\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{longtable}\n"
    return tex

def build_root_subgroup_table(group):
    # Build a table of root subgroups
    Phi = group.root_system
    tex = r"\noindent \begin{longtable}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Generic element of root subgroup} \\\\\n"
    tex += r"    \hline" + "\n"
    tex += r"    \endhead" + "\n"
    for alpha in Phi.root_list:
        d_alpha = group.root_space_dimension(alpha)
        u = vector_variable(letter = 'u', length = d_alpha)
        x_alpha_u = group.root_subgroup_map(alpha, u)
        alpha_latex = f"{sp.latex(alpha)}"
        x_alpha_u_latex = f"{{{sp.latex(x_alpha_u)}}}" 
        tex += f"    ${alpha_latex}$ & ${x_alpha_u_latex}$ \\\\\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{longtable}\n"
    return tex

def build_hom_defect_table(group):
    # Build a table of homomorphism defect coefficients for multipliable roots
    # (these do not exist for non-multipliable roots)
    Phi = group.root_system
    assert(not Phi.is_reduced)
    tex = r"\noindent \begin{tabular}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    $\alpha$ & $q^i_\alpha(u,v)$ \\" + "\n"
    tex += r"    \hline" + "\n"
    for alpha in Phi.root_list:
        if Phi.is_multipliable_root(alpha):
            hdc = group.homomorphism_defect_coefficient_dict[alpha][2]
            
            # Retrieve the shortened view representation of the root
            short_alpha = Phi.short_form_roots[alpha]
            alpha_latex = f"{sp.latex(short_alpha)}"
            
            hdc_latex = f"{{{sp.latex(hdc)}}}" 
            tex += f"    ${alpha_latex}$ & ${hdc_latex}$ \\\\\n"
            tex += r"    \hline" + "\n"
    tex += "\\end{tabular}\n"
    return tex

def build_hom_defect_equations(group):
    # Generate a LaTeX string containing interleaved homomorphism defect equations.
    # For each multipliable root alpha, it outputs:
    #  1. The root prefix parameters on their own line.
    #  2. The symbolic defect equation on its own line.
    #  3. The concrete matrix equation counterpart on its own line.
    
    Phi = group.root_system
    assert not Phi.is_reduced, "Root system must be non-reduced to have multipliable roots"
    
    equations = []
    for alpha in Phi.root_list:
        if Phi.is_multipliable_root(alpha):
            alpha_latex = f"{sp.latex(alpha)}"
            
            # --- 1. Compute and Retrieve Shared Variables ---
            dict_entry = group.homomorphism_defect_coefficient_dict[alpha]
            u = dict_entry[0]
            v = dict_entry[1]
            hdc = group.homomorphism_defect_map(alpha, u, v)
            
            # Just for some sanity checks
            d_alpha = group.root_space_dimension(alpha)
            d_2alpha = group.root_space_dimension(2*alpha)
            assert len(u) == d_alpha
            assert len(v) == d_alpha
            assert len(hdc) == d_2alpha
            
            # --- 2. Format Prefix Line ---
            prefix_line = (
                "\\[\n"
                rf"\alpha = {alpha_latex}"
                "\n\\]"
            )
            equations.append(prefix_line)
            
            # --- 3. Format Symbolic Equation Line ---
            u_elem = _unwrap_scalar(u)
            v_elem = _unwrap_scalar(v)
            hdc_elem = _unwrap_scalar(hdc)
            
            X_alpha_u_sym = r"X_{\alpha}\left(" + sp.latex(u_elem) + r"\right)"
            X_alpha_v_sym = r"X_{\alpha}\left(" + sp.latex(v_elem) + r"\right)"
            X_alpha_sum_sym = r"X_{\alpha}\left(" + sp.latex(u_elem) + " + " + sp.latex(v_elem) + r"\right)"
            X_2alpha_hdc_sym = r"X_{2\alpha}\left(" + sp.latex(hdc_elem) + r"\right)"
            
            symbolic_eq_line = (
                "\\[\n"
                rf"{X_alpha_u_sym} {X_alpha_v_sym} = {X_alpha_sum_sym} \cdot {X_2alpha_hdc_sym}"
                "\n\\]"
            )
            equations.append(symbolic_eq_line)
            
            # --- 4. Format Matrix Counterpart Equation Line ---
            X_alpha_u_mat = group.root_subgroup_map(alpha, u)
            X_alpha_v_mat = group.root_subgroup_map(alpha, v)
            X_2alpha_hdc_mat = group.root_subgroup_map(2*alpha, hdc)
            X_alpha_sum_mat = group.root_subgroup_map(alpha, u+v)

            X_alpha_u_mat_latex = f"{{{sp.latex(X_alpha_u_mat)}}}"
            X_alpha_v_mat_latex = f"{{{sp.latex(X_alpha_v_mat)}}}"
            X_2alpha_hdc_mat_latex = f"{{{sp.latex(X_2alpha_hdc_mat)}}}"
            X_alpha_sum_mat_latex = f"{{{sp.latex(X_alpha_sum_mat)}}}"
            
            matrix_eq_line = (
                "\\[\n"
                "\\resizebox{\\textwidth}{!}{$"
                f"{X_alpha_u_mat_latex} {X_alpha_v_mat_latex} = {X_alpha_sum_mat_latex} {X_2alpha_hdc_mat_latex}"
                "$}\n"
                "\\]"
            )
            equations.append(matrix_eq_line)
            
    if equations:
        tex = "\n".join(equations) + "\n"
    else:
        tex = ""
        
    return tex

def build_weyl_element_table(group):
    # Generate a table of Weyl elemnets
    
    Phi = group.root_system
    tex = r"\noindent \begin{longtable}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    $\alpha$ & $w_\alpha$ \\" + "\n" # Added LaTeX row separator
    tex += r"    \hline" + "\n"
    tex += r"    \endhead" + "\n"
    for alpha in Phi.root_list:
        w_alpha = group.weyl_element_map(alpha)
        
        # Retrieve the shortened view representation of the root
        short_alpha = Phi.short_form_roots[alpha]
        alpha_latex = f"{sp.latex(short_alpha)}"
        
        w_alpha_latex = f"{{{sp.latex(w_alpha)}}}"
        # Using fr"..." lets you safely use \\ for LaTeX newlines
        tex += fr"    ${alpha_latex}$ & ${w_alpha_latex}$ \\" + "\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{longtable}\n"
    return tex

def build_weyl_conjugation_table(group):
    # Generate a table of Weyl conjugation coefficients
    
    Phi = group.root_system
    
    # 1. Setup the LaTeX table alignment and headers with longtable
    tex = r"\noindent \begin{longtable}{|l|l|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    \textbf{$\alpha$} & \textbf{$\beta$} & \textbf{$\gamma = \sigma_{\alpha}(\beta)$} & $\varphi_{\alpha\beta}(u)$ \\" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    \endhead" + "\n"
    
    # 2. Iterate through every possible pair of roots
    for alpha in Phi.root_list:
        for beta in Phi.root_list:
            d_beta = group.root_space_dimension(beta)
            u = vector_variable(letter='u', length=d_beta)
            
            gamma = group.root_system.reflect_root(
                hyperplane_root=alpha,
                root_to_reflect=beta
            )
            
            d_gamma = group.root_space_dimension(gamma)
            assert d_beta == d_gamma
            
            phi_u = group.weyl_conjugation_coefficient_map(alpha, beta, u)
            raw_expr = phi_u[0] if len(phi_u) == 1 else phi_u.T
            
            # Retrieve the shortened view representations
            short_alpha = Phi.short_form_roots[alpha]
            short_beta = Phi.short_form_roots[beta]
            short_gamma = Phi.short_form_roots[gamma]
            
            # 3. Convert all elements to LaTeX strings
            alpha_latex = f"${sp.latex(short_alpha)}$"
            beta_latex = f"${sp.latex(short_beta)}$"
            gamma_latex = f"${sp.latex(short_gamma)}$"
            expr_latex = f"${sp.latex(raw_expr)}$"
            
            # 4. Append row to the table
            tex += f"    {alpha_latex} & {beta_latex} & {gamma_latex} & {expr_latex} \\\\\n"
            tex += r"    \hline" + "\n"
            
    tex += "\\end{longtable}\n"
    return tex

def build_weyl_conjugation_equations(group):
    # Generate a LaTeX string containing interleaved Weyl conjugation equations.
    # For each unique Weyl element representative, each pair of roots alpha and beta 
    # will output:
    #  1. The root prefix parameters on their own line.
    #  2. The symbolic equation on its own line.
    #  3. The matrix equation on its own line.
    
    # If multiple roots alpha yield the exact same Weyl matrix w_alpha, only the 
    # first encountered representative is processed to eliminate redundancy.
    
    equations = []
    seen_weyl_matrices = []  # Tracks unique w_alpha matrices we've processed
    
    for alpha in group.root_system.root_list:
        w_alpha = group.weyl_element_map(alpha)
        
        # Check if we have already printed equations for this specific Weyl matrix
        if any(w_alpha.equals(seen) for seen in seen_weyl_matrices):
            continue
            
        seen_weyl_matrices.append(w_alpha)
        w_alpha_inverse = w_alpha.inv()
        
        for beta in group.root_system.root_list:
            gamma = group.root_system.reflect_root(hyperplane_root=alpha, root_to_reflect=beta)
            d_beta = group.root_space_dimension(beta)
            d_gamma = group.root_space_dimension(gamma)
            assert d_beta == d_gamma, "Reflected roots have mismatched dimensions"
            
            # --- 1. Compute Shared Variables ---
            u = vector_variable('u', d_beta)
            u_elem = _unwrap_scalar(u)
            u_latex = sp.latex(u_elem)
            
            phi_u = group.weyl_conjugation_coefficient_map(alpha, beta, u)
            phi_u_elem = _unwrap_scalar(phi_u)
            phi_u_latex = sp.latex(phi_u_elem)
            
            # --- 2. Format Prefix Line ---
            alpha_val_latex = sp.latex(alpha)
            beta_val_latex = sp.latex(beta)
            gamma_val_latex = sp.latex(gamma)
            
            prefix = (
                rf"\alpha = {alpha_val_latex}, "
                rf"\beta = {beta_val_latex}, "
                rf"\sigma_{{\alpha}}(\beta) = {gamma_val_latex}"
            )
            
            prefix_line = (
                "\\[\n"
                f"{prefix}\n"
                "\\]"
            )
            equations.append(prefix_line)
            
            # --- 3. Format Symbolic Equation ---
            lhs_symbolic = (
                rf"w_{{\alpha}} "
                rf"X_{{\beta}}\left({u_latex}\right) "
                rf"w_{{\alpha}}^{{-1}}"
            )
            rhs_symbolic = rf"X_{{\sigma_{{\alpha}}\left(\beta\right)}}\left({phi_u_latex}\right)"
            
            symbolic_eq_line = (
                "\\[\n"
                "\\scaletoalign{\\textwidth}{"
                f"{lhs_symbolic} = {rhs_symbolic}"
                "}\n"
                "\\]"
            )
            equations.append(symbolic_eq_line)
            
            # --- 4. Format Matrix Counterpart Equation ---
            x_beta_u = group.root_subgroup_map(beta, u)
            if d_gamma > 1:
                assert len(phi_u) == d_gamma
            RHS_matrix = group.root_subgroup_map(gamma, phi_u)
            
            w_alpha_latex = f"{{{sp.latex(w_alpha)}}}"
            x_beta_u_latex = f"{{{sp.latex(x_beta_u)}}}"
            w_alpha_inv_latex = f"{{{sp.latex(w_alpha_inverse)}}}"
            RHS_matrix_latex = f"{{{sp.latex(RHS_matrix)}}}"
            
            matrix_eq_line = (
                "\\[\n"
                "\\scaletoalign{\\textwidth}{"
                f"{w_alpha_latex} {x_beta_u_latex} {w_alpha_inv_latex} = {RHS_matrix_latex}"
                "}\n"
                "\\]"
            )
            equations.append(matrix_eq_line)
            
    if equations:
        macro_def = (
            "\\providecommand{\\scaletoalign}[2]{%\n"
            "  \\sbox0{$#2$}% \n"
            "  \\ifdim\\wd0>#1\\relax \n"
            "    \\resizebox{#1}{!}{\\usebox0}% \n"
            "  \\else \n"
            "    \\usebox0% \n"
            "  \\fi \n"
            "}\n"
        )
        tex = macro_def + "\n".join(equations) + "\n"
    else:
        tex = ""
        
    return tex