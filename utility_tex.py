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
    #       a. (COMPLETE) Summary table - name string, matrix size, rank, defining equations
    #       b. (COMPLETE) Bilinear form info (if there is one) - summary table (dimension, Witt index, primitive elemtn, epsilon), matrix
    #       c. (COMPLETE) Lie algebra info - defining equations, generic Lie algebra element
    #       d. (COMPLETE) Torus info - generic torus element, trivial characters
    # 2. Root system
    #       a. (COMPLETE) Summary table - Dynkin type, irreducible?, reduced?, simply laced?, number of roots
    #       b. (COMPLETE) Dynkin diagram
    #       c. (COMPLETE) Cartan matrix
    #       d. (COMPLETE) Table of roots with info on simple, positive/negative, multipliable
    #       e. (COMPLETE) List of root, coroot pairs
    #       f. (COMPLETE) Table of linear combinations of roots
    # 3. Root spaces
    #       a. (COMPLETE) Table with root, root space dimension
    #       b. (COMPLETE) Table with root, generic root space element
    # 4. Root subgroups
    #       a. (COMPLETE) Table with root, generic root subgroup element
    #       b. (COMPLETE) Table with root, homomorphism defect coefficient
    #       c. Explicit equations for homomorphism/pseudo-homomorphism property
    # 5. Commutators
    #       a. (COMPLETE) Table of linear combinations of roots
    #       b. (COMPLETE) Table with root1, root2, commutator coefficient
    #       c. Explicit equations for commutator relation
    # 6. Weyl group
    #       a. (COMPLETE) Table of root, Weyl group element
    #       b. (TO DO: FIX WHEN IT DOESN'T FIT ON ONE PAGE) Table of root1, root2, Weyl group conjugation coefficient
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
    if Phi.is_reduced:
        replacements['rootsystemreducedfalse'] = 'rootsystemreducedtrue'
    else:
        replacements.update({
            "rootsystemreducedtrue" : 'rootsystemreducedfalse',
            "HomomorphismDefectTablePlaceholder": build_hom_defect_table(G)
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
    
    # Root spaces, subgroups, Weyl group
    replacements.update({
        "RootSpaceDimensionPlaceholder": build_dimension_table(G),
        "RootSpaceTablePlaceholder" : build_root_space_table(G),
        "RootSubgroupTablePlaceholder": build_root_subgroup_table(G),
        "WeylGroupPlaceholder" : build_weyl_element_table(G),
        "WeylConjugationCoefficientPlaceholder" : build_weyl_conjugation_table(G)
    })
    
    # Commutators
    if len(Phi.summable_non_proportional_pairs) > 0:
        replacements.update({
            "ifsummablepairsfalse" : 'ifsummablepairstrue',
            "LinearCombinationsPlaceholder": build_root_linear_combos_table(Phi),
            "CommutatorCoefficientTablePlaceholder" : build_commutator_coefficient_table(G)
        })
    else:
        replacements.update({
            "ifsummablepairstrue" : 'ifsummablepairsfalse',
            "LinearCombinationsPlaceholder": "",
            "CommutatorCoefficientTablePlaceholder" : ""
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
    """
    Builds a pure TikZ-based LaTeX diagram for a single irreducible root system component,
    completely eliminating any dependency on external diagram packages.
    """
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
        # Standard configuration: long roots are open nodes, short root is last node
        for i in range(1, rank + 1):
            tikz_lines.append(f"  \\node[node] (v{i}) at ({(i-1)*step}, 0) {{}};")
        for i in range(1, rank - 1):
            tikz_lines.append(f"  \\draw[thick] (v{i}) -- (v{i+1});")
        # Double bond between the last two nodes
        if rank >= 2:
            tikz_lines.append(f"  \\draw[thick, double, double distance=2pt] (v{rank-1}) -- (v{rank});")
            # Draw a simple direction arrow in the middle of the double bond pointing right to left or left to right
            # For standard B_n, arrow points towards the short root (v_n)
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
            # For standard C_n, arrow points towards the long root (v_{n-1})
            mid_x = (rank - 1.5) * step
            tikz_lines.append(f"  \\draw[thick] ({mid_x - 0.05}, 0.1) -- ({mid_x + 0.05}, 0) -- ({mid_x - 0.05}, -0.1);")

    elif family == 'D':
        # Path graph with a fork on the right side
        # Main spine up to node rank - 2
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
            alpha_lat = sp.latex(alpha)
            beta_lat = sp.latex(beta)
            sorted_pairs = sorted(combos.keys())
            pairs_str = ", ".join(f"({i},{j})" for (i, j) in sorted_pairs)
            row_text = f"    ${alpha_lat}$ & ${beta_lat}$ & ${pairs_str}$ \\\\\n"
            
            pair_rows.append((len(sorted_pairs), row_text))

    # Sort rows by the number of linear combinations
    pair_rows.sort(key=lambda x: x[0])

    # Construct the LaTeX table
    table_lines = [
        r"\begin{tabular}{|l|l|c|}",
        r"    \hline",
        r"    $\alpha$ & $\beta$ & Pairs $(i,j)$ \\",
        r"    \hline"
    ]
    
    for _, row in pair_rows:
        table_lines.append(row.rstrip())
        table_lines.append(r"    \hline")
        
    table_lines.append(r"\end{tabular}" + "\n")
    
    return "\n".join(table_lines)

def build_commutator_coefficient_table(group):
    Phi = group.root_system
    assert len(group.commutator_coefficient_dict) > 0

    # We will gather blocks of rows grouped by (alpha, beta) so we can sort by total number of combinations
    table_blocks = []

    for (alpha, beta) in Phi.summable_non_proportional_pairs:
        d_alpha = group.root_space_dimension(alpha)
        d_beta = group.root_space_dimension(beta)
        u = vector_variable(letter='u', length=d_alpha)
        v = vector_variable(letter='v', length=d_beta)
        
        linear_combos = Phi.integer_linear_combos(alpha, beta)
        if not linear_combos:
            continue
            
        alpha_lat = sp.latex(alpha)
        beta_lat = sp.latex(beta)
        
        block_rows = []
        # Sort keys to ensure deterministic table ordering per pair
        for key in sorted(linear_combos.keys()):
            i, j = key[0], key[1]
            combo = linear_combos[key]
            
            coeff = group.commutator_coefficient_map(alpha, beta, i, j, u, v)
            raw_expr = coeff[0] if len(coeff) == 1 else coeff.T
            
            combo_lat = sp.latex(combo)
            expr_lat = sp.latex(raw_expr)
            
            # Format single row for this specific coefficient entry
            row_text = (
                f"    ${alpha_lat}$ & ${beta_lat}$ & {i} & {j} & "
                f"${combo_lat}$ & ${expr_lat}$ \\\\\n"
            )
            block_rows.append(row_text)
            
        if block_rows:
            # Store the count of entries for this root pair, and the accumulated rows
            table_blocks.append((len(block_rows), block_rows))

    # Sort blocks by the quantity of entries (index 0 of the tuple)
    table_blocks.sort(key=lambda x: x[0])

    combo_table = ""
    if table_blocks:
        # 6 columns matching the elements extracted per row
        combo_table += r"\begin{tabular}{|l|l|c|c|l|c|}" + "\n"
        combo_table += r"    \hline" + "\n"
        combo_table += (
            "    $\\alpha$ & $\\beta$ & $i$ & $j$ & "
            " $i\\alpha + j\\beta$ & $N_{ij}^{\\alpha\\beta}(u,v)$ \\\\\n"
        )
        combo_table += r"    \hline" + "\n"
        
        # Unpack the sorted blocks and write rows sequentially
        for _, rows in table_blocks:
            for row in rows:
                combo_table += row
                combo_table += r"    \hline" + "\n"
                
        combo_table += "\\end{tabular}\n"
    else:
        combo_table += "\\noindent No pairs of roots generate positive integer linear combinations.\n"
        
    return combo_table

def build_dimension_table(group):
    Phi = group.root_system
    tex = r"\noindent \begin{tabular}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    \textbf{Root ($\alpha$)} & \textbf{Dimension of root space ($d_{\alpha}$)} \\" + "\n"
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
    Phi = group.root_system
    assert(not Phi.is_reduced)
    tex = r"\noindent \begin{tabular}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Homomorphism defect coefficient} \\\\\n"
    tex += r"    \hline" + "\n"
    for alpha in Phi.root_list:
        if Phi.is_multipliable_root(alpha):
            hdc = group.homomorphism_defect_coefficient_dict[alpha][2]
            alpha_latex = f"{sp.latex(alpha)}"
            hdc_latex = f"{{{sp.latex(hdc)}}}" 
            tex += f"    ${alpha_latex}$ & ${hdc_latex}$ \\\\\n"
            tex += r"    \hline" + "\n"
    tex += "\\end{tabular}\n"
    return tex

def build_weyl_element_table(group):
    Phi = group.root_system
    tex = r"\noindent \begin{tabular}{|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += "    \\textbf{Root} & \\textbf{Weyl group element} \\\\\n"
    tex += r"    \hline" + "\n"
    for alpha in Phi.root_list:
        w_alpha = group.weyl_element_map(alpha)
        alpha_latex = f"{sp.latex(alpha)}"
        w_alpha_latex = f"{{{sp.latex(w_alpha)}}}"
        tex += f"    ${alpha_latex}$ & ${w_alpha_latex}$ \\\\\n"
        tex += r"    \hline" + "\n"
    tex += "\\end{tabular}\n"
    return tex

def build_weyl_conjugation_table(self):
    Phi = self.root_system
    
    # 1. Setup the LaTeX table alignment and headers
    tex = r"\noindent \begin{tabular}{|l|l|l|c|}" + "\n"
    tex += r"    \hline" + "\n"
    tex += r"    \textbf{Root ($\alpha$)} & \textbf{Root ($\beta$)} & \textbf{$\gamma = \sigma_{\alpha}(\beta)$} & \textbf{Weyl conjugation coefficient} \\" + "\n"
    tex += r"    \hline" + "\n"
    
    # 2. Iterate through every possible pair of roots
    for alpha in Phi.root_list:
        for beta in Phi.root_list:
            d_beta = self.root_space_dimension(beta)
            u = vector_variable(letter='u', length=d_beta)
            
            gamma = self.root_system.reflect_root(
                hyperplane_root=alpha,
                root_to_reflect=beta
            )
            
            d_gamma = self.root_space_dimension(gamma)
            assert d_beta == d_gamma
            
            phi_u = self.weyl_conjugation_coefficient_map(alpha, beta, u)
            raw_expr = phi_u[0] if len(phi_u) == 1 else phi_u.T
            
            # 3. Convert all elements to LaTeX strings
            alpha_latex = f"${sp.latex(alpha)}$"
            beta_latex = f"${sp.latex(beta)}$"
            gamma_latex = f"${sp.latex(gamma)}$"
            expr_latex = f"${sp.latex(raw_expr)}$"
            
            # 4. Append row to the table
            tex += f"    {alpha_latex} & {beta_latex} & {gamma_latex} & {expr_latex} \\\\\n"
            tex += r"    \hline" + "\n"
            
    tex += "\\end{tabular}\n"
    return tex
