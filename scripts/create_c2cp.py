import argparse
import json
import random
import numpy as np
from itertools import permutations, combinations, chain
from collections import defaultdict

random.seed(0)
np.random.seed(0)

# Separate RNG for letter generation to avoid affecting main sampling logic
letter_rng = random.Random(42)

RELATION_TYPES = ["parent", "ancestor", "child", "descendant", "collider", "confounder"]

# Hypothesis templates for each relation type
HYPOTHESIS_TEMPLATES = {
    "parent": "{node_i} directly affects {node_j}.",
    "ancestor": "{node_i} influences {node_j} through some mediator(s) but not directly.",
    "child": "{node_i} is directly affected by {node_j}.",
    "descendant": "{node_j} influences {node_i} through some mediator(s) but not directly.",
    #"collider": "{node_i} and {node_j} together cause some other variable(s).",
    "collider": " There exists at least one collider (i.e., a common child) of {node_i} and {node_j}.",
    #"confounder": "Some variable(s) cause(s) both {node_i} and {node_j}.",
    "confounder": "There exists at least one confounder (i.e., a common parent) of {node_i} and {node_j}.",
     
}

class Dag:
    def __init__(self, idx, edges, ci_relations, reconstruct_graphs, mec_indices):
        self.idx = idx
        self.edges = edges
        self.ci_relations = ci_relations
        self.reconstruct_graphs = reconstruct_graphs
        self.mec_indices = mec_indices
        self.mec_indices_ref = set()

class Mec:
    def __init__(self, dag_indices, ci_relations):
        self.dag_indices=dag_indices
        self.ci_relations = ci_relations
 



def powerset(iterable):
    """Generate all subsets of an iterable."""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

def find_descendants(adj, node):
    """Find all descendants of a node in a directed graph."""
    descendants = {node}
    stack = [node]
    while stack:
        curr = stack.pop()
        for child in adj.get(curr, ()):
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return descendants

def find_all_paths(adj, x, y, n):
    """Find all paths between x and y, returning (nodes_on_path, collider_nodes) pairs."""
    paths = []

    def dfs(curr, visited, colliders, was_incoming):
        if curr == y:
            paths.append((visited.copy(), colliders.copy()))
            return
        for neighbor in range(1, n + 1):
            if neighbor in visited:
                continue
            has_out = neighbor in adj.get(curr, set())
            has_in = curr in adj.get(neighbor, set())
            if not (has_out or has_in):
                continue
            visited.add(neighbor)
            if has_out:  # curr -> neighbor
                dfs(neighbor, visited, colliders, True)
            else:  # curr <- neighbor
                if was_incoming:
                    colliders.add(curr)
                dfs(neighbor, visited, colliders, False)
                if was_incoming:
                    colliders.discard(curr)
            visited.discard(neighbor)

    dfs(x, {x}, set(), False)
    return paths

def is_d_separated(paths, descendants, cond_set, x, y):
    """Check if x and y are d-separated given conditioning set."""
    if x in cond_set or y in cond_set:
        return False

    for path_nodes, colliders in paths:
        # Path is blocked if a non-collider is conditioned on
        if cond_set & (path_nodes - colliders):
            continue
        # Path is blocked if any collider (and descendants) is NOT conditioned on
        blocked = any(not (cond_set & descendants[c]) for c in colliders)
        if not blocked:
            return False
    return True

def compute_ci_relations(edges, n):
    adj = {}
    for i, j in edges:
        if i not in adj:
            adj[i] = set()
        adj[i].add(j)
    descendants = {i: find_descendants(adj, i) for i in range(1, n + 1)}
    ci_relations = []
    nodes = list(range(1, n + 1))
    for x in range(1, n):
        for y in range(x + 1, n + 1):
            paths = find_all_paths(adj, x, y, n)
            for cond_tuple in powerset(nodes):
                if is_d_separated(paths, descendants, set(cond_tuple), x, y):
                    ci_relations.append([[x, y], list(cond_tuple)])
    return ci_relations


def relabel_edges(edges, perm):
    """Relabel edges according to a permutation."""
    return frozenset((perm[i], perm[j]) for i, j in edges)


def generate_all_dags(n, m=None):
    """Generate all DAGs with n nodes and compute their properties.

    If m is specified, randomly sample m DAGs instead of generating all.
    """
    all_pairs = [(i, j) for i in range(1, n) for j in range(i + 1, n + 1)]
    all_orders = [[0] + list(p) for p in permutations(range(1, n + 1))]
    dags = []
    shuffle_dags = []
    graph_to_shuffle = {}

    # Generate all possible edge sets
    all_edge_sets = list(powerset(all_pairs))

    # Sample m edge sets if specified and m < total
    if m is not None and m < len(all_edge_sets):
        all_edge_sets = random.sample(all_edge_sets, m)

    # Generate DAGs from edge sets
    for edges in all_edge_sets:
        dag = Dag (idx=len(dags)+1, edges= list(edges), ci_relations = compute_ci_relations(edges, n),
                  reconstruct_graphs = [], mec_indices = [] )
        dags.append(dag)

    # Generate all permutations
    for dag in dags:
        edges = tuple(tuple(e) for e in dag.edges)
        for order in all_orders:
            new_edges = frozenset((order[i], order[j]) for i, j in edges)
            if new_edges not in graph_to_shuffle:
                graph_to_shuffle[new_edges] = len(shuffle_dags)
                shuffle_dags.append((dag.idx, order))
                dag.reconstruct_graphs.append(new_edges)
    
    for dag in dags:
        dag.ci_relations.sort()

    for dag in dags:
        for i, (ref_idx, order) in enumerate(shuffle_dags):
            ref_dag = dags[ref_idx - 1]
            if len(ref_dag.ci_relations) != len(dag.ci_relations):
                continue
            reprojected = sorted(
                [sorted([order[j] for j in a]), sorted([order[j] for j in b])]
                for a, b in ref_dag.ci_relations)
            if reprojected == dag.ci_relations:
                dag.mec_indices.append(i)
                dag.mec_indices_ref.add(ref_idx)
    
    return dags, shuffle_dags, graph_to_shuffle
    

def build_mec_list(dags, n):
    """Build list of MECs from DAGs.

    Uses PC algorithm to compute forced edges from CI relations,
    rather than intersection of sampled DAGs.

    DAGs are grouped into the same MEC if they have the same mec_indices_ref,
    which accounts for permutation equivalence.
    """
    seen_id = set()
    mec_list = []
    variables = list(range(1, n + 1))

    for dag in dags:
        # Use mec_indices_ref to group DAGs by MEC (accounts for permutation equivalence)
        mec_id = frozenset(dag.mec_indices_ref)
        if mec_id in seen_id:
            continue
        seen_id.add(mec_id)

        # Use PC algorithm to get CPDAG and forced edges
        cpdag = get_cpdag(variables, dag.ci_relations)
        forced_edges = sorted(list(cpdag['directed']))

        mec = {
            'ci_relations': dag.ci_relations,
            'forced_edges': forced_edges,
        }
        mec_list.append(mec)

    return mec_list

def compute_relation_types(edges, n):
    """Compute relation types between all ordered pairs of nodes.

    Returns a dict: (i, j) -> set of relation types that hold for this ordered pair.
    Relation types: parent, ancestor, child, descendant, collider, confounder
    """
    # Build adjacency structures
    adj = defaultdict(set)  # adj[i] = set of children of i
    for i, j in edges:
        adj[i].add(j)

    # Compute descendants and ancestors for each node
    descendants = {}
    ancestors = {}
    for node in range(1, n + 1):
        # Find descendants
        desc = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            for child in adj[curr]:
                if child not in desc:
                    desc.add(child)
                    stack.append(child)
        descendants[node] = desc

        # Find ancestors
        anc = set()
        stack = [node]
        while stack:
            curr = stack.pop()
            for parent in range(1, n + 1):
                if curr in adj[parent] and parent not in anc:
                    anc.add(parent)
                    stack.append(parent)
        ancestors[node] = anc

    # Find children of each node (direct)
    children = {node: adj[node] for node in range(1, n + 1)}
    # Find parents of each node (direct)
    parents = defaultdict(set)
    for i, j in edges:
        parents[j].add(i)

    relations = {}
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            rels = set()

            # parent: i -> j directly
            if j in children[i]:
                rels.add("parent")

            # ancestor: i is ancestor of j but not parent (i.e., path through mediator)
            if j in descendants[i] and j not in children[i]:
                rels.add("ancestor")

            # child: j -> i directly
            if i in children[j]:
                rels.add("child")

            # descendant: j is ancestor of i but not parent
            if i in descendants[j] and i not in children[j]:
                rels.add("descendant")

            # collider: there exists a node k such that i -> k and j -> k (direct children)
            for k in range(1, n + 1):
                if k != i and k != j:
                    if k in children[i] and k in children[j]:
                        rels.add("collider")
                        break

            # confounder: there exists a node k such that k -> i and k -> j (direct children)
            for k in range(1, n + 1):
                if k != i and k != j:
                    if i in children[k] and j in children[k]:
                        rels.add("confounder")
                        break

            relations[(i, j)] = rels

    return relations


def compute_mec_relations(forced_edges, n):
    """Compute which relations are implied by the forced edges of a MEC.

    Returns a dict: (i, j) -> set of relation types that hold for this ordered pair.
    A relation either holds (implied by forced edges) or doesn't.
    """
    return compute_relation_types(forced_edges, n)


def node_to_name(i, letter_map=None):
    """Convert node index to variable name.

    If letter_map is provided, use it. Otherwise default to Z, Y, X, ...
    """
    if letter_map:
        return letter_map[i]
    return chr(91 - i)


def generate_random_letters(n):
    """Generate a random mapping of node indices to distinct capital letters from A-Z."""
    letters = letter_rng.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', n)
    return {i: letters[i - 1] for i in range(1, n + 1)}


def generate_default_letters_permuted(n):
    """Generate a random permutation of the default letters Z, Y, X, ..."""
    # Default letters are Z, Y, X, W, V, ... (last n letters of alphabet in reverse)
    default_letters = [chr(91 - i) for i in range(1, n + 1)]  # Z, Y, X, ...
    permuted = letter_rng.sample(default_letters, n)
    return {i: permuted[i - 1] for i in range(1, n + 1)}

def list_to_text(items):
    """Convert list to natural language (e.g., ['A', 'B', 'C'] -> 'A, B, and C')."""
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]

def build_premise(ci_relations, n, letter_map=None):
    """Build the premise text from CI relations using mathematical notation."""
    # Build mapping from node pairs to their conditioning sets
    pair_to_conds = defaultdict(list)
    for pair, cond in ci_relations:
        pair_to_conds[tuple(pair)].append(cond)

    # Sort by conditioning set size
    pair_to_conds = {k: sorted(v, key=len) for k, v in pair_to_conds.items()}

    # Build premise
    var_names = [node_to_name(i, letter_map) for i in range(1, n + 1)]
    premise = f"Suppose there is a closed system of {n} variables, {list_to_text(var_names)}, whose causal structure forms a directed acyclic graph and satisfies faithfulness. "
    premise += f"All the conditional independence relations are: "

    # Build CI statements in mathematical notation: X indep Y | Z
    ci_statements = []
    for pair, conds in sorted(pair_to_conds.items()):
        i, j = pair
        ni, nj = node_to_name(i, letter_map), node_to_name(j, letter_map)
        for cond in conds:
            if cond:
                cond_names = ", ".join(node_to_name(c, letter_map) for c in cond)
                ci_statements.append(f"{ni} indep {nj} | {cond_names}")
            else:
                ci_statements.append(f"{ni} indep {nj}")

    premise += "; ".join(ci_statements) + "."

    return premise


def verbalize_mec(mec, n, random_letters=False):
    """Convert a MEC to NLI examples.

    If random_letters is True, each example gets a fresh random letter mapping from A-Z.
    If False, uses the default Z, Y, X, ... pattern but with a random permutation for each example.
    For each relation type, generates one positive (label=1) and one negative (label=0) example.
    """
    ci_relations = mec['ci_relations']
    forced_edges = mec['forced_edges']

    # Compute relations implied by forced edges
    mec_relations = compute_mec_relations(forced_edges, n)

    # Generate examples for each relation type
    nli_data = []

    for rel_type in RELATION_TYPES:
        # Find pairs where this relation holds and doesn't hold
        positive_pairs = []
        negative_pairs = []

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i == j:
                    continue
                rels = mec_relations.get((i, j), set())
                if rel_type in rels:
                    positive_pairs.append((i, j))
                else:
                    negative_pairs.append((i, j))

        # Pick one positive and one negative example (if available)
        if positive_pairs:
            i, j = random.choice(positive_pairs)
            # Generate fresh letter map for each example
            # If random_letters: pick random letters from A-Z
            # Else: use Z,Y,X,... but with a random permutation for diversity
            letter_map = generate_random_letters(n) if random_letters else generate_default_letters_permuted(n)
            premise = build_premise(ci_relations, n, letter_map)
            forced_edges_names = [[node_to_name(fi, letter_map), node_to_name(fj, letter_map)] for fi, fj in forced_edges]
            ni, nj = node_to_name(i, letter_map), node_to_name(j, letter_map)
            hypothesis = HYPOTHESIS_TEMPLATES[rel_type].format(node_i=ni, node_j=nj)
            nli_data.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': 1,
                'relation_type': rel_type,
                'forced_edges': forced_edges_names,
            })

        if negative_pairs:
            i, j = random.choice(negative_pairs)
            # Generate fresh letter map for each example
            letter_map = generate_random_letters(n) if random_letters else generate_default_letters_permuted(n)
            premise = build_premise(ci_relations, n, letter_map)
            forced_edges_names = [[node_to_name(fi, letter_map), node_to_name(fj, letter_map)] for fi, fj in forced_edges]
            ni, nj = node_to_name(i, letter_map), node_to_name(j, letter_map)
            hypothesis = HYPOTHESIS_TEMPLATES[rel_type].format(node_i=ni, node_j=nj)
            nli_data.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': 0,
                'relation_type': rel_type,
                'forced_edges': forced_edges_names,
            })

    return nli_data

def generate_nli_dataset(n, m=None, random_letters=False):
    dags, _, _ = generate_all_dags(n, m)
    print("number of dags: ", len(dags))
    mec_list = build_mec_list(dags, n)
    print("number of mecs: ", len(mec_list))

    nli_data = []
    for mec in mec_list:
        nli_data.extend(verbalize_mec(mec, n, random_letters=random_letters))

    # Shuffle at the end for random ordering in output
    random.shuffle(nli_data)

    return nli_data


def parse_node_range(node_arg):
    """Parse node argument which can be a single number or a range like '4-6'."""
    if '-' in node_arg:
        start, end = node_arg.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(node_arg)]


def parse_m_values(m_arg, num_nodes):
    """Parse -m argument which can be a single number or comma-separated values.

    Returns a list of m values, one for each n value.
    If a single value is given, it's used for all n values.
    Use 0 for 'all DAGs' (will be converted to None).
    """
    if ',' in m_arg:
        values = [int(x.strip()) for x in m_arg.split(',')]
        if len(values) != num_nodes:
            raise ValueError(f"Number of -m values ({len(values)}) must match number of n values ({num_nodes})")
        return [v if v > 0 else None for v in values]
    else:
        v = int(m_arg)
        return [v if v > 0 else None] * num_nodes


def build_skeleton(variables, ci_relations):
    """
    Build skeleton: edge exists between A and B iff
    there is NO independence statement for (A, B) given any conditioning set.

    ci_relations is a list of [[x, y], cond_list] pairs.
    """
    # Build set of pairs that have CI relations
    ci_pairs = set()
    for pair, _ in ci_relations:
        ci_pairs.add(frozenset(pair))

    skeleton = set()
    for a, b in combinations(variables, 2):
        pair = frozenset({a, b})
        # Edge exists if there's NO CI relation for this pair
        if pair not in ci_pairs:
            skeleton.add(pair)
    return skeleton


def find_separating_set(a, b, ci_relations):
    """Find the conditioning set that makes a and b independent.

    ci_relations is a list of [[x, y], cond_list] pairs.
    Returns the conditioning set (as a set), or None if not found.
    """
    target = frozenset({a, b})
    for pair, cond in ci_relations:
        if frozenset(pair) == target:
            return set(cond)
    return None


def orient_v_structures(variables, skeleton, ci_relations):
    """
    Orient v-structures (colliders).
    For unshielded triple X - Z - Y (X,Y not adjacent),
    if X indep Y | S where Z not in S, then X -> Z <- Y
    """
    directed = set()  # (cause, effect) tuples

    for z in variables:
        neighbors = [v for v in variables if frozenset({v, z}) in skeleton]

        for x, y in combinations(neighbors, 2):
            # Check if x and y are NOT adjacent (unshielded triple)
            if frozenset({x, y}) in skeleton:
                continue

            # Find separating set for x and y
            sep_set = find_separating_set(x, y, ci_relations)

            if sep_set is not None and z not in sep_set:
                # V-structure: x -> z <- y
                directed.add((x, z))
                directed.add((y, z))

    return directed


def apply_meek_rules(variables, skeleton, directed):
    """Apply Meek's rules to propagate orientations."""
    undirected = set(skeleton)
    directed = set(directed)

    # Remove edges that are already directed
    for cause, effect in directed:
        undirected.discard(frozenset({cause, effect}))

    changed = True
    while changed:
        changed = False

        # Rule 1: If X -> Y - Z and X, Z not adjacent, then Y -> Z
        for y in variables:
            for x in variables:
                if (x, y) in directed:
                    for z in variables:
                        if z != x and z != y:
                            if frozenset({y, z}) in undirected:
                                if frozenset({x, z}) not in skeleton:
                                    directed.add((y, z))
                                    undirected.discard(frozenset({y, z}))
                                    changed = True

        # Rule 2: If X -> Y -> Z and X - Z, then X -> Z
        for y in variables:
            for x in variables:
                if (x, y) in directed:
                    for z in variables:
                        if (y, z) in directed and z != x:
                            if frozenset({x, z}) in undirected:
                                directed.add((x, z))
                                undirected.discard(frozenset({x, z}))
                                changed = True

        # Rule 3: If X - Y, X - Z, Y -> W, Z -> W, and Y,Z not adjacent, then X -> W
        for x in variables:
            x_neighbors = [v for v in variables if frozenset({x, v}) in undirected]
            for y, z in combinations(x_neighbors, 2):
                if frozenset({y, z}) not in skeleton:  # Y, Z not adjacent
                    for w in variables:
                        if w != x and (y, w) in directed and (z, w) in directed:
                            if frozenset({x, w}) in undirected:
                                directed.add((x, w))
                                undirected.discard(frozenset({x, w}))
                                changed = True

        # Rule 4: If D - A - C, D -> C -> B, and B not adjacent to D, then A -> B
        for a in variables:
            a_neighbors = [v for v in variables if frozenset({a, v}) in undirected]
            for d in a_neighbors:
                for c in variables:
                    if c != d and frozenset({a, c}) in undirected:  # A - C
                        if (d, c) in directed:  # D -> C
                            for b in variables:
                                if b != a and b != d and (c, b) in directed:  # C -> B
                                    if frozenset({b, d}) not in skeleton:  # B not adjacent to D
                                        if frozenset({a, b}) in undirected:
                                            directed.add((a, b))
                                            undirected.discard(frozenset({a, b}))
                                            changed = True

    return directed, undirected


def get_cpdag(variables, ci_relations):
    """Get the CPDAG from CI statements.

    Returns dict with:
    - skeleton: set of frozensets (undirected edges in skeleton)
    - directed: set of (cause, effect) tuples (oriented edges)
    - undirected: set of frozensets (edges that remain undirected)
    """
    skeleton = build_skeleton(variables, ci_relations)
    directed = orient_v_structures(variables, skeleton, ci_relations)
    directed, undirected = apply_meek_rules(variables, skeleton, directed)
    return {
        'skeleton': skeleton,
        'directed': directed,
        'undirected': undirected
    }


def main():
    parser = argparse.ArgumentParser(description='Generate causal NLI dataset')
    parser.add_argument('-n', type=str, default='6', help='Number of nodes (e.g., 5 or 4-6 for a range)')
    parser.add_argument('-m', type=str, default='5000', help='Max number of DAGs to sample per n (0 for all). Can be comma-separated for different values per n (e.g., "60,100,200" for n=4-6)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file')
    parser.add_argument('-r', '--random-letters', action='store_true',
                        help='Use randomly selected capital letters instead of Z,Y,X,...')
    args = parser.parse_args()


    # Parse node range
    node_counts = parse_node_range(args.n)

    # Parse m values (one per n, or single value for all)
    m_values = parse_m_values(args.m, len(node_counts))

    # Generate data for each node count
    all_nli_data = []
    for n, m in zip(node_counts, m_values):
        print(f"\n=== Generating for n={n}, m={m if m else 'all'} ===")
        nli_data = generate_nli_dataset(n, m, random_letters=args.random_letters)
        all_nli_data.extend(nli_data)

    # Shuffle all data together
    random.shuffle(all_nli_data)

    # Determine output filename
    if args.output:
        output_file = args.output
    elif len(node_counts) == 1:
        output_file = f'causal_nli_n={node_counts[0]}.json'
    else:
        output_file = f'causal_nli_n={node_counts[0]}-{node_counts[-1]}.json'

    # Generate JSON with indentation
    import re
    json_str = json.dumps(all_nli_data, indent=2)

    # Compact the forced_edges arrays to single lines
    json_str = re.sub(
        r'"forced_edges": \[\s+(\[[\s\S]*?\])\s+\]',
        lambda m: '"forced_edges": [' + re.sub(r'\s+', ' ', m.group(1)) + ']',
        json_str
    )

    with open(output_file, 'w') as f:
        f.write(json_str)
    print(f"\nWritten {len(all_nli_data)} NLI examples to {output_file}")


if __name__ == "__main__":
    main()