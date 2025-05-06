import os
import numpy as np
from gqlalchemy import Memgraph
import multiprocessing
from multiprocessing import Pool, cpu_count
import math
import random
import json
import logging
import time 
import argparse

# ----------------------
# CLI Arguments
# ----------------------
parser = argparse.ArgumentParser(description="MCTS Search from Start to Stop Protein(s)")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--start_protein", help="Single starting protein symbol (unprefixed)")
group.add_argument("--start_proteins", nargs='+', help="List of starting protein symbols (unprefixed)")
parser.add_argument("--stop_protein", required=True, help="Target protein symbol (unprefixed)")
parser.add_argument("--desired_length", type=int, required=True, help="Desired pathway length to reward closeness")
parser.add_argument("--iterations", type=int, required=True, help="Total MCTS iterations to distribute across workers")
parser.add_argument("--chain_length", type=int, required=True, help="Maximum length of found protein chains")
parser.add_argument("--output", required=True, help="Directory to output results")
args = parser.parse_args()

# Determine list of starting proteins
SEARCH_PROTEINS = args.start_proteins if args.start_proteins else [args.start_protein]
if args.stop_protein in SEARCH_PROTEINS:
    raise ValueError("stop_protein must not be one of the start proteins")

# ----------------------
# Constants and Parameters
# ----------------------
RESULTS_DIR = args.output
PROTEIN_ID_PREFIX = "HGNC:"
MEMGRAPH_HOST = '127.0.0.1'
MEMGRAPH_PORT = 7687

MCTS_ITERATIONS = args.iterations
MCTS_EXPLORATION_FACTOR_C = 1.414
MCTS_SIMULATION_DEPTH = 15
MCTS_MAX_PATH_LENGTH = args.chain_length
STOP_PROTEIN = args.stop_protein
DESIRED_LENGTH = args.desired_length

REWARD_CORRELATION_WEIGHT = 1.0
REWARD_LENGTH_WEIGHT = 0.7
REWARD_LOOP_BONUS = -90
REWARD_GOAL_BONUS = 100
REWARD_LENGTH_CLOSENESS = 0.01

NUM_PARALLEL_WORKERS = max(1, cpu_count() - 1)

# ----------------------
# Logging Setup
# ----------------------
def setup_logger():
    """Configure and return a logger with file and console handlers."""
    logger = logging.getLogger('mcts_search')
    logger.setLevel(logging.DEBUG)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler with DEBUG level
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(RESULTS_DIR, 'mcts_search.log'))
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create logger instance
logger = setup_logger()

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------
# Preload shared data once
# ----------------------
CORR_MATRIX_FILE = os.path.join(RESULTS_DIR, 'depmap_correlation_matrix.npy')
GENE_LIST_FILE = os.path.join(RESULTS_DIR, 'gene_list_for_corr_matrix.txt')

CORR_MATRIX = np.load(CORR_MATRIX_FILE)
with open(GENE_LIST_FILE) as gf:
    _genes = [line.strip() for line in gf]
GENE_TO_INDEX = {g: i for i, g in enumerate(_genes)}

# Global DB connection placeholder
DB_CONN = None

def init_worker(corr_matrix, gene_map):
    global CORR_MATRIX, GENE_TO_INDEX, DB_CONN
    CORR_MATRIX = corr_matrix
    GENE_TO_INDEX = gene_map
    # Set up per-worker logger
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    
    # Initialize persistent Memgraph connection once per worker
    #worker_logger.debug("Initializing database connection for worker")
    start_time = time.time()
    try:
        DB_CONN = Memgraph(host=MEMGRAPH_HOST, port=MEMGRAPH_PORT)
        DB_CONN.execute("RETURN 1")
        #worker_logger.debug(f"Database connection established in {time.time() - start_time:.3f} seconds")
    except Exception as e:
        #worker_logger.error(f"Failed to connect to database: {e}")
        raise

# ----------------------
# MCTS Node Definition
# ----------------------
class MCTSNode:
    def __init__(self, state, parent=None, path=None):
        self.state = state
        self.parent = parent
        self.path = path or [state]
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.is_terminal = len(self.path) >= MCTS_MAX_PATH_LENGTH or state == STOP_PROTEIN

    def select_child_uct(self, c):
        best_score, best_child = -float('inf'), None
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploit = child.value / child.visits
                explore = c * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score, best_child = score, child
        return best_child

    def add_child(self, node):
        self.children.append(node)

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

# ----------------------
# Graph, Reward and Merging
# ----------------------


#Test both undirected and directed MATCH; - or ->
def get_neighbors(protein_id):
    query = """
        MATCH (:Protein {id: $pid})-[:INTERACTS]->(nbr:Protein)
        RETURN nbr.id AS neighbor_id
    """
    params = {'pid': f"{PROTEIN_ID_PREFIX}{protein_id}"}
    try:
        results = DB_CONN.execute_and_fetch(query, parameters=params)
        return [rec['neighbor_id'][len(PROTEIN_ID_PREFIX):]
                for rec in results if rec['neighbor_id'].startswith(PROTEIN_ID_PREFIX)]
    except Exception as e:
        logging.error(f"Memgraph error for {protein_id}: {e}")
        return []


def calculate_reward(path, corr_matrix, gene_to_index):
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    #worker_logger.debug(f"Calculating reward for path of length {len(path)}")
    
    reward = 0.0
    valid = [g for g in path if g in gene_to_index]
    
    if len(valid) > 1:
        idx = [gene_to_index[g] for g in valid]
        sub = corr_matrix[np.ix_(idx, idx)]
        ut = np.triu_indices_from(sub, k=1)
        
        if ut[0].size:
            correlation_reward = np.mean(sub[ut]) * REWARD_CORRELATION_WEIGHT
            reward += correlation_reward
            #worker_logger.debug(f"Correlation reward: {correlation_reward:.4f}")
    
    length_reward = len(path) * REWARD_LENGTH_WEIGHT
    reward += length_reward
    #worker_logger.debug(f"Length reward: {length_reward:.4f}")
    
    if len(path) > 1 and path[-1] in path[:-1]:
        loop_reward = REWARD_LOOP_BONUS
        reward += loop_reward
        #worker_logger.debug(f"Loop reward: {loop_reward:.4f}")
    
    diff = abs(len(path) - DESIRED_LENGTH)
    closeness = max(0, DESIRED_LENGTH - diff)
    closeness_reward = closeness * REWARD_LENGTH_CLOSENESS
    reward += closeness_reward
    #worker_logger.debug(f"Length closeness reward: {closeness_reward:.4f}")
    
    if path and path[-1] == STOP_PROTEIN:
        goal_reward = REWARD_GOAL_BONUS
        reward += goal_reward
        #worker_logger.debug(f"Goal reached reward: {goal_reward:.4f}")
    
    #worker_logger.debug(f"Total reward: {reward:.4f}")
    return reward

def merge_duplicate_paths_streaming(results_iterator):
    """Memory-efficient version for very large result sets"""
    from collections import defaultdict
    import itertools
    
    # Sort results by path to group duplicates
    sorted_results = sorted(results_iterator, key=lambda x: str(x['path']))
    
    # Group by identical paths
    merged = []
    for path, group in itertools.groupby(sorted_results, key=lambda x: str(x['path'])):
        group_list = list(group)
        if len(group_list) == 1:
            merged.append(group_list[0])
        else:
            # Merge this group of duplicates
            base = group_list[0].copy()
            total_visits = sum(r['visits'] for r in group_list)
            weighted_value = sum(r['visits'] * r['avg_value'] for r in group_list) / total_visits
            
            base['visits'] = total_visits
            base['avg_value'] = weighted_value
            merged.append(base)
    
    return merged


# ----------------------
# MCTS Phases
# ----------------------

def select(node, c):
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    #worker_logger.debug(f"Selecting node from state {node.state}")
    
    while not node.is_terminal:
        if not node.is_fully_expanded():
            #worker_logger.debug(f"Node {node.state} not fully expanded, returning")
            return node
        
        node = node.select_child_uct(c)
        #worker_logger.debug(f"Selected child node {node.state}")
    
    #worker_logger.debug(f"Reached terminal node {node.state}")
    return node

def expand(node):
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    #worker_logger.debug(f"Expanding node {node.state}")
    
    if node.is_terminal:
        #worker_logger.debug(f"Node {node.state} is terminal, cannot expand")
        return node
    
    if node.untried_actions is None:
        #worker_logger.debug(f"Fetching neighbors for {node.state}")
        node.untried_actions = get_neighbors(node.state)
        
        if node.parent:
            if node.parent.state in node.untried_actions:
                #worker_logger.debug(f"Removing parent {node.parent.state} from untried actions")
                node.untried_actions.remove(node.parent.state)
        
        random.shuffle(node.untried_actions)
        #worker_logger.debug(f"Found {len(node.untried_actions)} untried actions for {node.state}")
    
    if node.untried_actions:
        nxt = node.untried_actions.pop()
        #worker_logger.debug(f"Selected untried action {nxt}")
        
        child = MCTSNode(nxt, node, node.path + [nxt])
        node.add_child(child)
        #worker_logger.debug(f"Created new child node {nxt}")
        return child
    
    node.is_terminal = True
    #worker_logger.debug(f"No untried actions left, marking node {node.state} as terminal")
    return node

def simulate(node, max_depth):
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    #worker_logger.debug(f"Starting simulation from node {node.state}")
    
    path = list(node.path)
    state = node.state
    steps = 0
    
    while len(path) < MCTS_MAX_PATH_LENGTH and steps < max_depth:
        nbrs = get_neighbors(state)
        
        if not nbrs:
            #worker_logger.debug(f"No neighbors found for {state}, ending simulation")
            break
            
        if len(path) > 1:
            prev = path[-2]
            nbrs = [n for n in nbrs if n != prev] or nbrs
            
        state = random.choice(nbrs)
        path.append(state)
        steps += 1
        #worker_logger.debug(f"Simulation step {steps}: moved to {state}")
        
        if state == STOP_PROTEIN:
            #worker_logger.debug(f"Reached goal state {STOP_PROTEIN}, ending simulation")
            break
    
    reward = calculate_reward(path, CORR_MATRIX, GENE_TO_INDEX)
    #worker_logger.debug(f"Simulation ended after {steps} steps with reward {reward:.4f}")
    return reward

def backpropagate(node, reward):
    #worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    #worker_logger.debug(f"Backpropagating reward {reward:.4f}")
    
    nodes_updated = 0
    while node:
        node.update(reward)
        nodes_updated += 1
        node = node.parent
    
    #worker_logger.debug(f"Updated {nodes_updated} nodes during backpropagation")

# ----------------------
# Worker
# ----------------------

def run_mcts_worker(args_tuple):
    start, iters = args_tuple
    worker_logger = logging.getLogger(f'mcts_search.worker.{multiprocessing.current_process().name}')
    worker_logger.info(f"Starting MCTS worker for protein {start} with {iters} iterations")
    
    worker_start_time = time.time()
    root = MCTSNode(start)
    
    for i in range(iters):
        iteration_start = time.time()
        #worker_logger.debug(f"Iteration {i+1}/{iters} for protein {start}")
        
        leaf = select(root, MCTS_EXPLORATION_FACTOR_C)
        child = expand(leaf)
        reward = simulate(child, MCTS_SIMULATION_DEPTH)
        backpropagate(child, reward)
        
        #worker_logger.debug(f"Iteration {i+1} completed in {time.time() - iteration_start:.3f} seconds")
    
    # Collect results - Fix for logical issue #1: collect ALL paths to the target
    results = []
    stack = [root]
    
    while stack:
        nd = stack.pop()
        # Remove the condition "not nd.children" to collect all paths to the target
        """ if nd.path and nd.path[-1] == STOP_PROTEIN:
            avg = nd.value / nd.visits if nd.visits else 0
            results.append({'start': start, 'path': nd.path, 'visits': nd.visits, 'avg_value': avg, 'length': len(nd.path)})
            #worker_logger.debug(f"Found valid path to target: length={len(nd.path)}, avg_value={avg:.4f}") """
        avg = nd.value / nd.visits if nd.visits else 0
        results.append({'start': start, 'path': nd.path, 'visits': nd.visits, 'avg_value': avg, 'length': len(nd.path)})
        stack.extend(nd.children)
    
    worker_logger.info(f"MCTS worker for protein {start} completed in {time.time() - worker_start_time:.2f} seconds. Found {len(results)} valid paths.")
    return results


# ----------------------
# Main
# ----------------------
if __name__ == '__main__':
    logger.info(f"MCTS Search started with parameters: start_proteins={SEARCH_PROTEINS}, stop_protein={STOP_PROTEIN}, desired_length={DESIRED_LENGTH}")
    
    base = MCTS_ITERATIONS // NUM_PARALLEL_WORKERS
    rem = MCTS_ITERATIONS % NUM_PARALLEL_WORKERS
    tasks = []
    
    for start in SEARCH_PROTEINS:
        for i in range(NUM_PARALLEL_WORKERS):
            iters = base + (1 if i < rem else 0)
            tasks.append((start, iters))
    
    logger.info(f"Starting MCTS: {MCTS_ITERATIONS} iterations over {NUM_PARALLEL_WORKERS} workers for proteins {SEARCH_PROTEINS}")
    
    start_time = time.time()
    with Pool(processes=NUM_PARALLEL_WORKERS, initializer=init_worker, initargs=(CORR_MATRIX, GENE_TO_INDEX)) as pool:
        logger.info(f"Worker pool initialized with {NUM_PARALLEL_WORKERS} processes")
        all_results = pool.map(run_mcts_worker, tasks)
        logger.info(f"All worker processes completed")
    
    # After collecting all results from the worker pool
    flat = [r for sub in all_results for r in sub]
    logger.info(f"Total paths found before merging: {len(flat)}")

    # Merge duplicate paths
    merged_results = merge_duplicate_paths_streaming(flat)
    logger.info(f"Total unique paths after merging: {len(merged_results)}")
    
    #Flat for unmerged
    grouped = {}
    for r in merged_results:
        grouped.setdefault(r['start'], []).append(r)
    
    # Fix for logical issue #2: sort all paths globally
    all_valid_paths = []
    for start, paths in grouped.items():
        logger.info(f"Protein {start} found {len(paths)} valid paths")
        all_valid_paths.extend(paths)
    
    all_valid_paths.sort(key=lambda x: (x['avg_value'], -abs(x['length']-DESIRED_LENGTH)), reverse=True)
    top_overall = all_valid_paths[:20]
    
    output = {
        'top_overall_paths': top_overall,
        'parameters': {
            'start_proteins': SEARCH_PROTEINS,
            'stop_protein': STOP_PROTEIN,
            'desired_length': DESIRED_LENGTH,
            'iterations': MCTS_ITERATIONS,
            'chain_length': MCTS_MAX_PATH_LENGTH
        }
    }
    
    out_file = os.path.join(RESULTS_DIR, 'top_mcts_paths.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"MCTS Search completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to {out_file}")
    
    # Log top 3 paths for quick reference
    for i, path in enumerate(top_overall[:3]):
        logger.info(f"Top path #{i+1}: start={path['start']}, length={path['length']}, value={path['avg_value']:.4f}")
