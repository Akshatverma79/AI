# delivery_talkers_fixed_final.py
import pygame
import sys
import random
from collections import deque, namedtuple

# ---------------------------
# Config (tweak these)
# ---------------------------
CELL_SIZE = 64
GRID_ROWS = 7
GRID_COLS = 10
SCREEN_WIDTH = CELL_SIZE * GRID_COLS
SCREEN_HEIGHT = CELL_SIZE * GRID_ROWS + 140  # extra for UI panel
FPS = 1  # slower animation by default
AGENT_MOVE_DELAY_MS = 300  # delay after each cell step (ms)

# Colors
WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
AGENT_COLORS = [(60, 120, 220), (220, 100, 100), (120, 200, 120)]
PICKUP_COLOR = (255, 200, 0)
DROP_COLOR = (100, 60, 180)
PATH_COLOR = (150, 150, 150)

# ---------------------------
# Data classes
# ---------------------------
Order = namedtuple("Order", ["id", "pickup", "drop", "status"])
# status: 'waiting', 'assigned', 'picked', 'delivered'

class Agent:
    def __init__(self, name, pos, color):
        self.name = name
        self.pos = pos  # (r,c)
        self.color = color
        self.inbox = []  # messages: tuples (sender, msg)
        self.task = None  # order id assigned
        self.path = []  # list of cells to follow (each cell is (r,c))
        self.carrying = False

    def send(self, other, message):
        other.inbox.append((self.name, message))

    def broadcast(self, agents, message):
        for other in agents:
            if other is not self:
                self.send(other, message)

    def receive_all(self):
        msgs = list(self.inbox)
        self.inbox.clear()
        return msgs

    def set_path(self, path_list):
        # path_list should be a list of cells to move to (in order)
        self.path = list(path_list)

    def step_along_path(self):
        if self.path:
            next_cell = self.path.pop(0)
            self.pos = next_cell
            return True
        return False

# ---------------------------
# Utilities: BFS for pathfinding
# ---------------------------
def neighbors(cell):
    r, c = cell
    for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            yield (nr, nc)

def bfs_path(start, goal, blocked=set()):
    """Return path from start to goal inclusive. Return [] if not reachable."""
    if start == goal:
        return [start]
    q = deque([start])
    parent = {start: None}
    while q:
        u = q.popleft()
        for v in neighbors(u):
            if v in parent or v in blocked:
                continue
            parent[v] = u
            if v == goal:
                # reconstruct path
                path = []
                cur = v
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            q.append(v)
    return []

def path_distance_safe(start, goal):
    """Return number of steps in shortest path, or a large number if unreachable."""
    p = bfs_path(start, goal)
    if not p:
        return float('inf')
    return len(p) - 1  # steps (edges), not nodes

# ---------------------------
# Simulation setup
# ---------------------------
def create_sample_orders():
    # Create some sample orders; ensure they are not at agent start positions
    used = set()
    orders = []
    idc = 1
    # generate 6 orders
    while len(orders) < 6:
        pr = random.randint(0, GRID_ROWS-1)
        pc = random.randint(0, GRID_COLS-1)
        dr = random.randint(0, GRID_ROWS-1)
        dc = random.randint(0, GRID_COLS-1)
        if (pr,pc) == (1,1) or (dr,dc) == (GRID_ROWS-2, GRID_COLS-2):
            continue
        if (pr,pc) == (dr,dc):
            continue
        if (pr,pc) in used or (dr,dc) in used:
            continue
        used.add((pr,pc)); used.add((dr,dc))
        orders.append(Order(id=idc, pickup=(pr,pc), drop=(dr,dc), status='waiting'))
        idc += 1
    return orders

# ---------------------------
# Messaging / Decision logic
# ---------------------------
def announce_order_and_choose_taker(agents, order, logs):
    """
    Simulate simple message exchange and pick nearest agent as taker.
    Returns the chosen Agent object (or None if none reachable).
    """
    # compute distances (safe)
    dists = {}
    for ag in agents:
        d = path_distance_safe(ag.pos, order.pickup)
        dists[ag.name] = d

    # broadcast CL (close) messages for trace
    for ag in agents:
        ag.broadcast(agents, f"CL{order.id}")
        logs.append(f"{ag.name} -> broadcast CL{order.id}")

    # choose nearest agent (if multiple tied, pick first)
    nearest_name = min(dists, key=lambda k: dists[k])
    if dists[nearest_name] == float('inf'):
        logs.append(f"No agent can reach Order {order.id} pickup (unreachable). Skipping assignment.")
        return None

    taker = next(a for a in agents if a.name == nearest_name)
    taker.broadcast(agents, f"TA{order.id}")
    logs.append(f"{taker.name} -> broadcast TA{order.id}")
    return taker

def process_inboxes(agents, logs):
    # For UI only: collect recent messages and also append to logs
    recent = []
    for ag in agents:
        msgs = ag.receive_all()
        for sender, msg in msgs:
            recent.append((ag.name, sender, msg))
            logs.append(f"{sender} -> {ag.name}: {msg}")
    return recent

# ---------------------------
# Reassignment helper
# ---------------------------
def assign_nearest_waiting_to_agent(agent, agents, orders_map, logs):
    """Find nearest waiting order and assign to given agent. Returns assigned order id or None."""
    waiting = [o for o in orders_map.values() if o['status'] == 'waiting']
    if not waiting:
        return None
    # choose nearest by path distance; treat unreachable as inf
    def dist_to_pick(o):
        d = path_distance_safe(agent.pos, o['pickup'])
        return d
    waiting_sorted = sorted(waiting, key=dist_to_pick)
    best = waiting_sorted[0]
    if dist_to_pick(best) == float('inf'):
        return None
    # assign
    best['status'] = 'assigned'
    agent.task = best['id']

    # compute robust full path:
    p1 = bfs_path(agent.pos, best['pickup'])  # includes agent.pos and pickup
    p2 = bfs_path(best['pickup'], best['drop'])  # includes pickup and drop

    full_path = []
    # move to pickup: add steps after current position (if any)
    if len(p1) > 1:
        full_path.extend(p1[1:])  # move along to pickup
    # now move from pickup to drop: if pickup==drop p2 == [pickup] -> no moves needed
    if len(p2) > 1:
        full_path.extend(p2[1:])

    agent.set_path(full_path)
    logs.append(f"{agent.name} assigned to order {best['id']} (reassign), path len {len(full_path)}")

    # If agent is already at pickup, perform immediate pickup (and possible immediate delivery)
    if agent.pos == best['pickup']:
        # immediate pickup
        if not agent.carrying:
            agent.carrying = True
            best['status'] = 'picked'
            logs.append(f"{agent.name} immediate-pick order {best['id']} (at pickup).")
        # if pickup == drop then immediate delivery
        if best['pickup'] == best['drop']:
            best['status'] = 'delivered'
            agent.task = None
            agent.carrying = False
            agent.path = []
            logs.append(f"{agent.name} immediate-delivered order {best['id']} (pickup==drop).")

    return best['id']

# ---------------------------
# Main simulation with pygame
# ---------------------------
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Delivery Talkers — Simulation (Final Fix)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)
    bigfont = pygame.font.SysFont(None, 26)

    # Agents initial positions
    agents = [
        Agent("A", (1,1), AGENT_COLORS[0]),
        Agent("B", (GRID_ROWS-2, GRID_COLS-2), AGENT_COLORS[1]),
    ]

    # Orders (list of Order namedtuples)
    orders = create_sample_orders()
    # Convert orders to mutable dicts
    orders_map = {}
    for o in orders:
        orders_map[o.id] = {"id": o.id, "pickup": o.pickup, "drop": o.drop, "status": o.status}

    logs = []
    recent_msgs = []

    time_step = 0
    order_queue = [o for o in orders_map.values()]
    current_assignments = {}  # order_id -> agent.name

    # spawn one order every N steps
    steps_between_orders = 6
    spawning_index = 0

    running = True
    while running:
        clock.tick(FPS)
        time_step += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Spawn new order (assign at spawn time)
        if spawning_index < len(order_queue) and time_step % steps_between_orders == 1:
            order = order_queue[spawning_index]
            order_id = order['id']
            logs.append(f"Order {order_id} appeared at {order['pickup']} -> {order['drop']}")
            # choose taker
            taker = announce_order_and_choose_taker(agents, Order(**order), logs)
            # show messages
            recent_msgs = process_inboxes(agents, logs)
            if taker is not None:
                order['status'] = 'assigned'
                current_assignments[order_id] = taker.name
                taker.task = order_id

                # compute robust path (fixed logic)
                p1 = bfs_path(taker.pos, order['pickup'])
                p2 = bfs_path(order['pickup'], order['drop'])
                full_path = []
                if len(p1) > 1:
                    full_path.extend(p1[1:])  # steps to pickup
                if len(p2) > 1:
                    full_path.extend(p2[1:])  # steps from pickup to drop
                taker.set_path(full_path)

                logs.append(f"{taker.name} assigned to order {order_id}, path len {len(full_path)}")

                # immediate pickup/delivery if positioned at pickup or pickup==drop
                if taker.pos == order['pickup']:
                    taker.carrying = True
                    order['status'] = 'picked'
                    logs.append(f"{taker.name} immediate-picked order {order_id} (spawn).")
                    if order['pickup'] == order['drop']:
                        order['status'] = 'delivered'
                        taker.task = None
                        taker.carrying = False
                        taker.path = []
                        current_assignments.pop(order_id, None)
                        logs.append(f"{taker.name} immediate-delivered order {order_id} (pickup==drop).")
            else:
                logs.append(f"No taker assigned for Order {order_id} at spawn (unreachable).")
            spawning_index += 1

        # Move agents along their paths
        for ag in agents:
            if ag.task is not None and ag.path:
                moved = ag.step_along_path()
                if moved:
                    # small pause so movement is visible
                    pygame.time.delay(AGENT_MOVE_DELAY_MS)

                    assigned_order = orders_map.get(ag.task)
                    if assigned_order:
                        # Picked?
                        if (not ag.carrying) and (ag.pos == assigned_order['pickup']):
                            ag.carrying = True
                            assigned_order['status'] = 'picked'
                            logs.append(f"{ag.name} picked order {ag.task}")
                        # Delivered?
                        elif ag.carrying and (ag.pos == assigned_order['drop']):
                            assigned_order['status'] = 'delivered'
                            logs.append(f"{ag.name} delivered order {ag.task}")
                            # clear agent
                            ag.task = None
                            ag.carrying = False
                            ag.path = []
                            # remove mapping(s)
                            to_remove = [oid for oid, aname in current_assignments.items() if aname == ag.name]
                            for oid in to_remove:
                                current_assignments.pop(oid, None)
            # end movement for this agent

        # Global proactive reassignment pass:
        # For every idle agent, try to assign a waiting order (once)
        for ag in agents:
            if ag.task is None:
                reassigned = assign_nearest_waiting_to_agent(ag, agents, orders_map, logs)
                if reassigned:
                    current_assignments[reassigned] = ag.name

        # Also periodically process inboxes for UI (optional)
        _ = process_inboxes(agents, logs)

        # Draw background
        screen.fill(WHITE)

        # Draw grid
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                rect = pygame.Rect(c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, LIGHT_GRAY if (r+c)%2==0 else WHITE, rect)
                pygame.draw.rect(screen, GRAY, rect, 1)

        # Draw all orders (pickup as yellow box, drop as purple)
        for oid, ordict in orders_map.items():
            pr, pc = ordict['pickup']
            dr, dc = ordict['drop']
            # show if not delivered
            if ordict['status'] != 'delivered':
                # pickup icon
                pr_rect = pygame.Rect(pc*CELL_SIZE+8, pr*CELL_SIZE+8, CELL_SIZE-16, CELL_SIZE-16)
                pygame.draw.rect(screen, PICKUP_COLOR, pr_rect)
                id_surf = font.render(f"P{oid}", True, BLACK)
                screen.blit(id_surf, (pc*CELL_SIZE+10, pr*CELL_SIZE+10))

                # drop icon
                dr_rect = pygame.Rect(dc*CELL_SIZE+18, dr*CELL_SIZE+18, CELL_SIZE-36, CELL_SIZE-36)
                pygame.draw.rect(screen, DROP_COLOR, dr_rect)
                id2 = font.render(f"D{oid}", True, WHITE)
                screen.blit(id2, (dc*CELL_SIZE+20, dr*CELL_SIZE+20))

        # Draw agent paths (where applicable)
        for ag in agents:
            cur = ag.pos
            for i, cell in enumerate([cur] + ag.path):
                cx = cell[1]*CELL_SIZE + CELL_SIZE//2
                cy = cell[0]*CELL_SIZE + CELL_SIZE//2
                radius = 4 if i==0 else 2
                pygame.draw.circle(screen, PATH_COLOR, (cx, cy), radius)

        # Draw agents
        for ag in agents:
            r, c = ag.pos
            center = (c*CELL_SIZE + CELL_SIZE//2, r*CELL_SIZE + CELL_SIZE//2)
            pygame.draw.circle(screen, ag.color, center, CELL_SIZE//3)
            name_surf = font.render(ag.name, True, WHITE)
            screen.blit(name_surf, (center[0]-8, center[1]-8))

            # show task badge
            if ag.task is not None:
                badge = bigfont.render(f"{ag.task}", True, BLACK)
                screen.blit(badge, (c*CELL_SIZE + 4, r*CELL_SIZE + 4))
                # show carrying indicator
                if ag.carrying:
                    carry_surf = font.render("📦", True, BLACK)
                    screen.blit(carry_surf, (c*CELL_SIZE + CELL_SIZE - 28, r*CELL_SIZE + 4))

        # UI panel (bottom)
        panel_rect = pygame.Rect(0, GRID_ROWS*CELL_SIZE, SCREEN_WIDTH, 140)
        pygame.draw.rect(screen, (245,245,245), panel_rect)
        pygame.draw.rect(screen, GRAY, panel_rect, 1)

        # show logs (last few)
        display_logs = logs[-8:][::-1]  # latest first
        for i, line in enumerate(display_logs):
            text = font.render(line, True, BLACK)
            screen.blit(text, (8, GRID_ROWS*CELL_SIZE + 6 + i*18))

        # show agents' status on right panel
        for i, ag in enumerate(agents):
            info = f"{ag.name} pos:{ag.pos} task:{ag.task or 'idle'} carrying:{ag.carrying}"
            txt = font.render(info, True, BLACK)
            screen.blit(txt, (SCREEN_WIDTH - 420, GRID_ROWS*CELL_SIZE + 6 + i*18))

        # show step count and waiting orders
        waiting_ids = [str(o['id']) for o in orders_map.values() if o['status']=='waiting']
        wi_text = font.render("Waiting: " + (", ".join(waiting_ids) if waiting_ids else "None"), True, BLACK)
        screen.blit(wi_text, (8, GRID_ROWS*CELL_SIZE + 6 + 8*18))

        ts = font.render(f"Step: {time_step}", True, BLACK)
        screen.blit(ts, (SCREEN_WIDTH - 100, GRID_ROWS*CELL_SIZE + 6 + 8*18))

        pygame.display.flip()

        # stop if all delivered and all orders spawned
        done_all = all(o['status']=='delivered' for o in orders_map.values())
        if done_all and spawning_index >= len(order_queue):
            logs.append("All orders delivered. Simulation complete.")
            pygame.time.delay(1500)
            running = False

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    random.seed(42)
    run_simulation()
