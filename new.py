"""
ctf_sim.py

Grid-based Capture-The-Flag simulation with:
 - A* pathfinding for movement
 - Attackers (seek enemy flag) and Defenders (patrol + chase intruders inside a defensive zone)
 - Tagging (opponent presence on same cell sends agent back to spawn and drops flag)
 - Animated output saved as a GIF and capture logs printed at the end.

Usage:
  pip install matplotlib imageio numpy
  python ctf_sim.py

Author: ChatGPT (GPT-5 Thinking mini)
"""

import random, heapq
import matplotlib
matplotlib.use("Agg")        # Use Agg backend for off-screen rendering
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

# ---------------- PARAMETERS ----------------
GRID_W, GRID_H = 16, 10       # grid size (width, height)
NUM_ATTACKERS = 2
NUM_DEFENDERS = 1
OBSTACLE_DENSITY = 0.10
MAX_STEPS = 400
FPS = 10
OUT_GIF = "ctf_animation.gif"  # output gif

# ---------------- HELPERS & A* ----------------
dirs4 = [(1,0),(-1,0),(0,1),(0,-1)]
def in_bounds(x,y): return 0 <= x < GRID_W and 0 <= y < GRID_H
def heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(start, goal, obstacles:set):
    """Return path (list of grid coords) from start to goal inclusive, or [] if unreachable."""
    if start == goal:
        return [start]
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start,goal), 0, start, None))
    came_from = {}
    gscore = {start:0}
    closed = set()
    while open_heap:
        f, g, current, parent = heapq.heappop(open_heap)
        if current in closed: continue
        came_from[current] = parent
        if current == goal:
            # reconstruct
            path = [current]
            p = came_from[current]
            while p is not None:
                path.append(p)
                p = came_from[p]
            return list(reversed(path))
        closed.add(current)
        for dx,dy in dirs4:
            nb = (current[0]+dx, current[1]+dy)
            if not in_bounds(*nb): continue
            if nb in obstacles: continue
            if nb in closed: continue
            tentative = g + 1
            if tentative < gscore.get(nb, 1e9):
                gscore[nb] = tentative
                heapq.heappush(open_heap, (tentative + heuristic(nb,goal), tentative, nb, current))
    return []

# ---------------- ENTITIES ----------------
@dataclass
class Flag:
    team: int
    pos: Tuple[int,int]
    home_pos: Tuple[int,int]
    carried_by: Optional['Agent'] = None
    dropped_pos: Optional[Tuple[int,int]] = None

@dataclass
class Agent:
    team: int
    role: str              # 'attacker' or 'defender'
    pos: Tuple[int,int]
    spawn: Tuple[int,int]
    id: int
    carrying: Optional[Flag] = None
    stun_timer: int = 0    # respawn delay while > 0
    path: List[Tuple[int,int]] = field(default_factory=list)
    behavior_target: Optional[Tuple[int,int]] = None

    def step_towards(self, target, obstacles:set):
        if self.stun_timer > 0:
            self.stun_timer -= 1
            return
        if target is None:
            return
        if self.pos == target:
            self.path = []
            return
        if not self.path or self.path[-1] != target:
            self.path = a_star(self.pos, target, obstacles)
            if len(self.path) >= 2:
                self.path = self.path[1:]
        if self.path:
            self.pos = self.path.pop(0)

    def reset_to_spawn(self):
        self.pos = self.spawn
        self.carrying = None
        self.path = []
        self.stun_timer = 6

# ---------------- MAP & INITIALIZE ----------------
random.seed(1)
obstacles: Set[Tuple[int,int]] = set()
for x in range(GRID_W):
    for y in range(GRID_H):
        if random.random() < OBSTACLE_DENSITY:
            obstacles.add((x,y))

# define home/flag positions (left and right center)
left_home = (1, GRID_H//2)
right_home = (GRID_W-2, GRID_H//2)
# ensure clearance around homes
for dx in range(-2,3):
    for dy in range(-2,3):
        obstacles.discard((left_home[0]+dx, left_home[1]+dy))
        obstacles.discard((right_home[0]+dx, right_home[1]+dy))

flag_blue = Flag(team=0, pos=left_home, home_pos=left_home)
flag_red  = Flag(team=1, pos=right_home, home_pos=right_home)

agents: List[Agent] = []
def spawn_positions(side, count):
    positions = []
    xbase = 1 if side == 'left' else GRID_W-2
    ybase = GRID_H//2
    ring = 0
    while len(positions) < count:
        for dx in range(-ring, ring+1):
            for dy in range(-ring, ring+1):
                x = xbase + dx; y = ybase + dy
                if in_bounds(x,y) and (x,y) not in obstacles and (x,y) not in positions:
                    positions.append((x,y))
                    if len(positions) == count:
                        return positions
        ring += 1
    return positions

blue_spawns = spawn_positions('left', NUM_ATTACKERS+NUM_DEFENDERS)
red_spawns  = spawn_positions('right', NUM_ATTACKERS+NUM_DEFENDERS)

aid = 0
for i,p in enumerate(blue_spawns):
    role = 'attacker' if i < NUM_ATTACKERS else 'defender'
    agents.append(Agent(team=0, role=role, pos=p, spawn=p, id=aid)); aid += 1
for i,p in enumerate(red_spawns):
    role = 'attacker' if i < NUM_ATTACKERS else 'defender'
    agents.append(Agent(team=1, role=role, pos=p, spawn=p, id=aid)); aid += 1

# ---------------- BEHAVIOR LOGIC ----------------
logs: List[str] = []
captured_counts = [0,0]

def team_flag(team): return flag_blue if team==0 else flag_red
def enemy_flag(team): return flag_red if team==0 else flag_blue
def defensive_zone_center(team): return team_flag(team).home_pos

def choose_goal(agent:Agent):
    # If stunned, return to spawn
    if agent.stun_timer > 0:
        return agent.spawn
    # If carrying enemy flag, go home
    if agent.carrying:
        return agent.spawn
    if agent.role == 'attacker':
        ef = enemy_flag(agent.team)
        # if friend carries the flag -> go home, if enemy carries -> intercept
        if ef.carried_by is not None:
            carrier = ef.carried_by
            if carrier.team != agent.team:
                return carrier.pos
            else:
                return agent.spawn
        if ef.dropped_pos is not None:
            return ef.dropped_pos
        return ef.pos
    else:
        # defender: look for intruders in defensive zone, else patrol near home
        center = defensive_zone_center(agent.team)
        zone_radius = 4
        intruders = [a for a in agents if a.team != agent.team and a.stun_timer==0]
        intruders_in_zone = [a for a in intruders if heuristic(a.pos, center) <= zone_radius]
        if intruders_in_zone:
            intruders_in_zone.sort(key=lambda x: heuristic(x.pos, agent.pos))
            return intruders_in_zone[0].pos
        if agent.behavior_target is None or agent.behavior_target == agent.pos or random.random() < 0.05:
            hx, hy = center
            tx = max(0, min(GRID_W-1, hx + random.randint(-2,2)))
            ty = max(0, min(GRID_H-1, hy + random.randint(-2,2)))
            agent.behavior_target = (tx,ty)
        return agent.behavior_target

# ---------------- INTERACTIONS ----------------
def resolve_interactions(step:int):
    pos_to_agents: Dict[Tuple[int,int], List[Agent]] = {}
    for a in agents:
        pos_to_agents.setdefault(a.pos, []).append(a)
    # Tagging: if opposing teams share a cell => tagged and reset
    for pos, alist in pos_to_agents.items():
        teams_present = set(a.team for a in alist)
        if len(teams_present) > 1:
            for a in alist:
                enemies = [e for e in alist if e.team != a.team]
                if enemies:
                    if a.carrying:
                        f = a.carrying
                        f.carried_by = None
                        f.dropped_pos = a.pos
                        logs.append(f"Step {step}: Agent {a.id} (team {a.team}) dropped flag at {a.pos} after being tagged.")
                    a.reset_to_spawn()
    # Picking up flags when an opposing agent is at the flag home or dropped position
    for f in (flag_blue, flag_red):
        if f.carried_by is not None: continue
        pos = f.dropped_pos if f.dropped_pos is not None else f.pos
        for a in agents:
            if a.team != f.team and a.pos == pos and a.stun_timer == 0:
                f.carried_by = a
                a.carrying = f
                f.dropped_pos = None
                logs.append(f"Step {step}: Agent {a.id} (team {a.team}) picked flag of team {f.team} at {pos}.")
                break
    # Capture: carrier reaches their spawn/home
    for a in agents:
        if a.carrying and a.pos == a.spawn:
            captured_counts[a.team] += 1
            logs.append(f"Step {step}: Agent {a.id} (team {a.team}) captured flag of team {a.carrying.team}!")
            f = a.carrying
            f.carried_by = None
            f.pos = f.home_pos
            f.dropped_pos = None
            a.carrying = None
            a.stun_timer = 6

# ---------------- SIMULATION & ANIMATION ----------------
def run_simulation_and_render():
    images = []
    fig = plt.figure(figsize=(8,5))
    for step in range(MAX_STEPS):
        # decision + movement
        for a in agents:
            goal = choose_goal(a)
            a.step_towards(goal, obstacles)
        # interactions (tagging, pick/drop, capture)
        resolve_interactions(step)
        # draw
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlim(-0.5, GRID_W-0.5); ax.set_ylim(-0.5, GRID_H-0.5)
        ax.set_xticks(range(GRID_W)); ax.set_yticks(range(GRID_H))
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_aspect('equal')
        ax.grid(True, linewidth=0.4, linestyle=':')
        # obstacles
        for (x,y) in obstacles:
            ax.add_patch(patches.Rectangle((x-0.5,y-0.5),1,1,facecolor='black',alpha=0.9))
        # homes
        ax.add_patch(patches.Rectangle((flag_blue.home_pos[0]-0.5, flag_blue.home_pos[1]-0.5),1,1,fill=False, linewidth=2))
        ax.add_patch(patches.Rectangle((flag_red.home_pos[0]-0.5, flag_red.home_pos[1]-0.5),1,1,fill=False, linewidth=2))
        # agents
        xs=[]; ys=[]; cs=[]; sizes=[]
        for a in agents:
            xs.append(a.pos[0]); ys.append(a.pos[1])
            if a.stun_timer > 0:
                cs.append('gray'); sizes.append(120)
            else:
                if a.team == 0:
                    cs.append('dodgerblue' if a.role=='attacker' else 'cyan')
                else:
                    cs.append('crimson' if a.role=='attacker' else 'lightcoral')
                sizes.append(160 if a.role=='attacker' else 120)
        if xs:
            ax.scatter(xs, ys, s=sizes, c=cs, edgecolors='k', linewidths=0.5, zorder=4)
        # flags
        fx=[]; fy=[]; fc=[]
        for f in (flag_blue, flag_red):
            if f.carried_by:
                fx.append(f.carried_by.pos[0]); fy.append(f.carried_by.pos[1]); fc.append('yellow' if f.team==0 else 'orange')
            else:
                pos = f.dropped_pos if f.dropped_pos is not None else f.pos
                fx.append(pos[0]); fy.append(pos[1]); fc.append('blue' if f.team==0 else 'red')
        ax.scatter(fx, fy, s=220, c=fc, marker='P', edgecolors='k', zorder=5)
        ax.set_title(f"Step {step}  | Scores - Blue: {captured_counts[0]}  Red: {captured_counts[1]}  | Events: {len(logs)}")
        # agent id labels
        for a in agents:
            ax.text(a.pos[0], a.pos[1], f"{a.id}", color='black', fontsize=7, ha='center', va='center')
        # capture frame to image buffer
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(img.copy())
        # stop when any capture occurs (you can change this to require N captures)
        if captured_counts[0] >= 1 or captured_counts[1] >= 1:
            break
    # save gif
    imageio.mimsave(OUT_GIF, images, fps=FPS)
    return OUT_GIF

if __name__ == "__main__":
    print("Running Capture-The-Flag simulation...")
    gifpath = run_simulation_and_render()
    print(f"Animation saved to: {gifpath}\n")
    print("--- Capture Log ---")
    for ln in logs:
        print(ln)
    print("\n--- Result ---")
    if captured_counts[0] > captured_counts[1]:
        print(f"Winner: Blue team (team 0) with {captured_counts[0]} captures.")
    elif captured_counts[1] > captured_counts[0]:
        print(f"Winner: Red team (team 1) with {captured_counts[1]} captures.")
    else:
        print("Draw or timeout - no clear winner. Scores:", captured_counts)
    print("\nTip: tweak GRID_W, GRID_H, NUM_ATTACKERS, NUM_DEFENDERS, and OBSTACLE_DENSITY to change difficulty and behavior.")