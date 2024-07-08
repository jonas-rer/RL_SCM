import numpy as np
import pandas as pd
from tqdm import tqdm

np.seterr(all="raise")

class Site:

    # constructor method
    def __init__(self, name, demand, childs, parent, error, cost, total_cost, inventory, ss):
        self.name = name
        self.demand = demand
        self.childs = childs
        self.parent = parent
        self.error = error
        self.cost = cost

        self.total_cost = total_cost
        self.inventory = inventory

        self.ss = ss

    def assign_parent(self, parent):
        self.parent = parent

    def dep_demand(self):
        if self.demand:
            return self.demand
        else:
            return sum([c.dep_demand() for c in self.childs])

    def replenish(self):
        if self.parent:
            self.inventory += self.dep_demand()*10
            self.parent.inventory -= self.dep_demand()*10
            self.total_cost += self.cost

    def __repr__(self):
        return f"Site: {self.name} ss: {self.ss}, inventrory: {self.inventory}, cost: {self.total_cost}"




ss = 100

F = Site("F",0.35, [], None, 0.1, 1, 0, 0, 20)
E = Site("E",0.35, [], None, 0.1, 1, 0, 0, 20)
D = Site("D",None, [E, F], None, None, 100, 0, 0, 20)
C = Site("C",0.3, [], None, 0.1, 1, 0, 0, 20)
B = Site("B",None, [C, D], None, None, 100, 0, 0, 20)
A = Site("A",None, [B], None, None, None, 0, 0, None)

F.assign_parent(D)
E.assign_parent(D)

D.assign_parent(B)
C.assign_parent(B)

B.assign_parent(A)

Stack = [F, E, D, C, B, A]


def stack_inventory(stack):
    inv = 0
    for s in stack:
        inv += s.inventory
    return inv


def reset(stack):
    for s in stack:
        s.total_cost = 0
        s.inventory = 0


def shuffle_ss(stack):
    ss_new = np.random.randint(0, 100, len(stack)-1)
    ss_new = np.round(np.divide(ss_new, ss_new.sum()) * 75).astype(int)+5

    while ss_new.sum() != 100:
        if ss_new.sum() > 100:
            ss_new[ss_new == ss_new.max()] -= 1
        else:
            ss_new[ss_new == ss_new.min()] += 1

    for i in range(len(stack[:-1])):
        stack[i].ss = ss_new[i]


def improve(stack):
    costs = np.asarray([s.total_cost for s in stack[:-1]])
    inventories = np.asarray([s.inventory/s.ss for s in stack[:-1]])
    if any([c > 0 for c in costs]):
        low_cost = np.where(costs == min(costs))[0]
        high_inventory = np.where(inventories[low_cost] == max(inventories[low_cost]))[0][0]
        candidate = low_cost[high_inventory]
        s_from = Stack[candidate]
        s_to = Stack[list(costs).index(max(costs))]
        update = max(1, (s_from.inventory*s_to.ss - s_to.inventory*s_from.ss)/(s_from.inventory + s_to.inventory))
        update = np.floor(min(s_from.ss/5, update))
        s_to.ss += update
        s_from.ss -= update


def run_stack(stack):
    if stack_inventory(stack) < ss/2:
        for site in stack:
            if site.ss:
                site.inventory += site.ss

    for site in stack:
        if site.ss is not None:
            if site.ss == 0:
                a=1
            if site.inventory < site.ss/2:
                site.replenish()

    for site in stack:
        if site.demand:
            site.inventory -= site.demand * (1+max(min(np.random.normal(0, site.error), 1), -1))



result = [pd.DataFrame(columns=["cost", "F ss", "E ss", "D ss", "C ss", "B ss"])]


def objective_function(stack):
    for i in tqdm(range(100000), position=1, leave=False, desc="Running"):
        run_stack(stack)

        if i % 1000 == 0:
            improve(stack)
        if i % 3000 == 0:
            improve(stack)
            reset(stack)

    reset(stack)
    for i in range(50000):
        run_stack(stack)

    obj = sum([s.total_cost for s in stack])
    prms = [s.ss for s in stack]

    result.append(pd.DataFrame([[obj]+prms[:-1]], columns=["cost", "F ss", "E ss", "D ss", "C ss", "B ss"]))


for i in tqdm(range(30), position=0, leave=True, desc="Optimizing"):
    shuffle_ss(Stack)
    reset(Stack)
    objective_function(Stack)
    a = 1

res = pd.concat(result).sort_values("cost")