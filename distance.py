import pandas as pd
import gurobipy as gp
import itertools as it


# Input: Two partition assignment dictionaries on the same graph (P,Q)
#        and a Boolean parameter (normalize) to indicate whether the value returned
#        should be normalized (False: return un-normalized)
# Output: The unweighted Rand distance
def RandDistance(P,Q,normalize):
    
    disagree = 0
    count = 0
    
    for u,v in it.combinations(P.keys(),2):
        count += 1
        if P[u] == P[v] and Q[u] != Q[v]:
            #print(u,v)
            disagree += 1
        elif P[u] != P[v] and Q[u] == Q[v]:
            #print(u,v)
            disagree += 1
            
    if normalize:
        return float(disagree)/count # normalized value
    else:
        return disagree # un-normalized value


# Input: Two partition assignment dictionaries on the same graph (P,Q),
#        a graph adjacency dictionary, and a Boolean parameter to indicate
#        whether the value returned should be normalized (False: return un-normalized)
# Output: The unweighted boundary distance
def BoundaryDistance(P,Q,adj,normalize):
    
    disagree = 0
    
    for u,v in adj:
        if P[u] == P[v] and Q[u] != Q[v]:
            disagree += 1
        elif P[u] != P[v] and Q[u] == Q[v]:
            disagree += 1
            
    if normalize:
        return float(disagree)/len(adj.keys()) # normalized value
    else:
        return disagree # un-normalized value


# Input: Two partition assignment dictionaries on the same graph (P,Q), Gerrychain graph object,
#        column string to weight by ('UNWEIGHTED' for unweighted case), and Boolean parameter (normalize)
#        to indicate whether the value returned should be normalized (False: return un-normalized)
# Output: The transfer distance and a corresponding optimal part matching
def TransferDistance(P,Q,graph,col,normalize):

    # Set-up
    
    # Vertices
    Vertices = [val for val in P.keys()]

    # Parts
    Parts = []
    for val in P.values():
        if val not in Parts:
            Parts.append(val)

    # Auxiliary graph edges
    Edges = {}

    for p in Parts:
        for q in Parts:
            Edges[(p,q)] = 0

    if col == 'UNWEIGHTED':
        for vert in Vertices:
            Edges[(P[vert],Q[vert])] += 1
    else:
        total_sum = 0
        for n in graph.nodes:
            Edges[(P[graph.nodes[n]['GEOID20']],Q[graph.nodes[n]['GEOID20']])] += graph.nodes[n][col]
            total_sum += graph.nodes[n][col]

        
    # Gurobi input

    # Matching data
    pairs,weights = gp.multidict(Edges)

    # Declare and initialize model
    m = gp.Model('RAP')

    # Create decision variables for the RAP model
    x = m.addVars(pairs, name='assign')

    # Create job constraints
    jobs = m.addConstrs((x.sum('*',p) == 1 for p in Parts), name='jobs')

    # Create resource constraints
    resources = m.addConstrs((x.sum(q,'*') <= 1 for q in Parts), name='resources')

    # Objective: maximize total matching weight of all assignments
    m.setObjective(x.prod(weights), gp.GRB.MAXIMIZE)


    # Optimization

    # Run optimization engine
    m.optimize()
    

    # Gather optimal matching
    matching_opt = {}

    for v in m.getVars():
        if v.x > 1e-6:
            matching_opt[v.varName[7]] = v.varName[9]
            #print(v.varName, v.x)


    # Return optimal total matching score and optimal matching
    if col == 'UNWEIGHTED':
        if normalize:
            return ((len(Vertices) - m.objVal)/(len(Vertices)-1)),matching_opt # normalized value
        else:
            return (len(Vertices) - m.objVal),matching_opt # un-normalized value
    else:
        if normalize:
            return ((total_sum - m.objVal)/(len(Vertices)-1)),matching_opt # normalized value
        else:
            return (total_sum - m.objVal),matching_opt # un-normalized value

