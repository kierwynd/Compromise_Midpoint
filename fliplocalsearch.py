from gerrychain.random import random

SEED = 0
while (SEED % 2 == 0):
    SEED = random.randrange(1000000001,9999999999)
#SEED = 7473378879
random.seed(SEED)
print(SEED)

import matplotlib.pyplot as plt
from gerrychain import Graph
import time
import csv
import collections as col


# Holds all district info from plan
class PLAN_VALUES:
    def __init__(self):
        self.numDistricts = 0
        self.borderNodes = [] # List of all border nodes
        self.borderNodesByDist = {} # Records border nodes for each district (used to calculate initial district perimeters & auxH)
        self.borderNodesYesNo = {} # For every node, record whether it's a border node (can be adapted to record # of districts node is adjacent to, besides its own)
        self.auxH = {}
        self.distPop = []
        self.meanDistPop = 0
        self.distPopPercent = []
        self.distPerimeter = []
        self.CutEdges = 0
        self.disagreeA = 0
        self.disagreeB = 0



# Input: Gerrychain graph object, adjacency dict
# Output: Gerrychain graph object with merged cut-vertices, adjacency dict, cut vertices dict
def merge_cut_vertices(graph,adj):
    
    # CHECK ALL CUT-VERTEX CANDIDATES -----------------------------------------------------------------

    # Check contiguity of aug. neighborhood of candidate upon removal of candidate using a breadth-first search

    cutVertices = {}

    for node in graph.nodes:
        if node != 'dummy':
            chosen_nbhd = [val for val in graph.nodes[node]['neighbors']]
            if 'dummy' in chosen_nbhd:
                chosen_nbhd.remove('dummy')
            chosen_nbhd_aug = [val for val in graph.nodes[node]['neighbors_aug']]
            if 'dummy' in chosen_nbhd_aug:
                chosen_nbhd_aug.remove('dummy')
            chosen_nbhd_aug = set(chosen_nbhd_aug)

            isPath = True
            x = chosen_nbhd[0]
            discovered = [x]
            bfsQ = col.deque()
            bfsQ.append(x)
            found = False
            while len(bfsQ) > 0:
                v = bfsQ.popleft()
                vNbhd = [val for val in graph.nodes[v]['neighbors']]
                if 'dummy' in vNbhd:
                    vNbhd.remove('dummy')
                vNbhd = set(vNbhd)
                vBoth = list(vNbhd.intersection(chosen_nbhd_aug)) # For aug-nbhd BFS
                if node in vBoth:
                    vBoth.remove(node)

                for w in vBoth:
                    if w not in discovered:
                        discovered.append(w)
                        bfsQ.append(w)


            for y in chosen_nbhd:
                if y not in discovered:
                    isPath = False
                    break


            # Add node to dictionary of cut-vertices if aug. neighborhood will become discontiguous
            if not isPath:
                cutVertices[node] = [] # this list will contain the nodes it surrounds


    print('Number of cut-vertices: ',len(cutVertices))
    
    
    # IDENTIFY UNITS SURROUNDED BY CUT-VERTICES -------------------------------------------------------

    # Start breadth-first search at a neighbor of a cut-vertex.
    # If search continues to reach more than 75 nodes, starting node is not surrounded.
    # If search stops before reaching 75 nodes, starting node is surrounded.

    for node in cutVertices:
        
        chosen_nbhd = [val for val in graph.nodes[node]['neighbors']]
        if 'dummy' in chosen_nbhd:
            chosen_nbhd.remove('dummy')
#         print(chosen_nbhd)
            
        for x in chosen_nbhd:
            
#             print('x: ',x)

            isSurrounded = True
            discovered = [x]
            bfsQ = col.deque()
            bfsQ.append(x)
            found = False
            while len(bfsQ) > 0:
                v = bfsQ.popleft()
                vBoth = [val for val in graph.nodes[v]['neighbors']]
                if 'dummy' in vBoth:
                    vBoth.remove('dummy')
                if node in vBoth:
                    vBoth.remove(node)

                for w in vBoth:
                    if w not in discovered:
                        discovered.append(w)
                        bfsQ.append(w)
#                         print('w: ',w)


                if len(discovered) > 75: # Large number due to odd peninsula of blocks in MO
                    isSurrounded = False
                    break
                    
#             print('isSurrounded: ',isSurrounded)

            # Record all nodes that cut-vertex "node" surrounds
            if isSurrounded:
                for d in discovered:
                    if d not in cutVertices[node]:
                        cutVertices[node].append(d)


    # Keep list of all surrounded units
    surrounded = []

    for cv in cutVertices:
        for unit in cutVertices[cv]:
            if unit not in surrounded:
                surrounded.append(unit)


    # If a unit is both a cut-vertex and surrounded, remove it from cutVertices
    for unit in surrounded:
        if unit in cutVertices:
            del cutVertices[unit]

    print('Number of cut-vertices (post-cleaning): ',len(cutVertices))
    
    # Make cut vertices dict with GEOIDs for surrounded units, so able to output full final assignment
    cutVertices_ID = {}
    for cv in cutVertices:
        cutVertices_ID[cv] = [graph.nodes[u]['GEOID20'] for u in cutVertices[cv]]
    
        
#     surrounded_outFile = open('IL_SurroundedTracts.csv','w')
#     surrounded_writer = csv.writer(surrounded_outFile,delimiter=',')
    
#     surrounded_writer.writerow(['GEOID20','Cut/Surrounded'])
    
#     for cv in cutVertices:
#         surrounded_writer.writerow([cv,'0'])
#         for surr in cutVertices[cv]:
#             surrounded_writer.writerow([surr,'1'])
        
#     surrounded_outFile.close()

    
    
    # MERGE DATA OF SURROUNDED UNITS WITH SURROUNDING UNIT --------------------------------------------
    
    for chosen in cutVertices:
        for unit in cutVertices[chosen]:
            
            graph.nodes[chosen]['POP20'] += graph.nodes[unit]['POP20']
            
            for nb in graph.nodes[unit]['neighbors']:
                graph.nodes[nb]['neighbors'].remove(unit)

            for nb in graph.nodes[unit]['neighbors_aug']:
                graph.nodes[nb]['neighbors_aug'].remove(unit)
                
            graph.remove_node(unit)


    edge_remove = []
    for e in adj:
        if e[0] in cutVertices:
            if e[1] in cutVertices[e[0]]:
                edge_remove.append(e)
                
        elif e[1] in cutVertices:
            if e[0] in cutVertices[e[1]]:
                edge_remove.append(e)
                
        elif e[0] in surrounded and e[1] in surrounded:
            edge_remove.append(e)
            
    for e in edge_remove:
        del adj[e]
    
            
    print('Units merged')
    #print(len(graph.nodes))
    
    return graph,adj,cutVertices_ID



# Input: Geopandas dataframe, bool to indicate grid graph or not
# Output: Gerrychain graph object, adjacency dict, cut vertices dict
def make_graph(df,grid_bool):

    # Create Graph objects from gerrychain with queen (aug) adjacency
    graph = Graph.from_geodataframe(df,'queen')

    # Give node attributes, if grid
    if grid_bool:
        for n in graph:
            graph.nodes[n]['POP20'] = 1
            graph.nodes[n]['GEOID20'] = str(int(graph.nodes[n]['id']))
    
    # Make sure all GEOIDs are strings, just in case
    for node in graph.nodes:
        temp = str(int(graph.nodes[node]['GEOID20']))
        graph.nodes[node]['GEOID20'] = temp
        
    # Make cleaned adjacency dictionary
    edgesLength = {}

    # Populate adjacency matrix with length of shared segment for normal adjacency and -1 for aug adjacency
    
    # Adjacency to outside
    for node in graph.nodes:
        if graph.nodes[node]['boundary_node']:
            if float(graph.nodes[node]['boundary_perim']) == 0:
                edgesLength[(node,'dummy')] = 0
                edgesLength[('dummy',node)] = 0
            else:
                edgesLength[(node,'dummy')] = float(graph.nodes[node]['boundary_perim'])/1000
                edgesLength[('dummy',node)] = float(graph.nodes[node]['boundary_perim'])/1000

    # Normal adjacency
    for e in graph.edges:
        if float(graph.edges[e]['shared_perim']) > 0:
            edgesLength[(e[0],e[1])] = float(graph.edges[e]['shared_perim'])/1000 
            edgesLength[(e[1],e[0])] = float(graph.edges[e]['shared_perim'])/1000
            
        else:
            edgesLength[(e[0],e[1])] = -1
            edgesLength[(e[1],e[0])] = -1

    
    # Add dummy node
    graph.add_node('dummy')
    graph.nodes['dummy']['GEOID20'] = '0'
    
    
    # Gather a list of neighbors and aug neighbors for every unit
    # Not using built-in Graph.neighbors(node) function because want separate normal and aug neighbors
    # without making two different graph objects and want dummy node
    neighborhoods = {}
    neighborhoods_aug = {}

    for node in graph.nodes:
        neighborhoods[node] = []
        neighborhoods_aug[node] = []
        
    neighborhoods['dummy'] = []
    neighborhoods_aug['dummy'] = []

    for pair in edgesLength:
        if edgesLength[pair] > 0:
            neighborhoods[pair[0]].append(pair[1])
            neighborhoods_aug[pair[0]].append(pair[1])
        elif edgesLength[pair] < 0:
            neighborhoods_aug[pair[0]].append(pair[1])
            
            
    for node in graph.nodes:
        graph.nodes[node]['neighbors'] = neighborhoods[node]
        graph.nodes[node]['neighbors_aug'] = neighborhoods_aug[node]

    
    # Calculate perimeter of each node (does NOT include state perimeter segments)
    perimeters = {}

    for node in graph.nodes:
        perimSum = 0
        for nb in graph.nodes[node]['neighbors']:
            if graph.nodes[nb]['GEOID20'] != '0':
                perimSum += edgesLength[(node,nb)]

        graph.nodes[node]['perimeter'] = perimSum
    
    
    
    if grid_bool:
        cutV = {}
    else:
        graph,edgesLength,cutV = merge_cut_vertices(graph,edgesLength)

      
    return graph,edgesLength,cutV



# Input: Allowed pop deviation, PLAN_VALUES object, Gerrychain graph object, adjacency dict,
#        dict of possible assignments for each node, column string to weight by ('UNWEIGHTED' for unweighted case),
#        beta fraction for beta-point
# Output: PLAN_VALUES object, objective value list
def get_district_info(eps,dvalues,graph,adj,possible,col,beta):
    
    obj_list = []
                
    # Gather units on the border of each district
    
    # borderNodes: List of all border nodes
    # borderNodesByDist: Records border nodes for each district (used to calculate initial district perimeters & auxH)
    # borderNodesYesNo: For every node, record whether it's a border node 
    #                   (can be adapted to record # of districts node is adjacent to, besides its own)

    for node in graph.nodes:
        dvalues.borderNodesYesNo[node] = 0

    for i in range(0,dvalues.numDistricts):
        dvalues.borderNodesByDist[i] = []

    for node in graph.nodes:
        dist = graph.nodes[node]['assignment']
        for nb in graph.nodes[node]['neighbors_aug']:
            otherDist = graph.nodes[nb]['assignment']
            if otherDist != dist:
                if node not in dvalues.borderNodesByDist[dist]:
                    dvalues.borderNodesByDist[dist].append(node)
                if otherDist != 0 and dist != 0 and adj[(node,nb)] > 0: # Don't want nodes to move to dummy district 0
                    dvalues.borderNodes.append(node)
                    dvalues.borderNodesYesNo[node] = 1
                    break # Exit neighborhood loop if node is already identified as on the border 
                          #(DON'T need to count units multiple times)

                    
    # Make auxiliary district graph for hole check
    for i in range(0,dvalues.numDistricts):
        for j in range(0,dvalues.numDistricts):
            dvalues.auxH[(i,j)] = 0

    for i in range(0,dvalues.numDistricts):
        for b in dvalues.borderNodesByDist[i]:
            for n in graph.nodes[b]['neighbors_aug']:
            #for n in graph.nodes[b]['neighbors']:
                if graph.nodes[n]['assignment'] != i:
                    dvalues.auxH[(i,graph.nodes[n]['assignment'])] += 1
                    
                    
    
    # Calculate population of each district
    dvalues.distPop = [0 for i in range(0,dvalues.numDistricts)]

    for node in graph.nodes:
        if node != 'dummy':
            dvalues.distPop[graph.nodes[node]['assignment']] += graph.nodes[node]['POP20']

    dvalues.meanDistPop = sum(dvalues.distPop)/((dvalues.numDistricts)-1) #numDistricts-1 bc don't want to include dummy district
    dvalues.distPop[0] = dvalues.meanDistPop # Change dummy districts population to the expected

    # Calculate each district's absolute percent difference from expected population
    dvalues.distPopPercent = [abs(1-(dp)/(dvalues.meanDistPop)) for dp in dvalues.distPop]

    print('Initial max population deviation: ',max(dvalues.distPopPercent))

            
        
    # Calculate district perimeters
    dvalues.distPerimeter = [0.0 for i in range(0,dvalues.numDistricts)]

    # Calculate state perimeter (remains constant throughout)
    statePerim = 0
    for pair in adj:
        if pair[0] == 'dummy' and adj[pair] > 0: # only need one, since edges are double-recorded
            statePerim += adj[pair]

    dvalues.distPerimeter[0] = statePerim # Set dummy district perimeter to state perimeter

    for i in range(1,dvalues.numDistricts):
        for b in dvalues.borderNodesByDist[i]:
            for n in graph.nodes[b]['neighbors']:
                if graph.nodes[n]['assignment'] != graph.nodes[b]['assignment'] and adj[(n,b)] > 0 and n != 'dummy':
                    dvalues.distPerimeter[i] += adj[(n,b)]

    print('Initial total perimeter: ',sum(dvalues.distPerimeter))
            
        
            
    # Calculate # cut edges
    count_cut = 0
    for pair in adj:
        if pair[0] != 'dummy' and pair[1] != 'dummy' and adj[pair] > 0:
            if graph.nodes[pair[0]]['assignment'] != graph.nodes[pair[1]]['assignment']:
                count_cut += 1

    dvalues.CutEdges = count_cut/2
    print('Initial total cut edges: ',dvalues.CutEdges)
    
    
    
    # Calculate number/weight of reassignment vertices for equidistance objective
    disagreement = 0
    
    if col == 'UNWEIGHTED':
        for n in possible:
            if len(possible[n]) > 1 and n in graph.nodes:
                disagreement += 1
                
    else:
        for n in possible:
            if len(possible[n]) > 1 and n in graph.nodes:
                disagreement += graph.nodes[n][col]
                
    if beta <= 0.5:
        dvalues.disagreeA = 0
        dvalues.disagreeB = disagreement
    else:
        dvalues.disagreeA = disagreement
        dvalues.disagreeB = 0
    
    obj_list.append(abs(dvalues.disagreeA-(beta/(1-beta))*dvalues.disagreeB))
    
    print('Initial disagreement: ',obj_list[0])
        
                    
    return obj_list,dvalues



# Checks contiguity of district distFrom upon removal of node n
# Input: node, N(node) intersect V(district(node)), N_aug(node) intersect V(district(node), Gerrychain graph object
# Output: Bool indicating whether or not distFrom remains contiguous upon removal of node
def checkContig(n,BOTH,BOTH_aug,graph):
    
    # Check contiguity of distFrom upon removal of chosen using a breadth-first search
    isPath = True
    x = BOTH[0] # Recall, nodes is N(n) intersect V(district(n))
    discovered = [x]
    bfsQ = col.deque()
    bfsQ.append(x)
    found = False
    while len(bfsQ) > 0:
        v = bfsQ.popleft()
        vNbhd = set(graph.nodes[v]['neighbors'])
        #vBoth = [x for x in vNbhd if data[x] == From] # For basic BFS
        vBoth = list(vNbhd.intersection(BOTH_aug)) # For aug-nbhd BFS (geo-graph)
        if n in vBoth:
            vBoth.remove(n)

        for w in vBoth:
            if w not in discovered:
                discovered.append(w)
                bfsQ.append(w)

    for y in BOTH:
        if y not in discovered:
            isPath = False
            break
            
    return isPath


# Checks if moving node n from district From to district To creates a hole
# Input: node, district From, district To, Gerrychain graph object, PLAN_VALUES object
# Output: Bool indicating whether or not flipping node creates hole
def checkHole(n,From,To,graph,dvalues):
    
    isPath = True
    disconnect = False

    # Increment if To and otherDist will have an additional node pair
    # Decrement if From and otherDist lose a node pair
    for vert in graph.nodes[n]['neighbors_aug']:
        otherDist = graph.nodes[vert]['assignment']
        if To != otherDist:
            dvalues.auxH[To,otherDist] += 1
            dvalues.auxH[otherDist,To] += 1

        if From != otherDist:
            dvalues.auxH[From,otherDist] -= 1
            dvalues.auxH[otherDist,From] -= 1
            
            # If edge deleted, signal for possible disconnect
            if (dvalues.auxH[From,otherDist] == 0) or (dvalues.auxH[otherDist,From] == 0):
                disconnect = True
                #numDisconnect += 1


    if disconnect:

        # Gather aux neighborhoods
        auxNeighborhoods = [[] for p in range(0,dvalues.numDistricts)]

        for i in range(0,dvalues.numDistricts):
            for j in range(0,dvalues.numDistricts):
                if dvalues.auxH[i,j] > 0:
                    auxNeighborhoods[i].append(j)
                    
        # Remove Too and all Too adjacencies
        auxNeighborhoods[To] = []

        for i in range(0,len(auxNeighborhoods)):
            temp = auxNeighborhoods[i]
            if To in temp:
                temp.remove(To)
                auxNeighborhoods[i] = temp

        # Check that new auxiliary graph H is connected
        isPath = True
        zones = [i for i in range(0,dvalues.numDistricts)]
        zones.remove(To)
        x = zones[0]
        discovered = [x]
        bfsQ = col.deque()
        bfsQ.append(x)
        found = False
        while len(bfsQ) > 0:
            v = bfsQ.popleft()
            auxNbhd = auxNeighborhoods[v]

            for w in auxNbhd:
                if w not in discovered:
                    discovered.append(w)
                    bfsQ.append(w)

        for y in zones:
            if y not in discovered:
                isPath = False
                break


        # Reject chosen node if moving it creates a hole/surrounded zone
        if not isPath:

            # Revert auxH back to previous
            for vert in graph.nodes[n]['neighbors_aug']:
                otherDist = graph.nodes[vert]['assignment']
                if To != otherDist:
                    dvalues.auxH[To,otherDist] -= 1 # Decrement here to undo the increment!
                    dvalues.auxH[otherDist,To] -= 1

                if From != otherDist:
                    dvalues.auxH[From,otherDist] += 1 # Increment here to undo the decrement!
                    dvalues.auxH[otherDist,From] += 1

                    

    return isPath



# SIMPLE HILL CLIMBING CHECK
# Checks that moving node n from district From to district To satisfies constraints and improves objective
# Input: district From, district To, chosen node, PLAN_VALUES object,
#        allowed population deviation, Gerrychain graph object, dict of possible assignments,
#        column string to weight by ('UNWEIGHTED' for unweighted case), current objective value
#        beta fraction for beta-point
# Output: updated values dict
def FlipCheck_simple(From,To,NODE,dvalues,eps,graph,possible,col,obj_current,beta):
    
    # Use bool to keep track of whether objective/constraints are satisfied
    checks = True
    
    # Record updates in case move is accepted
    updatedValues = {}
    
    # Determine intersection of N(chosen node)/N_aug(chosen node) and V(From)
    both = [n for n in graph.nodes[NODE]['neighbors'] if graph.nodes[n]['assignment'] == From]
    both_aug = [n for n in graph.nodes[NODE]['neighbors_aug'] if graph.nodes[n]['assignment'] == From]
    
    # Reject move if node is the only unit in its district
    if len(both) == 0:
        checks = False
        updatedValues['why'] = 'empty'

    # Check pop constraint
    if checks:

        # Determine new district populations
        newPopFrom = dvalues.distPop[From] - graph.nodes[NODE]['POP20']
        newPopTo = dvalues.distPop[To] + graph.nodes[NODE]['POP20']

        percentFrom = abs(1-(newPopFrom/dvalues.meanDistPop))
        percentTo = abs(1-(newPopTo/dvalues.meanDistPop))

        dev = max(percentFrom,percentTo)

        if dev > eps:
            checks = False
            updatedValues['why'] = 'pop'
        else:
            updatedValues['pop'] = [newPopFrom,newPopTo]


    # If previous checks pass, check perim objective/constraint
    if checks:
            
        perimFrom = 0
        perimTo = 0
        for nb in graph.nodes[NODE]['neighbors']:
            if graph.nodes[nb]['assignment'] == From:
                perimFrom += float(graph.edges[(NODE,nb)]['shared_perim'])/1000
                #perimFrom += adj[(NODE,nb)]
            elif graph.nodes[nb]['assignment'] == To:
                perimTo += float(graph.edges[(NODE,nb)]['shared_perim'])/1000
                #perimTo += adj[(NODE,nb)]

        newPerimFrom = dvalues.distPerimeter[From] + (2*perimFrom - graph.nodes[NODE]['perimeter'])
        newPerimTo = dvalues.distPerimeter[To] + (graph.nodes[NODE]['perimeter'] - 2*perimTo)

        newPerimeter = newPerimFrom + newPerimTo
        updatedValues['perim'] = [newPerimFrom,newPerimTo]

#         if 'perim' in param.constraints:
#             if (newPerimeter + sum(dvalues.distPerimeter) - dvalues.distPerimeter[From]
#                                              - dvalues.distPerimeter[To]) > param.perim_threshold:

#                 checks = False
#                 updatedValues['why'] = 'perim'
#             else:
#                 updatedValues['perim'] = [newPerimFrom,newPerimTo]

            
    # If previous checks pass, check cut edges objective/constraint
    if checks:
            
            cut_orig = []
            cut_new = []
            for nb in graph.nodes[NODE]['neighbors']:
                for u in graph.nodes[nb]['neighbors']:
                    if graph.nodes[nb]['assignment'] != graph.nodes[u]['assignment'] and (nb,u) not in cut_orig:
                        cut_orig.append((nb,u))
                        cut_orig.append((u,nb))
                    if nb == NODE:
                        if To != graph.nodes[u]['assignment'] and (nb,u) not in cut_new:
                            cut_new.append((nb,u))
                            cut_new.append((u,nb))
                    if u == NODE:
                        if To != graph.nodes[nb]['assignment'] and (nb,u) not in cut_new:
                            cut_new.append((nb,u))
                            cut_new.append((u,nb))
                    if (nb != NODE) and (u != NODE):
                        if graph.nodes[nb]['assignment'] != graph.nodes[u]['assignment'] and (nb,u) not in cut_new:
                            cut_new.append((nb,u))
                            cut_new.append((u,nb))

            cut_orig_val = len(cut_orig)/2
            cut_new_val = len(cut_new)/2

            newCutEdges = dvalues.CutEdges - cut_orig_val + cut_new_val
            updatedValues['cut_edges'] = newCutEdges

#             if 'cut_edges' in param.constraints:
#                 if newCutEdges > param.cut_edges_threshold:
#                     checks = False
#                     updatedValues['why'] = 'cut_edges'
#                 else:
#                     updatedValues['cut_edges'] = newCutEdges
                
    
    
    # If previous checks pass, check tight triangle constraints
    if checks:
        
        temp = [int(val) for val in possible[NODE]]
        
        if int(To) not in temp:
            checks = False
            updatedValues['why'] = 'triangle'
        
    # If previous checks pass, check equidistance objective
    if checks:
        
        temp = [int(val) for val in possible[NODE]]
        
        if col == 'UNWEIGHTED':
            if int(To) == temp[0]:
                new_disagreeA = dvalues.disagreeA - 1
                new_disagreeB = dvalues.disagreeB + 1
            else:
                new_disagreeB = dvalues.disagreeB - 1
                new_disagreeA = dvalues.disagreeA + 1
                
        else:
            if int(To) == temp[0]:
                new_disagreeA = dvalues.disagreeA - graph.nodes[NODE][col]
                new_disagreeB = dvalues.disagreeB + graph.nodes[NODE][col]
            else:
                new_disagreeB = dvalues.disagreeB - graph.nodes[NODE][col]
                new_disagreeA = dvalues.disagreeA + graph.nodes[NODE][col]
                
                
        if abs(new_disagreeA - (beta/(1-beta))*new_disagreeB) > abs(dvalues.disagreeA - (beta/(1-beta))*dvalues.disagreeB):
            checks = False
            updatedValues['why'] = 'equi'
        else:
            updatedValues['disagree'] = (new_disagreeA,new_disagreeB)
         
                    
                    
    # If previous checks pass, check contiguity -- mandatory!
    if checks:
        checks = checkContig(NODE,both,both_aug,graph)
        if not checks:
            updatedValues['why'] = 'contig'
                
                
    # If previous checks pass, check for hole -- mandatory!
    if checks:
        checks = checkHole(NODE,From,To,graph,dvalues)
        if not checks:
            updatedValues['why'] = 'hole'

        
    # Return values
    if checks:
        updatedValues['accept'] = True
    else:
        updatedValues['accept'] = False
        
    return updatedValues




# Input: PLAN_VALUES object, Gerrychain graph, dict of possible assignments
#        allowed population deviation, column string to weight by ('UNWEIGHTED' for unweighted case),
#        objective value list, beta fraction for beta-point
# Output: Gerrychain graph, objective value list, # iterations, run time
def perform_iterations(dvalues,graph,possible,eps,col,objective_values,beta):
    
    start = time.time()
    start_cpu = time.process_time()

    finished = False
    k = 0
    numIt = 0

    while not finished:

        # Give user idea of algorithm progress
        k += 1
        #print(k)

        # Check convergence, or complete a pre-determined number of iterations
        if k > 1:
            if (abs(objective_values[-2] - objective_values[-1]) < 1):
                numIt += 1
            else:
                numIt = 0

            if numIt >= 100:
                finished = True
                continue


        # Determine unit in consideration for move and its district
        good_choice = False
        while not good_choice:
        
            #print(len(dvalues.borderNodes))
            index = int(len(dvalues.borderNodes)*random.random())
            chosen_node = dvalues.borderNodes[index]
            distFrom = graph.nodes[chosen_node]['assignment']
            
            if len(possible[chosen_node]) > 1:
                good_choice = True

        # Choose district that chosen unit can move too
        candidates = []

        for nb in graph.nodes[chosen_node]['neighbors']:
            otherDist = graph.nodes[nb]['assignment']
            if nb != 'dummy' and otherDist != distFrom and otherDist not in candidates:
                candidates.append(otherDist)

        index = int(len(candidates)*random.random())
        distTo = candidates[index]

#                 print('\nNode: ',chosen_node)
#                 print('distFrom: ',distFrom)
#                 print('distTo: ',distTo)


        newValues = FlipCheck_simple(distFrom,distTo,chosen_node,dvalues,eps,graph,possible,col,objective_values[-1],beta)
                      

        # Once move is accepted/rejected, update everything
        if newValues['accept']:

            #print('Accepted!\n')

            # Update full record
            graph.nodes[chosen_node]['assignment'] = distTo 

            # Update population
                
            # Update district populations
            dvalues.distPop[distFrom] = newValues['pop'][0]
            dvalues.distPop[distTo] = newValues['pop'][1]

            # Update district percent difference from expected population
            dvalues.distPopPercent[distFrom] = abs(1-(dvalues.distPop[distFrom])/(dvalues.meanDistPop))
            dvalues.distPopPercent[distTo] = abs(1-(dvalues.distPop[distTo])/(dvalues.meanDistPop))

            # Update district perimeters
            dvalues.distPerimeter[distFrom] = newValues['perim'][0]
            dvalues.distPerimeter[distTo] = newValues['perim'][1]
                    
            # Update number of cut edges
            dvalues.CutEdges = newValues['cut_edges']
            
            # Update disagreement
            dvalues.disagreeA = newValues['disagree'][0]
            dvalues.disagreeB = newValues['disagree'][1]
            
            objective_values.append(abs(dvalues.disagreeA-(beta/(1-beta))*dvalues.disagreeB))

            # Update borderNodes
            toAppend = []
            for vert in graph.nodes[chosen_node]['neighbors']:
                isBorderNode = 0
                distVert = graph.nodes[vert]['assignment']
                for nb in graph.nodes[vert]['neighbors']:
                    otherDist = graph.nodes[nb]['assignment']
                    if otherDist != distVert and otherDist != 0 and distVert != 0:
                        if dvalues.borderNodesYesNo[vert] == 0:
                            toAppend.append(vert)
                        isBorderNode = 1
                        break

                dvalues.borderNodesYesNo[vert] = isBorderNode


            newBN = [bn for bn in dvalues.borderNodes if dvalues.borderNodesYesNo[bn] == 1]
            for bn in toAppend:
                newBN.append(bn)
                
            dvalues.borderNodes = [bn for bn in newBN]
            newBN = []
            

        else:

            temp = objective_values[-1]
            objective_values.append(temp)
            #print('Rejected! because ',newValues['why'])
            

    return graph, objective_values, k, time.time()-start, time.process_time()-start_cpu





# Input: Geopandas dataframe, assignment dictionaries for plans A and B,
#        dictionary of possible assignments for each node (transfer-tight),
#        allowed pop deviation, column string to weight by ('UNWEIGHTED' for unweighted case),
#        bool to indicate grid graph or not, beta fraction for beta-point
# Output: none
def FlipLocalSearch(df,planA,planB,possible,eps,col,grid_bool,beta):
    
    # Load state data
    print('\nLoading spatial data for local search warm-start now (shapefile may take a minute or two to load, depending on size)')
    print('...')
    graph,adj,cutV = make_graph(df,grid_bool)
    print('\nDone!\n')
    
    # Iterate through plans
    for i in range(0,1):
        
        if beta <= 0.5:
            init_plan = [planA]
        else:
            init_plan = [planB]
        
        for plan in init_plan:

            # Create PLAN_VALUES object to store district info
            dvalues = PLAN_VALUES()

            # Make all districts integers
            for n in plan:
                temp = int(plan[n])
                plan[n] = temp

            dvalues.numDistricts = max(plan.values()) + 1


            # Make assignment attribute for nodes in graph object
            for node in graph.nodes:
                if node == 'dummy':
                    graph.nodes[node]['assignment'] = 0
                else:
                    graph.nodes[node]['assignment'] = plan[graph.nodes[node]['GEOID20']]


            # Get district info
            obj_list,dvalues = get_district_info(eps,dvalues,graph,adj,possible,col,beta)

            print('\nRunning iterations now\n...')

            # Run iterations
            final_graph,final_obj,final_it,runtime,time_cpu = perform_iterations(dvalues,graph,possible,eps,
                                                                                 col,obj_list,beta)

            # Plot objective values
            plt.plot(final_obj)
            plt.show()
            print('Final disagreement = ',final_obj[-1])


            final_assignment = {}
            for node in graph.nodes:
                if node != 'dummy':
                    final_assignment[graph.nodes[node]['GEOID20']] = graph.nodes[node]['assignment']
                    #print('GEOID20: ',graph.nodes[node]['GEOID20'],'\tDISTRICT: ',graph.nodes[node]['assignment'])
                    if node in cutV:
                        for u in cutV[node]:
                            final_assignment[u] = graph.nodes[node]['assignment']
                            #print('GEOID20: ',u,'\tDISTRICT: ',graph.nodes[node]['assignment'])



    print('\n\nDone running plans!')
    
    return final_assignment