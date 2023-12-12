import csv
import distance
import gurobipy as gp
from gurobipy import GRB
import itertools as it
import pandas as pd
import geopandas as geopd
import matplotlib.pyplot as plt
import random
from gerrychain import Graph
import fliplocalsearch
import math
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


# Input: File path string for district plan
# Output: Assignment dictionary
def read_plan_from_csv(filename):
    
    plan_file = open(filename,'r')
    reader = csv.reader(plan_file,delimiter=',')

    labels = next(reader)
    plan = {}
    for line in reader:
        if line != [] and line != ['','']:
            plan[line[0]] = line[1]

    plan_file.close()
    
    return plan


# Input: Midpoint plan assignment dictionary, new dict for plan A, file names for plan B,
#        file name for shapefile, file name for midpoint plan, figure name for midpoint plan
# Output: None (outputs midpoint plan and midpoint figure)
def output_and_visualize_result(mid_plan,new_plan_A,file_B,shp,file_mid,fig_mid):

    # Convert midpoint plan to pandas dataframe and output to csv
    df_mid = pd.DataFrame.from_dict(mid_plan,orient='index',columns=['id','district'])
    #print(df_mid)
    df_mid.to_csv(file_mid,index=False)

    # Load permuted plan A into pandas
    for i in new_plan_A:
        new_plan_A[i] = (i,new_plan_A[i])

    df_A = pd.DataFrame.from_dict(new_plan_A,orient='index',columns=['id','district'])

    # Load original plan B into pandas
    df_B = pd.read_csv(file_B)

    # Load shapefile
    gdf = geopd.read_file(shp)

    if 'GEOID20' in gdf:
        mapping = {'GEOID20': 'id'}
        gdf = gdf.rename(columns=mapping)

    # Have to convert id columns to int
    gdf['id'] = gdf['id'].astype('int64')
    df_mid['id'] = df_mid['id'].astype('int64')
    df_A['id'] = df_A['id'].astype('int64')
    df_B['id'] = df_B['id'].astype('int64')

    # Join data - add district column to geodataframe
    gdf = gdf.merge(df_mid, on='id')
    gdf = gdf.rename(columns={'district':'district_mid'})
    gdf = gdf.merge(df_A, on='id')
    gdf = gdf.rename(columns={'district':'district_A'})
    gdf = gdf.merge(df_B, on='id')
    gdf = gdf.rename(columns={'district':'district_B'})
    #print(gdf)
    
    # Dissolve by districts
    gdf_diss_A = gdf.dissolve(by='district_A')
    gdf_diss_B = gdf.dissolve(by='district_B')
    gdf_diss_mid = gdf.dissolve(by='district_mid')

    # Plot plans
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(20,40))
    ax1 = gdf.plot(ax=ax1, column='district_A',edgecolor='k',linewidth=0.25,cmap='Paired')
    ax2 = gdf.plot(ax=ax2, column='district_mid',edgecolor='k',linewidth=0.25,cmap='Paired')
    ax3 = gdf.plot(ax=ax3, column='district_B',edgecolor='k',linewidth=0.25,cmap='Paired')
    
    ax1 = gdf_diss_A.plot(ax=ax1, column='id',edgecolor='k',linewidth=2,facecolor='None')
    ax2 = gdf_diss_mid.plot(ax=ax2, column='id',edgecolor='k',linewidth=2,facecolor='None')
    ax3 = gdf_diss_B.plot(ax=ax3, column='id',edgecolor='k',linewidth=2,facecolor='None')
    
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    #plt.savefig(fname=fig_mid,dpi=600,bbox_inches='tight')
    plt.show()

    # Save figures separately and in grayscale
    grayBig = cm.get_cmap('Greys', 512)
    newcmap = ListedColormap(grayBig(np.linspace(0, 0.4, 256)))
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
    ax = gdf.plot(ax=ax, column='district_A',edgecolor='k',linewidth=0.02,cmap=newcmap)
    ax = gdf_diss_A.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
    ax.axis('off')
    plt.savefig(fname=fig_mid+'PlanA.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
    ax = gdf.plot(ax=ax, column='district_mid',edgecolor='k',linewidth=0.02,cmap=newcmap)
    ax = gdf_diss_mid.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
    ax.axis('off')
    plt.savefig(fname=fig_mid+'PlanBeta.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
    ax = gdf.plot(ax=ax, column='district_B',edgecolor='k',linewidth=0.02,cmap=newcmap)
    ax = gdf_diss_B.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
    ax.axis('off')
    plt.savefig(fname=fig_mid+'PlanB.pdf',dpi=300,bbox_inches='tight')
    plt.close()



# Input: List of betapoint plan assignment dictionaries (which includes new_A), file name for plan B,
#        file name for shapefile, beginning of file name for betapoint plans (no extension),
#        beginning of figure name for betapoint plans
# Output: None (outputs betapoint plans and betapoint figures)
def output_and_visualize_result_sequence(beta_seq,file_B,shp,file_mid,fig_mid):

    # Load shapefile
    gdf = geopd.read_file(shp)
    
    if 'GEOID20' in gdf:
        mapping = {'GEOID20': 'id'}
        gdf = gdf.rename(columns=mapping)
        
    # Have to convert id columns to int
    gdf['id'] = gdf['id'].astype('int64')
    
    # Iterate through assignment dictionaries
    for i in range(0,len(beta_seq)):
        
        if i == 0:
            temp = {val:beta_seq[i][val] for val in beta_seq[i]}
            for n in temp:
                temp[n] = (n,temp[n])
            
            beta_seq[i] = {val:temp[val] for val in temp}
        
        # Convert plan to pandas dataframe and output to csv
        df_mid = pd.DataFrame.from_dict(beta_seq[i],orient='index',columns=['id','district'])
        #print(df_mid)
        if i == 0:
            df_mid.to_csv(file_mid + '_A.csv',index=False)
        else:
            df_mid.to_csv(file_mid + '_'+str(i)+'.csv',index=False)
            
        # Have to convert id columns to int
        df_mid['id'] = df_mid['id'].astype('int64')

        # Join data - add district column to geodataframe
        gdf = gdf.merge(df_mid, on='id')
        if i == 0:
            gdf = gdf.rename(columns={'district':'district_A'})
        else:
            gdf = gdf.rename(columns={'district':'district_beta_'+str(i)})


    # Load original plan B into pandas
    df_B = pd.read_csv(file_B)
    df_B.to_csv(file_mid + '_B.csv',index=False)

    # Have to convert id columns to int
    df_B['id'] = df_B['id'].astype('int64')

    # Join data - add district column to geodataframe
    gdf = gdf.merge(df_B, on='id')
    gdf = gdf.rename(columns={'district':'district_B'})
    #print(gdf)

    # Set grayscale colormap
    grayBig = cm.get_cmap('Greys', 512)
    newcmap = ListedColormap(grayBig(np.linspace(0, 0.4, 256)))

    # Plot plans
    for i in range(0,len(beta_seq)+1):
        
        if i == 0:
            gdf_diss = gdf.dissolve(by='district_A')
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
            ax = gdf.plot(ax=ax, column='district_A',edgecolor='k',linewidth=0.02,cmap=newcmap)
            #ax = gdf.plot(ax=ax, column='district_A',edgecolor='None',cmap=newcmap)
            ax = gdf_diss.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
            ax.axis('off')
            plt.savefig(fname=fig_mid + '_A.pdf',dpi=300,bbox_inches='tight')
            plt.show()
        elif i == len(beta_seq):
            gdf_diss = gdf.dissolve(by='district_B')
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
            ax = gdf.plot(ax=ax, column='district_B',edgecolor='k',linewidth=0.02,cmap=newcmap)
            #ax = gdf.plot(ax=ax, column='district_B',edgecolor='None',cmap=newcmap)
            ax = gdf_diss.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
            ax.axis('off')
            plt.savefig(fname=fig_mid + '_B.pdf',dpi=300,bbox_inches='tight')
            plt.show()
        else:
            gdf_diss = gdf.dissolve(by='district_beta_'+str(i))
            fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(2,2))
            ax = gdf.plot(ax=ax, column='district_beta_'+str(i),edgecolor='k',linewidth=0.02,cmap=newcmap)
            #ax = gdf.plot(ax=ax, column='district_beta_'+str(i),edgecolor='None',cmap=newcmap)
            ax = gdf_diss.plot(ax=ax, column='id',edgecolor='k',linewidth=1,facecolor='None')
            ax.axis('off')
            plt.savefig(fname=fig_mid + '_beta_'+str(i)+'.pdf',dpi=300,bbox_inches='tight')
            plt.show()
        



# Input: GerryChain graph object, file path strings for district plans A & B,
#        column string to weight on ('UNWEIGHTED' for unweighted case)
# Output: Transfer distance, district centers, assignment possibility dictionary, 
#         empty core Boolean (True if there are empty cores), re-labeled plan A dict
def MIP_prep(graph,filename_A,filename_B,col):
    
    plan_A = read_plan_from_csv(filename_A)
    plan_B = read_plan_from_csv(filename_B)
    
    TD,matching = distance.TransferDistance(plan_A,plan_B,graph,col,False)
    print('\n\nTransfer distance: ',TD)

    print('Matching: ',matching)
    
    # District re-labeling
    for n in plan_A:
        temp = plan_A[n]
        plan_A[n] = matching[temp]

    # Determine number of districts
    # (assumes district labels are consecutive integers, lowest being 1)
    K = int(max(plan_A.values()))
    #print(K)

    # Make assignment dictionaries for nodes in graph object
    plan_A_nodes = {}
    plan_B_nodes = {}

    for node in graph.nodes:
        plan_A_nodes[node] = plan_A[graph.nodes[node]['GEOID20']]
        plan_B_nodes[node] = plan_B[graph.nodes[node]['GEOID20']]

    # Determine fixed sets
    fixed = {}
    for i in range(1,K+1):
        fixed[str(i)] = []

    for n in plan_A_nodes:
        if plan_A_nodes[n] == plan_B_nodes[n]:
            fixed[plan_A_nodes[n]].append(n)

    # for f in fixed:
    #     print(f,' : ',fixed[f])

    # Check for empty or zero-weight fixed cores
    empty_cores = False
    
    if col == 'UNWEIGHTED':
        for f in fixed:
            if fixed[f] == []:
                empty_cores = True
    else:
        for f in fixed:
            if fixed[f] == []:
                empty_cores = True
            else:
                temp = 0
                for n in fixed[f]:
                    temp += graph.nodes[n][col]
                    
                if temp == 0:
                    empty_cores = True
            

    print('\nAre there any empty/zero-weight cores? ',empty_cores)

    # Choose centers from nonempty cores
    centers = {}
    
    if col == 'UNWEIGHTED':
        for f in fixed:
            if fixed[f] == []:
                centers[f] = 'EMPTY'
            else:
                centers[f] = fixed[f][0]
    else:
        for f in fixed:
            if fixed[f] == []:
                centers[f] = 'EMPTY'
            else:
                temp = 0
                for n in fixed[f]:
                    temp += graph.nodes[n][col]
                    if graph.nodes[n][col] > 0:
                        centers[f] = n
                        break
                        
                if temp == 0:
                    centers[f] = 'EMPTY'
                

    #print(centers)

    # Determine possible assignments for each node (1 option for fixed, 2 for reassignment)
    possible = {}
    for n in plan_A_nodes:
        if plan_A_nodes[n] == plan_B_nodes[n]:
            possible[n] = (plan_A_nodes[n])
        else:
            possible[n] = (plan_A_nodes[n],plan_B_nodes[n])

    #print(possible)
    
    return TD,centers,possible,empty_cores,plan_A,plan_B


# Input: Dictionary for district centers, dictionary for possible node assignments
# Output: Plan A MIP assignment dictionary, plan B MIP assignment dictionary, reassignment dictionary
#
# PRELIMINARY TEST FUNCTION
def MIP_objective_prep_Hess(centers,possible):
    
    A_MIP = {}
    B_MIP = {}
    R_MIP = {}
    
    cartesian_prod = list(it.product(centers.keys(), possible.keys()))
    
    for c,u in cartesian_prod:
        if len(possible[u]) == 1:
            if possible[u][0] == c:
                A_MIP[(centers[c],u)] = 1
                B_MIP[(centers[c],u)] = 1
            else:
                A_MIP[(centers[c],u)] = 0
                B_MIP[(centers[c],u)] = 0
                
        else:
            if possible[u][0] == c:
                A_MIP[(centers[c],u)] = 1
                B_MIP[(centers[c],u)] = 0
            elif possible[u][1] == c:
                A_MIP[(centers[c],u)] = 0
                B_MIP[(centers[c],u)] = 1
            else:
                A_MIP[(centers[c],u)] = 0
                B_MIP[(centers[c],u)] = 0
                
        R_MIP[(centers[c],u)] = A_MIP[(centers[c],u)] - B_MIP[(centers[c],u)]
        
    
    return A_MIP,B_MIP,R_MIP


# Input: Dictionary for district centers, dictionary for possible node assignments
# Output: Plan A MIP assignment dictionary, plan B MIP assignment dictionary, reassignment dictionary
def MIP_objective_prep_Labeling(centers,possible):
    
    A_MIP = {}
    B_MIP = {}
    R_MIP = {}
    
    cartesian_prod = list(it.product(centers.keys(), possible.keys()))
    
    for k,u in cartesian_prod:
        if len(possible[u]) == 1:
            if possible[u][0] == k:
                A_MIP[(k,u)] = 1
                B_MIP[(k,u)] = 1
            else:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 0
                
        else:
            if possible[u][0] == k:
                A_MIP[(k,u)] = 1
                B_MIP[(k,u)] = 0
            elif possible[u][1] == k:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 1
            else:
                A_MIP[(k,u)] = 0
                B_MIP[(k,u)] = 0
                
        R_MIP[(k,u)] = A_MIP[(k,u)] - B_MIP[(k,u)]
        
    
    return A_MIP,B_MIP,R_MIP



# Input: GerryChain graph object, dictionary of district centers, and Gurobi midpoint assignment variables
# Output: Plan assignment dictionary for minimizer (approximate midpoint)
#
# PRELIMINARY TEST FUNCTION
def solution_to_plan_Hess(graph,centers,assign_vars):

    midpoint_assign = {}
    
    for u in graph.nodes:
        for c in centers:
            if (abs(assign_vars[(centers[c],u)].x) > 1e-6):
                #print(f'\n Assign {u} to district with center {centers[c]}.')
                midpoint_assign[graph.nodes[u]['GEOID20']] = (graph.nodes[u]['GEOID20'],c)
    
    return midpoint_assign


# Input: GerryChain graph object, dictionary of district centers, and Gurobi midpoint assignment variables
# Output: Plan assignment dictionary for minimizer (approximate midpoint)
def solution_to_plan_Labeling(graph,centers,assign_vars):

    midpoint_assign = {}
    
    for u in graph.nodes:
        for k in centers:
            #if (abs(assign_vars[(k,u)].x) > 1e-6):
            if (round(abs(assign_vars[(k,u)].x)) > 0.5):
                #print(f'\n Assign {u} to district {k}.')
                midpoint_assign[graph.nodes[u]['GEOID20']] = (graph.nodes[u]['GEOID20'],k)
    
    return midpoint_assign


# Input: GerryChain graph object, dictionary for district centers, 
#        dictionary for possible node assignments, max allowed balance % deviation
# Output: Minimum value of |T(A,C)-T(C,B)|, plan assignment dictionary for minimizer
#
# PRELIMINARY TEST FUNCTION
def MIP_optimize_Hess(graph,centers,possible,eps):
    
    # Compute parameters
    num_districts = len(centers.keys())
    num_units = len(graph.nodes)
    cartesian_prod = list(it.product(centers.values(), graph.nodes))
    
    # MIP  model formulation
    model = gp.Model('Hess')

    # Primary decision variables (x_ij)
    assign = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')
    
    
    # Basic facility location constraints
    
    # Only assign unit u to district center c if c is actually a district center
    model.addConstrs((assign[(c,u)] <= assign[(c,c)] for c,u in cartesian_prod), name='OnlyAssignToCenter')
    
    # Every unit must be assigned to some district center c
    model.addConstrs((gp.quicksum(assign[(c,u)] for c in centers.values()) == 1 for u in graph.nodes), name='AllAssigned')
    
    # There must be the desired number of districts
    model.addConstr(gp.quicksum(assign[(c,c)] for c in centers.values()) == num_districts, name='CorrectNumDist')
    
    
    # District constraints
    
    # Population balance
    P_bar = 0
    for u in graph.nodes:
        P_bar += graph.nodes[u]['POP20']
    P_bar = float(P_bar)/num_districts
    print('\nIdeal district size: ',P_bar,'\n\n')
    
    model.addConstrs(((1-eps)*P_bar*assign[(c,c)] <= 
                      gp.quicksum(assign[c,u]*graph.nodes[u]['POP20'] for u in graph.nodes) 
                      for c in centers.values()), name='Pop_LB')
    model.addConstrs((gp.quicksum(assign[c,u]*graph.nodes[u]['POP20'] for u in graph.nodes) 
                      <= (1+eps)*P_bar*assign[(c,c)] for c in centers.values()), name='Pop_UB')
    
    # Contiguity
    three_prod = []
    for c in centers.values():
        for u in graph.nodes:
            for v in graph.neighbors(u):
                three_prod.append((c,u,v))
                
    # Flow variables
    flow = model.addVars(three_prod, vtype=GRB.CONTINUOUS, name='Flow')
    
    # Flow balance
    for c in centers.values():
        for u in graph.nodes:
            if u != c:
                model.addConstr(gp.quicksum(flow[(c,v,u)] - flow[(c,u,v)] for v in list(graph.neighbors(u)))
                                == assign[(c,u)])
    
    # Flow into u from c
    for c in centers.values():
        for u in graph.nodes:
            if u != c:
                model.addConstr(gp.quicksum(flow[(c,v,u)] for v in list(graph.neighbors(u)))
                                <= (num_units - 1)*assign[(c,u)])
                
    # Flow conservation
    for c in centers.values():
        model.addConstr(gp.quicksum(flow[(c,v,c)] for v in list(graph.neighbors(c))) == 0)
           
    
    # Tight triangle inequality constraints
    
    for u in graph.nodes:
        for c in set(centers.keys()).difference(set(possible[u])):
            model.addConstr(assign[(centers[c],u)] == 0)
            
    for u in graph.nodes:
        if len(possible[u]) == 1:
            model.addConstr(assign[(centers[possible[u]],u)] == 1)
    
    
    # Objective
    
    A_assign,B_assign,R_diff = MIP_objective_prep_Hess(centers,possible)
    
    # Variable (d) to linearize objective
    linearize_d = model.addVar(vtype=GRB.INTEGER, name='D')
    
    # Constraints to linearize absolute value objective
    model.addConstr(-linearize_d <= gp.quicksum(gp.quicksum(R_diff[(c,u)]*(A_assign[(c,u)] - assign[(c,u)]) 
                                                            for u in graph.nodes) for c in centers.values())
                    - gp.quicksum(gp.quicksum(R_diff[(c,u)]*(A_assign[(c,u)] + assign[(c,u)] - 1) for u in graph.nodes)
                                  for c in centers.values()), name='Linearize_D_LB')
    model.addConstr(gp.quicksum(gp.quicksum(R_diff[(c,u)]*(A_assign[(c,u)] - assign[(c,u)]) for u in graph.nodes) 
                                for c in centers.values())
                    - gp.quicksum(gp.quicksum(R_diff[(c,u)]*(A_assign[(c,u)] + assign[(c,u)] - 1) for u in graph.nodes) 
                                  for c in centers.values()) <= linearize_d, name='Linearize_D_UB')
    
    # Set objective
    model.setObjective(0.5*linearize_d, GRB.MINIMIZE)
    
    # Solve and gather solution
    model.optimize()
    
    #print('\n\nObjective value: ',model.objVal)
    
    assignment_dict = solution_to_plan_Hess(graph,centers,assign)
            
    return model.objVal,assignment_dict


# Input: GerryChain graph object, dictionary for district centers, 
#        dictionary for possible node assignments, max allowed balance % deviation,
#        bool to also optimize for compactness after or not, column string to weight by
#        ('UNWEIGHTED' for unweighted case), beta fraction, plan assignment dictionary for initial solution (0 if none)
# Output: Minimum value of |T(A,C)-T(C,B)|, plan assignment dictionary for minimizer,
#         true value of beta for which resulting plan is a beta-point
def MIP_optimize_Labeling(graph,centers,possible,eps,comp_bool,col,beta,warm_assignment):
    
    # Compute parameters
    num_districts = len(centers.keys())
    num_units = len(graph.nodes)
    cartesian_prod = list(it.product(centers.keys(), graph.nodes))
    
    # MIP  model formulation
    model = gp.Model('Labeling')

    # Primary decision variables (x_ku)
    assign = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')
    
    # Flow variables
    source = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Source')
    amount = model.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Amount')
    
    # Basic facility location constraints
    
    # Every unit u must be assigned to some district k
    model.addConstrs((gp.quicksum(assign[(k,u)] for k in centers.keys()) == 1 for u in graph.nodes), name='AllAssigned')
    
    # Every district k can only have one flow source u
    model.addConstrs((gp.quicksum(source[(k,u)] for u in graph.nodes) == 1 for k in centers.keys()), name='OnlyOneSource')
    
    # Only assign unit u to be flow source of district center k if u is actually assigned to k
    model.addConstrs((source[(k,u)] <= assign[(k,u)] for k,u in cartesian_prod), name='OnlySourceIfAssigned')

    
    # District constraints
    
    # Weight balance (typically pop)
    if col == 'UNWEIGHTED':
        
        W_bar = float(num_units)/num_districts
        print('\nIdeal district size: ',W_bar,'\n\n')
        
        model.addConstrs(((1-eps)*W_bar <= gp.quicksum(assign[k,u] for u in graph.nodes) 
                          for k in centers.keys()), name='Weight_LB')
        model.addConstrs((gp.quicksum(assign[k,u] for u in graph.nodes) <= (1+eps)*W_bar 
                          for k in centers.keys()), name='Weight_UB')
        
    else:
        
        W_bar = 0
        for u in graph.nodes:
            W_bar += graph.nodes[u][col]
        W_bar = float(W_bar)/num_districts
        print('\nIdeal district size: ',W_bar,'\n\n')
    
        model.addConstrs(((1-eps)*W_bar <= 
                          gp.quicksum(assign[k,u]*graph.nodes[u][col] for u in graph.nodes) 
                          for k in centers.keys()), name='Weight_LB')
        model.addConstrs((gp.quicksum(assign[k,u]*graph.nodes[u][col] for u in graph.nodes) 
                          <= (1+eps)*W_bar for k in centers.keys()), name='Weight_UB')


    # Compute big M value
    if col == 'UNWEIGHTED':
        
        bigM = int((1+eps)*W_bar)
        
    else:
        
        weight_list = [graph.nodes[n][col] for n in graph.nodes]
        weight_list.sort()
        bigM = 0
        total = 0
        
        for val in weight_list:
            total += val
            if total > ((1+eps)*W_bar):
                break
            else:
                bigM += 1
                
                
    print('\nBig M value: ',bigM,'\n')

    
    # Contiguity
    three_prod = []
    for k in centers.keys():
        for u in graph.nodes:
            for v in graph.neighbors(u):
                three_prod.append((k,u,v))
                
    # More flow variables
    flow = model.addVars(three_prod, vtype=GRB.CONTINUOUS, name='Flow')
    
    # Amount of flow generated
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(amount[(k,u)] <= (bigM+1)*source[(k,u)], name='Flow_Gen_'+str(k)+'_'+str(u))
    
    # Flow balance
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(gp.quicksum(flow[(k,v,u)] - flow[(k,u,v)] for v in list(graph.neighbors(u)))
                            == assign[(k,u)] - amount[(k,u)], name='Flow_Bal_'+str(k)+'_'+str(u))
    
    # Flow into u from k
    for k in centers.keys():
        for u in graph.nodes:
                model.addConstr(gp.quicksum(flow[(k,v,u)] for v in list(graph.neighbors(u)))
                                <= (bigM)*(assign[(k,u)] - source[(k,u)]), name='Flow_In_'+str(k)+'_'+str(u))
                  
        
    
    # Tight triangle inequality constraints
    
    # Restrict part assignments (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            for k in set(centers.keys()).difference(set(possible[u])):
                model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if graph.nodes[u][col] > 0:
                for k in set(centers.keys()).difference(set(possible[u])):
                    model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                    model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
            
    # Keep fixed cores intact (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            if len(possible[u]) == 1:
                model.addConstr(assign[(possible[u],u)] == 1, name='Core_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if len(possible[u]) == 1 and graph.nodes[u][col] > 0:
                model.addConstr(assign[(possible[u],u)] == 1, name='Core_'+str(k)+'_'+str(u))
            
            
    # Set sources for nonempty cores
    for k in centers.keys():
        if centers[k] != 'EMPTY':
            model.addConstr(source[(k,centers[k])] == 1, name='Source_'+str(k)+'_'+str(u))
    
    
    # Objective
    
    A_assign,B_assign,R_diff = MIP_objective_prep_Labeling(centers,possible)
    
    # Variable (d) to linearize objective
    linearize_d = model.addVar(vtype=GRB.CONTINUOUS, name='D')
    
    # Constraints to linearize absolute value objective
    if col == 'UNWEIGHTED':
        model.addConstr(-linearize_d <= gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1)
                                                                 for u in graph.nodes)
                                      for k in centers.keys()), name='Linearize_D_LB')
        model.addConstr(gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) for u in graph.nodes) 
                                    for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                 for u in graph.nodes) 
                                      for k in centers.keys()) <= linearize_d, name='Linearize_D_UB')
        
    else:
        model.addConstr(-linearize_d <= gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1)
                                                                 for u in graph.nodes)
                                      for k in centers.keys()), name='Linearize_D_LB')
        model.addConstr(gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                 for u in graph.nodes) 
                                      for k in centers.keys()) <= linearize_d, name='Linearize_D_UB')
    
    # Set objective
    model.setObjective(0.5*linearize_d, GRB.MINIMIZE)


#     for k in centers.keys():
#         print(type(k))
    
    # Enter warm-start initial values, if applicable
    if type(warm_assignment) != type(0):
        #print('warm start:\n',warm_assignment)
        for u in graph.nodes:
            assign[(str(warm_assignment[graph.nodes[u]['GEOID20']]),u)].Start = 1

    
#     # Find multiple optimal solutions
#     model.Params.PoolSearchMode = 2
#     model.Params.PoolSolutions = 10
    
    
    # Solve and gather solution
    model.optimize()
    
    #print('\n\nObjective value: ',model.objVal)
    
    
#     # Collect optimal solutions
#     assign_list = []
#     for count in range(model.SolCount):
#         model.Params.SolutionNumber = count
#         assign_list.append(solution_to_plan_Labeling(graph,centers,assign))

    
    assignment_dict = solution_to_plan_Labeling(graph,centers,assign)


    print('Objective value: ',model.objVal)
    print('d value: ',linearize_d.x)
    
    obj_exp_one = 0
    obj_exp_two = 0

    if col == 'UNWEIGHTED':
        for k in centers.keys():
            for u in graph.nodes:
                #print('assign[('+str(k)+','+str(u)+')]: ',assign[(k,u)].x)
                obj_exp_one += R_diff[(k,u)]*(A_assign[(k,u)] - round(abs(assign[(k,u)].x)))
                obj_exp_two += R_diff[(k,u)]*(A_assign[(k,u)] + round(abs(assign[(k,u)].x)) - 1)
    else:
        for k in centers.keys():
            for u in graph.nodes:
                #print('assign[('+str(k)+','+str(u)+')]: ',assign[(k,u)].x)
                obj_exp_one += graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - round(abs(assign[(k,u)].x)))
                obj_exp_two += graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + round(abs(assign[(k,u)].x)) - 1)

                                                                     
    print('obj expression 1: ',obj_exp_one)
    print('obj expression 2: ',obj_exp_two)
    obj_exp = obj_exp_one - (beta/(1-beta))*obj_exp_two
    print('obj expression: ',obj_exp)


    true_beta = float(obj_exp_one/(obj_exp_one + obj_exp_two))
    print('true_beta: ',true_beta)

    
    if comp_bool:
    
        # Optimize for compactness
        cut_edges_obj,assignment_comp_dict = MIP_optimize_compactness_Labeling(graph,centers,possible,eps,col,beta,
                                                                               assignment_dict,abs(obj_exp),0,bigM)
        
        return model.objVal,assignment_comp_dict,true_beta
    
    else:
        return model.objVal,assignment_dict,true_beta

    
    
# Input: GerryChain graph object, dictionary for district centers, 
#        dictionary for possible node assignments, max allowed balance % deviation,
#        column string to weight by ('UNWEIGHTED' for unweighted case), beta fraction
#        plan assignment dictionary for initial solution, distance objective value for initial solution,
#        amount of slack allowed for constraints maintaining (approximate) equidistance,
#        big M value for contiguity constraints already calculated when finding midpoint
# Output: Minimum value of cut edges compactness metric, plan assignment dictionary for minimizer
def MIP_optimize_compactness_Labeling(graph,centers,possible,eps,col,beta,warm_assignment,warm_obj_val,dist_slack,bigM):
    
    # Compute parameters
    num_districts = len(centers.keys())
    num_units = len(graph.nodes)
    cartesian_prod = list(it.product(centers.keys(), graph.nodes))
    
    # MIP  model formulation
    model = gp.Model('Labeling')

    # Primary decision variables (x_ku)
    assign = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Assign')
    
    # Flow variables
    source = model.addVars(cartesian_prod, vtype=GRB.BINARY, name='Source')
    amount = model.addVars(cartesian_prod, vtype=GRB.CONTINUOUS, name='Amount')
    
    # Basic facility location constraints

    # Every unit u must be assigned to some district k
    model.addConstrs((gp.quicksum(assign[(k,u)] for k in centers.keys()) == 1 for u in graph.nodes), name='AllAssigned')
    
    # Every district k can only have one flow source u
    model.addConstrs((gp.quicksum(source[(k,u)] for u in graph.nodes) == 1 for k in centers.keys()), name='OnlyOneSource')
    
    # Only assign unit u to be flow source of district center k if u is actually assigned to k
    model.addConstrs((source[(k,u)] <= assign[(k,u)] for k,u in cartesian_prod), name='OnlySourceIfAssigned')

    
    # District constraints
    
    # Weight balance (typically pop)
    if col == 'UNWEIGHTED':
        
        W_bar = float(num_units)/num_districts
        print('\nIdeal district size: ',W_bar,'\n\n')
        
        model.addConstrs(((1-eps)*W_bar <= gp.quicksum(assign[k,u] for u in graph.nodes) 
                          for k in centers.keys()), name='Weight_LB')
        model.addConstrs((gp.quicksum(assign[k,u] for u in graph.nodes) <= (1+eps)*W_bar 
                          for k in centers.keys()), name='Weight_UB')
        
    else:
        
        W_bar = 0
        for u in graph.nodes:
            W_bar += graph.nodes[u][col]
        W_bar = float(W_bar)/num_districts
        print('\nIdeal district size: ',W_bar,'\n\n')
    
        model.addConstrs(((1-eps)*W_bar <= 
                          gp.quicksum(assign[k,u]*graph.nodes[u][col] for u in graph.nodes) 
                          for k in centers.keys()), name='Weight_LB')
        model.addConstrs((gp.quicksum(assign[k,u]*graph.nodes[u][col] for u in graph.nodes) 
                          <= (1+eps)*W_bar for k in centers.keys()), name='Weight_UB')

    
    # Contiguity
    three_prod = []
    for k in centers.keys():
        for u in graph.nodes:
            for v in graph.neighbors(u):
                three_prod.append((k,u,v))
                
    # More flow variables
    flow = model.addVars(three_prod, vtype=GRB.CONTINUOUS, name='Flow')
    
    # Amount of flow generated
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(amount[(k,u)] <= (bigM+1)*source[(k,u)], name='Flow_Gen_'+str(k)+'_'+str(u))
    
    # Flow balance
    for k in centers.keys():
        for u in graph.nodes:
            model.addConstr(gp.quicksum(flow[(k,v,u)] - flow[(k,u,v)] for v in list(graph.neighbors(u)))
                            == assign[(k,u)] - amount[(k,u)], name='Flow_Bal_'+str(k)+'_'+str(u))
    
    # Flow into u from k
    for k in centers.keys():
        for u in graph.nodes:
                model.addConstr(gp.quicksum(flow[(k,v,u)] for v in list(graph.neighbors(u)))
                                <= (bigM)*(assign[(k,u)] - source[(k,u)]), name='Flow_In_'+str(k)+'_'+str(u))
                
    
    # Tight triangle inequality constraints
    
    # Restrict part assignments (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            for k in set(centers.keys()).difference(set(possible[u])):
                model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if graph.nodes[u][col] > 0:
                for k in set(centers.keys()).difference(set(possible[u])):
                    model.addConstr(assign[(k,u)] == 0, name='Triangle_'+str(k)+'_'+str(u))
                    model.addConstr(source[(k,u)] == 0, name='NoSource_'+str(k)+'_'+str(u))
            
    # Keep fixed cores intact (for vertices with positive weight)
    if col == 'UNWEIGHTED':
        for u in graph.nodes:
            if len(possible[u]) == 1:
                model.addConstr(assign[(possible[u],u)] == 1, name='Core_'+str(k)+'_'+str(u))
    else:
        for u in graph.nodes:
            if len(possible[u]) == 1 and graph.nodes[u][col] > 0:
                model.addConstr(assign[(possible[u],u)] == 1, name='Core_'+str(k)+'_'+str(u))
            
            
    # Set sources for nonempty cores
    for k in centers.keys():
        if centers[k] != 'EMPTY':
            model.addConstr(source[(k,centers[k])] == 1, name='Source_'+str(k)+'_'+str(u))
    
    
    # Distance constraint
    
    A_assign,B_assign,R_diff = MIP_objective_prep_Labeling(centers,possible)

    # Safeguard check for distance slack
    if dist_slack < 0:
        dist_slack = 0
    
    # Constraints to maintain (approximate) equidistance
    if col == 'UNWEIGHTED':
        model.addConstr(-2*(warm_obj_val+dist_slack) <= gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                   for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                  for u in graph.nodes)
                                      for k in centers.keys()), name='Distance_LB')
        model.addConstr(gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) for u in graph.nodes) 
                                    for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                  for u in graph.nodes) 
                                      for k in centers.keys()) <= 2*(warm_obj_val+dist_slack), name='Distance_UB')
    
    else:
        model.addConstr(-2*(warm_obj_val+dist_slack) <= gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                                   for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                  for u in graph.nodes)
                                      for k in centers.keys()), name='Distance_LB')
        model.addConstr(gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] - assign[(k,u)]) 
                                                for u in graph.nodes) for k in centers.keys())
                        - (beta/(1-beta))*gp.quicksum(gp.quicksum(graph.nodes[u][col]*R_diff[(k,u)]*(A_assign[(k,u)] + assign[(k,u)] - 1) 
                                                                  for u in graph.nodes) 
                                      for k in centers.keys()) <= 2*(warm_obj_val+dist_slack), name='Distance_UB')
        

    
    # Compactness variables
    cut_Y = model.addVars(list(graph.edges), vtype=GRB.BINARY, name='Cut_Edges')
    
    edges_prod = []
    for k in centers.keys():
        for e in graph.edges:
            edges_prod.append((k,e))
    
    cut_Z = model.addVars(edges_prod, vtype=GRB.CONTINUOUS, name='Cut_Edges_Z')
    
    # Compactness constraints
    
    # Cut if endpoints are not both in district k
    for k in centers.keys():
        for e in graph.edges:
            model.addConstr(assign[(k,e[0])] - assign[(k,e[1])] <= cut_Z[(k,e)], name='CutEndpts_'+str(k)+'_'+str(e))
            
    # Cut if cut with respect to some district
    for e in graph.edges:
        model.addConstr(gp.quicksum(cut_Z[(k,e)] for k in centers.keys()) == cut_Y[e], name='CutDist_'+str(e))
    
    
    
    # Set objective (minimize number of cut edges)
    model.setObjective(gp.quicksum(cut_Y[e] for e in graph.edges), GRB.MINIMIZE)
    
    
    # Enter warm-start initial values
    for u in graph.nodes:
        assign[(warm_assignment[graph.nodes[u]['GEOID20']][1],u)].Start = 1
    
    
    # Solve and gather solution
    model.optimize()
    
    #print('\n\nObjective value: ',model.objVal)
    
    
    assignment_dict = solution_to_plan_Labeling(graph,centers,assign)
           
        
    return model.objVal,assignment_dict



# Input: File path strings for shapefile and district assignment files A & B,
#        max allowed fractional pop deviation, bool to use local search warm-start or not,
#        bool to use initial plan warm-start or not, bool to also optimize for compactness or not,
#        column string to weight by, beta fraction (e.g., 0.5 for midpoint), bool to indicate grid graph or not
# Output: Initial transfer dist, minimum value of |T(A,C)-T(C,B)|,
#         plan assignment dictionary for minimizer (approximate midpoint),
#         true value of beta for which resulting plan is a beta-point
def TD_midpoint(shp,filename_A,filename_B,epsilon,warm_LS,warm_Init,comp_bool,col,beta,grid_bool):
    
    # Read in shapefile with geopandas
    df = geopd.read_file(shp)

    # Create Graph object from gerrychain
    graph = Graph.from_geodataframe(df)
    
    # Give node attributes, if grid
    if grid_bool:
        for n in graph:
            graph.nodes[n]['POP20'] = 1
            graph.nodes[n]['GEOID20'] = str(int(graph.nodes[n]['id']))

    print('\nShapefile loaded\n')
    
    transfer_dist,centers,possible,empty,new_plan_A,new_plan_B = MIP_prep(graph,filename_A,filename_B,col)
    
    # Obtain warm-start solution, if applicable
    if warm_LS:
        
        print('\nLocal search warm start!\n')

        # Perform local search
        warm_start_assignment = fliplocalsearch.FlipLocalSearch(df,new_plan_A,new_plan_B,possible,epsilon,col,grid_bool,beta)
        
    elif warm_Init:
        
        print('\nEndpoint plan warm start!\n')
        
        warm_start_assignment = {}
        
        if beta <= 0.5:
            for val in new_plan_A:
                warm_start_assignment[val] = new_plan_A[val]
        else:
            for val in new_plan_B:
                warm_start_assignment[val] = new_plan_B[val]
        
    else:
        warm_start_assignment = 0

    #print('warm start:\n',warm_start_assignment)

    obj_value,assignment,true_beta = MIP_optimize_Labeling(graph,centers,possible,epsilon,comp_bool,col,beta,warm_start_assignment)
    
    return transfer_dist,obj_value,assignment,new_plan_A,true_beta


# Input: File path strings for shapefile and district assignment files A & B, string to use for betapoint filenames,
#        max allowed fractional pop deviation, bool to use local search warm-start or not,
#        bool to use initial plan warm-start or not, bool to also optimize for compactness or not,
#        column string to weight by, sorted list of beta fractions (e.g., 0.5 for midpoint),
#        bool to indicate grid graph or not, int for initial recursion level, int for current recursion level
# Output: Initial transfer dist, minimum value of |T(A,C)-T(C,B)|,
#         plan assignment dictionary for minimizer (approximate midpoint)
def TD_sequence(shp,filename_A,filename_B,filename_seq,epsilon,warm_LS,warm_Init,comp_bool,col,beta_list,grid_bool,L,level):
    
    if len(beta_list) == 0:
        
        return []
    
    else:
        
        # Choose middle beta value
        index = math.ceil((len(beta_list)-1)/2)
        beta = beta_list[index]
        
        # Find beta-point partitions for middle beta value
        TD,opt_val,betapoint_plan_assignment,new_A,true_beta = TD_midpoint(shp,filename_A,filename_B,epsilon,warm_LS,
                                                                           warm_Init,comp_bool,col,beta,grid_bool)
        
        
        # Convert midpoint plan to pandas dataframe and output to csv
        temp_str = filename_seq + '_new'
        df_mid = pd.DataFrame.from_dict(betapoint_plan_assignment,orient='index',columns=['id','district'])
        df_mid.to_csv(temp_str + '.csv',index=False)
        
        
        # Set up recursion
        
        # Number of copies of betapoint_plan_assignment to include in sequence
        num = 1
        
        # Create left and right beta lists
        if index == 0:
            left_beta = []
            
        else:
            left_beta = []
            
            for val in beta_list[0:index]:
                if val < true_beta:
                    left_beta.append(float(val/true_beta))
                else:
                    num += 1
        
        if index == len(beta_list)-1:
            right_beta = []
        else:
            right_beta = []
            
            for val in beta_list[index+1:]:
                if val > true_beta:
                    right_beta.append(float((val-true_beta)/(1-true_beta)))
            
            
        # Recurse
        left_seq = TD_sequence(shp,filename_A,temp_str + '.csv',temp_str,epsilon,warm_LS,
                               warm_Init,comp_bool,col,left_beta,grid_bool,L,level+1)
        right_seq = TD_sequence(shp,temp_str + '.csv',filename_B,temp_str,epsilon,warm_LS,
                                warm_Init,comp_bool,col,right_beta,grid_bool,L,level+1)
        
        copies = []
        
        for val in range(0,num):
            copies.append(betapoint_plan_assignment)
        
        if level == L:
            return [new_A] + left_seq + copies + right_seq
        else:
            return left_seq + copies + right_seq
        
        