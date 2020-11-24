import numpy as np
from sympy import *
from sympy.logic import simplify_logic
from sympy.parsing.sympy_parser import parse_expr

import itertools
import sys


def act(x):
    return (1 / (1 + np.exp(-x)))


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def combine(m, n):
    a = np.shape(m)[0]
    c = list()
    count = 0
    for i in range(a):
        if (m[i] == n[i]):
            c.append(m[i])
        elif (m[i] != n[i]):
            c.append(2)
            count += 1

    if (count > 1):
        return None
    else:
        return c


###############################################################
#Writing a Psi_i(with sympy)
def calc_form(X, y, form, pos):
    X1 = X[np.nonzero(y)]
    if np.shape(X1)[0] == 0:
        return "False"
    if np.shape(X1)[0] == np.shape(X)[0]:
        return "True"
    formula = ""
    size = np.shape(X1)
    for i in range(size[0]):
        if formula != "":
            formula = formula + "|"
        formula = formula + "("
        for j in range(size[1]):
            if X1[i][j] == 0:
                formula = formula + "~" + form[pos[j]] + "&"
            if X1[i][j] == 1:
                formula = formula + form[pos[j]] + "&"
        # Code below duplicates for, can't we just close the bracket, e.g. formula[-1]=')'
        if X1[i][size[1] - 1] == 0:
            formula = formula + "~" + form[pos[size[1] - 1]] + ")"
        if X1[i][size[1] - 1] == 1:
            formula = formula + form[pos[size[1] - 1]] + ")"

    form_cnf = to_cnf(formula, True)
    formula = str(form_cnf)
    return formula


########################################################################


def booleanConstraint(weights, bias):
    print("w", weights, "b", bias)

    #weights contains the list of weight matrices with shape h_i+1 X h_i
    #bias contains the list of bias vectors with shape h_i X 1

    #parameters
    num_layers = len(weights)
    print("num_layers", num_layers)
    h = np.zeros(num_layers, dtype=int)
    for j in range(num_layers):
        h[j] = np.shape(weights[j])[0]
        print('h[',j,']=np.shape(', weights[j], ')[0]=', h[j])
    rel_pos = [[np.nonzero(weights[j][i]) for i in range(h[j])]
               for j in range(num_layers)]
    print('rel_pos', rel_pos)

    #inputs
    fan_in = np.count_nonzero((weights[0])[0, :])
    print('fan_in', fan_in, '= count nonzero(', (weights[0])[0, :], ')')
    array = np.repeat([[0, 1]], fan_in, axis=0)
    print('array', array)
    X = cartesian_product(*array)
    print('X', X)
    num_inputs = np.shape(X)[0]
    print('num_inputs', num_inputs)

    #outputs
    y = list()
    for j in range(num_layers):
        print("Layer", j, 'h', h[j])
        weights_active = np.zeros((h[j], fan_in))
        print("  weights_active", weights_active.shape)
        output = np.zeros((h[j], num_inputs))
        print("  output", output.shape)
        for i in range(h[j]):
            print("    Neuron", i)
            weights_active[i] = (weights[j])[i, np.nonzero(weights[j][i])]
            print("    weights_active[", i, "]=", (weights[j]), "[", i, ",", np.nonzero(weights[j][i]), "]")
        a = np.matmul(weights_active, np.transpose(X))
        print("  a=np.matmul(", weights_active, ",", np.transpose(X), ")=", a)
        b = np.reshape(np.repeat(bias[j], np.shape(X)[0], axis=0), np.shape(a))
        output = act(a + b)
        print("output", output)
        crisp = np.where(output < 0.5, 0, 1)
        print("crisp", crisp)
        y.append(crisp)

    #list of formulas to be combined
    form_labels = list()
    for k in range(np.shape(weights[0])[1]):
        print(k, "form_labels", form_labels, " --> ", end='')
        form_labels.append("f" + str(k + 1))
        print(form_labels)
    for j in range(num_layers):
        print("Layer ", j)
        formula = list()
        print(form_labels)
        for i in range(h[j]):
            print("  i", i, formula, " --> ", end='')
            formula.append(
                "(" + calc_form(X, y[j][i], form_labels, rel_pos[j][i][0]) +
                ")")
            print(formula)
            print("calc_form(", X.tolist(), y[j][i], form_labels, rel_pos[j][i][0], ')')
        form_labels = formula
    return formula


def conjunt_list(psi):
    N = len(psi)
    c = 0
    psi_list = list()
    for i in range(N):
        if psi[i] == "&":
            psi_list.append(psi[c:i])
            c = i + 1
    psi_list.append(psi[c:N])
    return psi_list


#psi represents the list of psi_i rules extracted and already in CNF
def global_rules(psi_list, tractable=False):
    try:
        N = len(psi_list)
        psi_list_list = [conjunt_list(psi_list[i]) for i in range(N)]
        # print(psi_list_list)

        ############################################
        if tractable:
            #global rule psi (possible explosion)
            psi1 = "("
            for i in range(N - 1):
                psi1 = psi1 + psi_list[i] + ") | ("
            psi1 = psi1 + psi_list[N - 1] + ")"
            psi_cnf1 = to_cnf(psi1, True)
            psi_cnf1 = str(psi_cnf1)
            return psi_cnf1
        ############################################

        psi_conjuncts = list()
        for element in itertools.product(*psi_list_list):
            form_temp = element[0]
            for i in range(1, N):
                form_temp = form_temp + " | " + element[i]
            par = simplify_logic(parse_expr(form_temp))
            par = str(par)
            psi_conjuncts.append(par)
        psi = "(" + psi_conjuncts[0] + ")"
        for i in range(1, len(psi_conjuncts)):
            psi = psi + " & (" + psi_conjuncts[i] + ")"
        # print(psi)
        psi_cnf = to_cnf(psi, True)
        psi_cnf = str(psi_cnf)
    except:
        print("Oops!", sys.exc_info(), "occured.")
        psi_cnf = ""
        for psi in psi_list:
            psi_cnf = psi_cnf + " | (" + psi + ")"
        # print(psi_cnf)
        # psi_conjuncts = []

    return psi_cnf

if __name__ == '__main__':
    #######################################################################################test examples to CNF
    # a=["x","y","z"]
    # b=["x|y","y|z","k"]
    # d=["x","w"]
    # c=[a,b,d]
    # N= len(c)
    #
    # for element in itertools.product(*lista_psi):
    # # for element in itertools.product(a,b):
    #     print(element)
    #     form_temp = element[0]
    #     for i in range(1,N):
    #         form_temp = form_temp+"|"+element[i]
    #     print(form_temp)
    #     par = simplify_logic(parse_expr(form_temp))
    #     print(par)
    #     form_list.append(par)

    # l=["(Eight & odd & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even) | (Five & even & ~Eight & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Five & odd & ~Eight & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even) | (Nine & even & ~Eight & ~Five & ~Four & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Nine & odd & ~Eight & ~Five & ~Four & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even) | (One & odd & ~Eight & ~Five & ~Four & ~Nine & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even) | (Seven & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Six & ~Three & ~Two & ~Zero & ~even) | (Six & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Three & ~Two & ~Zero & ~even) | (Three & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Two & ~Zero & ~even) | (Two & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Zero & ~even) | (Zero & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~even) | (odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even)", "(Eight & even & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Eight & odd & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~even) | (Five & even & ~Eight & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Four & even & ~Eight & ~Five & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Nine & even & ~Eight & ~Five & ~Four & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Seven & even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Six & ~Three & ~Two & ~Zero & ~odd) | (Six & even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Three & ~Two & ~Zero & ~odd) | (Three & even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Two & ~Zero & ~odd) | (Two & even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Zero & ~odd) | (Two & odd & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Zero & ~even) | (Zero & even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~odd) | (even & ~Eight & ~Five & ~Four & ~Nine & ~One & ~Seven & ~Six & ~Three & ~Two & ~Zero & ~odd)"]
    #
    # l=["~TAIL & (BICYCLE | HANDLEBAR)", "ROOFSIDE & ~HEAD & ~TORSO", "SOFA & (~NECK | ~SCREEN)", "CAP & ~WING", "~POTTEDPLANT & (DOG | ~HEAD)", "BIRD & (~BODY | ~LEG)", "STERN & ~EYE & ~RIGHTSIDE", "~HEAD & (STERN | ~TORSO)", "HEAD & (HO | HORSE)", "CHAINWHEEL & ~HEAD", "~AEROPLANE & (BODY | ~HAND)", "BUS | (~CAR & ~HEAD)", "~HEAD & ~TORSO", "TABLE & ~EYE & ~LEG", "STERN & (~HEAD | ~TORSO)", "CHAIR", "~CHAIR & (MOTORBIKE | ~TORSO)", "~BODY & (TRAIN | ~TORSO)", "SHEEP & ~BOAT & ~PLANT", "STERN & ~HEAD", "CAT & ~FOOT", "~LEG & (BACKSIDE | ~HEADLIGHT)", "SCREEN | (~HEAD & ~PERSON)", "~FRONTSIDE & ~HORSE & ~TVMONITOR"]
    #
    #
    # lista=[str(to_cnf(i,True)) for i in l]
    # print(lista)
    #
    #
    #
    #
    # # lista=["x & y","(y|z)&(y|x)","k & w"]
    # # lista=["x & y","(y|z)&(y|x)","k & w"]
    # # print(lista)
    #
    #
    #
    # psi = global_rules(lista)
    # print(psi)
    #
    #
    # exit()
    #

    #TESTING EXAMPLES psi_i
    # # 1
    # w1 = np.array([[0.,1.1,-2.1],[3.8,-1.9,0.],[1.5,0.,-1.3]])
    w2 = np.array([[3.8,-1.9,0.]])
    # b1 = [1.2,0.,-1.9]
    b2 = [5.1]
    #
    w = [w2]
    b = [b2]

    # #2
    w1 = np.array([[0,1,-2,0,0],[0,0,3,-1,0],[0,1,0,-1,0]])
    w2 = np.array([[0,1,2]])
    b1 = [1,0,-1]
    b2 = [5]
    #
    w = [w1,w2]
    b = [b1,b2]

    #3
    # w1 = np.array([[0.,1.2,-2.1,0.,0.],[0.,0.,3.1,-1.9,0.],[0.,1.4,0.,-1.2,0.],[0.,1.4,0.,-1.5,0.]])
    # w2 = np.array([[0.,1.5,2.3,0.],[0.,-2.1,0.,3.],[-2.9,1.1,0.,0.]])
    # w3 = np.array([[2.1,2.2,0.]])
    # b1 = [1.1,0.,-1.2,1.1]
    # b2 = [1.1,-2.2,1.]
    # b3 = [0.1]
    #
    #
    # w = [w1,w2,w3]
    # b = [b1,b2,b3]

    #4
    # w1 = np.array([[0.,1.2,-2.1,0.9,0.],[1.8,0.,2.1,-1.9,0.],[0.,1.4,1.1,-1.2,0.],[0.,1.4,0.,-1.5,-1.]])
    # w2 = np.array([[1.1,1.5,2.3,0.],[0.,-2.1,0.9,3.],[-2.9,1.1,0.,1.5]])
    # w3 = np.array([[2.1,-2.2,0.9]])
    # b1 = [1.1,0.,-1.2,1.1]
    # b2 = [1.1,-2.2,1.]
    # b3 = [0.1]
    #
    # w = [w1,w2,w3]
    # b = [b1,b2,b3]

    # w1 = np.array([[-1.1,1.2,0.,0.,0.,0.],[0.,0.9,0.6,0.,0.,0.],[0.,0.,0.,1.4,-1.1,0.],[0.,0.,0.,0.,1.2,-1.]])
    # w2 = np.array([[-1.1,1.5,0.,0.],[0.,-2.1,0.9,0.],[0.,0.,0.9,-1.2]])
    # # w3 = np.array([[1.,1.,1.]])
    # b1 = [0.1,-0.1,0.3,0.05]
    # b2 = [0.2,-0.1,0.1]
    # # b3 = [0.1]
    #
    # w = [w1,w2]
    # b = [b1,b2]

    # a=parse_expr("~(x | True)")
    # b=parse_expr("~(x & False)")
    # c=parse_expr("~(True)")
    # d=parse_expr("~(False)")
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    # # print(e)
    # exit()

    #
    #
    # w1 = np.array([[1.6,0.,-0.9,0.],[0.,0.8,0.,-0.5],[0.,0.3,0.4,0.]])
    # w2 = np.array([[-1.1,0.,1.1],[0.,-0.9,1.1]])
    # w3 = np.array([[0.5,-0.6,0.]])
    # b1 = [0.1,0.1,-0.1]
    # b2 = [0.1,-0.1]
    # b3 = [0.1]
    #
    # w = [w1,w2,w3]
    # b = [b1,b2,b3]
    #
    #
    # #parsing and simplyfing
    #
    f = booleanConstraint(w, b)
    print("psi_i: ", f)
    # h=""
    # for i in range(len(f)-1):
    #     h=h+f[i]+"|"
    # h=h+f[len(f)-1]
    # g = parse_expr(h)
    # # g = parse_expr(f)
    # print("psi_parsata: ",g)
    # print("psi_simple: ",simplify_logic(g))
    #
    #
