import sys

pr_matrix = {"exercise": {
                "fit":      {"fit":(0.891, 8), "unfit":(0.009, 8), "dead":(0.1, 0)},
                "unfit":    {"fit":(0.18, 0),  "unfit":(0.72, 0),  "dead":(0.1, 0)},
                "dead":     {"fit":(0, 0),     "unfit":(0, 0),     "dead":(1, 0)}
            },
             "relax": {
                "fit":      {"fit":(0.693, 10), "unfit":(0.297, 10), "dead":(0.01, 0)},
                "unfit":    {"fit":(0, 5),      "unfit":(0.99, 5),   "dead":(0.01, 0)},
                "dead":     {"fit":(0, 0),      "unfit":(0, 0),      "dead":(1, 0)}
            }
           }
States = ["fit", "unfit", "dead"]
Actions = ["relax", "exercise"]

class memoize(dict):
    """ Memoization implementation inspired from 
        https://wiki.python.org/moin/PythonDecoratorLibrary/#Alternate_memoize_as_dict_subclass
    """
    def __init__(self, f):
        self.f = f

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        self[key] = self.f(*key)
        return self[key]


def prob(state, action, next_state):

    return pr_matrix[action][state][next_state][0]

def reward(state, action, next_state):
 
    return pr_matrix[action][state][next_state][1]

@memoize
def v(n, gamma, state):

    max = 0
    # V_n(s) = max( q_n(s, state_1), q_n(s, state_2), ..., q_n(s, state_n) )
    for action in Actions:
        q_val = q(n, gamma, state, action)
        if q_val > max:
            max = q_val
    return max

@memoize
def q(n, gamma, state, action):

    if n == 0:
        # q_0(s, a) = sum of  p(s, a, state_i)*r(s, a, state_i)
        q_val = 0
        for next_s in States:
            q_val = q_val + prob(state, action, next_s)*reward(state, action, next_s)
        return q_val

    # q_n+1(s, a) = q_0(s, a) + gamma*( sum of  p(s, a, state_i)*V_n(state_i) )
    val = 0
    for next_s in States:
        val = val + prob(state, action, next_s)*v(n-1, gamma, next_s)
    return q(0, gamma, state, action) + gamma*val


if len(sys.argv) == 4:
    n = int(sys.argv[1])
    gamma = float(sys.argv[2])
    s = sys.argv[3]

    print("State: ", s)
    print( " Exercise: ", q(n, gamma, s, "exercise") )
    print( " Relax: ", q(n, gamma, s, "relax"), "\n" )
else:
    print("Enter arguments (n, gamma, state) in form:\n python3 markov.py n gamma state")
    print("(Available states: fit, unfit, dead)")
