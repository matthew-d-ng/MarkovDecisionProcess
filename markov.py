
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

S = ["fit", "unfit", "dead"]
A = ["relax", "exercise"]


def prob(state, action, next_state):

    return pr_matrix[action][state][next_state][0]


def reward(state, action, next_state):
    
    return pr_matrix[action][state][next_state][1]


def v(n, gamma, state):

    max = 0
    for action in A:
        q_val = q(n, gamma, state, action)
        if q_val > max:
            max = q_val
    return max


def q(n, gamma, state, action):

    if n == 0:
        q_val = 0
        for next_s in S:
            q_val = q_val + prob(state, action, next_s)*reward(state, action, next_s)
        return q_val

    val = 0
    for next_s in S:
        val = val + prob(state, action, next_s)*v(n-1, gamma, next_s)
    return q(0, gamma, state, action) + gamma*val


n = 3
gamma = 0.9

for s in S:
    print("State: ", s)
    print( " Exercise: ", q(n, gamma, s, "exercise") )
    print( " Relax: ", q(n, gamma, s, "relax"), "\n" )