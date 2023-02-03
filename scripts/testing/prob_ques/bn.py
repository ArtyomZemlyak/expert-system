from pomegranate import (
    DiscreteDistribution,
    ConditionalProbabilityTable,
    State,
    BayesianNetwork,
)

tz = DiscreteDistribution({True: 0.99, False: 0.01})
t0 = DiscreteDistribution({True: 0.6, False: 0.4})
t1 = DiscreteDistribution({True: 0.4, False: 0.6})

t2 = ConditionalProbabilityTable(
    [[True, True, 0.2], [True, False, 0.8], [False, False, 0.95], [False, True, 0.05]],
    [t0],
)
t3 = ConditionalProbabilityTable(
    [[True, True, 0.8], [True, False, 0.2], [False, False, 0.8], [False, True, 0.2]],
    [t0],
)
t4 = ConditionalProbabilityTable(
    [[True, True, 0.75], [True, False, 0.25], [False, False, 0.8], [False, True, 0.2]],
    [t3],
)
t5 = ConditionalProbabilityTable(
    [[True, True, 0.25], [True, False, 0.75], [False, False, 0.8], [False, True, 0.2]],
    [t3],
)
u1 = ConditionalProbabilityTable(
    [[True, True, 0.9], [True, False, 0.1], [False, False, 0.9], [False, True, 0.1]],
    [t1],
)
u2 = ConditionalProbabilityTable(
    [[True, True, 0.9], [True, False, 0.1], [False, False, 0.9], [False, True, 0.1]],
    [t2],
)
u3 = ConditionalProbabilityTable(
    [[True, True, 0.9], [True, False, 0.1], [False, False, 0.9], [False, True, 0.1]],
    [t4],
)
u4 = ConditionalProbabilityTable(
    [[True, True, 0.9], [True, False, 0.1], [False, False, 0.9], [False, True, 0.1]],
    [t5],
)


s0 = State(tz, name="tz")
s1 = State(t0, name="t0")
s2 = State(t1, name="t1")
s3 = State(t2, name="t2")
s4 = State(t3, name="t3")
s5 = State(t4, name="t4")
s6 = State(t5, name="t5")
s7 = State(u1, name="u1")
s8 = State(u2, name="u2")
s9 = State(u3, name="u3")
s10 = State(u4, name="u4")


model = BayesianNetwork()
model.add_states(s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10)
model.add_edge(s0, s1)
model.add_edge(s0, s2)
model.add_edge(s1, s3)
model.add_edge(s3, s8)
model.add_edge(s1, s4)
model.add_edge(s4, s5)
model.add_edge(s4, s6)
model.add_edge(s5, s9)
model.add_edge(s6, s10)
model.add_edge(s2, s7)
model.bake()

print(model.structure)

# print(model.predict([[None, None, True]]))
# print(model.predict_proba([[None, None, True]]))

labels = ["tz", "t0", "t1", "t2", "t3", "t4", "t5", "u1", "u2", "u3", "u4"]

pattern = {
    "tz": True,
    "t0": True,
    # "t1": True,
    # "t2": True,
    "t3": False,
    # "t4": True,
    # "t5": True,
    # "u1": True,
    # "u2": True,
    # "u3": True,
    # "u4": True,
}

preds = model.predict([pattern])[0]
proba = model.predict_proba(pattern)

for label, pred, prob in zip(labels, preds, proba):
    print(label, pred, end="")

    if type(prob) != bool:
        print(" ", prob.parameters[0][pred], end="")
    print("")
