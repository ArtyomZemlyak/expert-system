# Import the library
import bnlearn
from pgmpy.factors.discrete import TabularCPD


# Define the network structure
edges = [("tz", "t0"), ("tz", "t1"), ("t0", "t2"), ("t0", "t3")]


tz = TabularCPD(variable="tz", variable_card=2, values=[[0.01], [0.99]])
# t0 = TabularCPD(variable='t0', variable_card=2, values=[[0.4], [0.6]])
# t1 = TabularCPD(variable='t1', variable_card=2, values=[[0.6], [0.4]])

# print(t0)

t0 = TabularCPD(
    variable="t0",
    variable_card=2,
    values=[[0.99, 0.4], [0.01, 0.6]],
    evidence=["tz"],
    evidence_card=[2],
)
t1 = TabularCPD(
    variable="t1",
    variable_card=2,
    values=[[0.99, 0.6], [0.01, 0.4]],
    evidence=["tz"],
    evidence_card=[2],
)
t2 = TabularCPD(
    variable="t2",
    variable_card=2,
    values=[[0.99, 0.67], [0.01, 0.33]],
    evidence=["t0"],
    evidence_card=[2],
)
t3 = TabularCPD(
    variable="t3",
    variable_card=2,
    values=[[0.99, 0.34], [0.01, 0.66]],
    evidence=["t0"],
    evidence_card=[2],
)
# print(t2)

# Make the actual Bayesian DAG
DAG = bnlearn.make_DAG(edges)
DAG = bnlearn.make_DAG(DAG, CPD=[tz, t0, t1, t2, t3])
# bnlearn.print_CPD(DAG)

labels = ["tz", "t0", "t1", "t2", "t3", "t4", "t5", "u1", "u2", "u3", "u4"]


variables = [
    # "tz",
    # "t0",
    # "t1",
    # "t2",
    "t3",
    # "t4",
    # "t5",
    # "u1",
    # "u2",
    # "u3",
    # "u4"
]

pattern = {
    "tz": 1,
    "t0": 0,
    # "t1": 1,
    # "t2": 1,
    # "t3": 0,
    # "t4": 1,
    # "t5": 1,
    # "u1": 1,
    # "u2": 1,
    # "u3": 1,
    # "u4": 1,
}

q1 = bnlearn.inference.fit(DAG, variables=variables, evidence=pattern)
# print(q1)
