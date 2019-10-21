### Tidy version of code used for Master Thesis

### Dependencies:
- [graph_tool](https://graph-tool.skewed.de/)
- numpy
- tqdm
- matplotlib


### Usage:

```python
from Algorithm_Classes.CG_Class import *
from CA_Classes.CA1DClass import *
from CA_Classes.CA2DClass import *
from CA_Classes.ReversibleClass import *
from tqdm import tqdm

Automaton = ElementaryAutomaton(rule=110,size=2)
#Automaton = TotalCA2D.by_bs_string("B3/S23")
#Automaton = SecondOrderAutomaton(ElementaryAutomaton(rule=110,size=2))

#CG_Algorithm = Complete_Graph_Calc(Automaton,show_progress=True)
CG_Algorithm = Cheap_Graph_Calc(Automaton,num_runs=10,num_tests=2,show_progress=True)

CG_Algorithm.display_graph(max_iter=0)

```