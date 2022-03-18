import numpy as np
import torch

#structural specification for MCTS
NUM_LAYER = 8
NUM_CHILD_IN_LAYER = [9,9*2]*4

class Node(object):
  def __init__(self,parent=None,q=0.0,n=0.0,arch=None):
    self._parent = parent
    self._num = -1.0
    self._children = []
    self.q = q
    self.n = n
    self.arch = arch

  @property
  def parent(self):
    return self._parent
  @property
  def children(self):
    return self._children
  @property
  def num(self):
    return self._num
  def set_num(self,num):
    self._num = num

  def add_child(self,child):
    self._children.append(child)

  def update(self,q=None,n=None):    
    if q is not None:
      self.q += q
    if n is not None:
      self.n += n
  @property
  def is_leaf(self):
    if self.arch[0] == NUM_LAYER-1:
      return True
    else:
      return False
  @property
  def is_root(self):
    if self._parent is None:
      return True
    else:
      return False
  def __call__(self):
    print(self.q,self.n)
    
def Selection(root_node,tree_nodes,c=0.6):
  #root_node.is_root must be true !
  trail = []

  cur_node = root_node
  for layer in range(NUM_LAYER): 

    num_child = NUM_CHILD_IN_LAYER[layer%NUM_LAYER]
    child_idx = [tree_nodes[child].arch[1] for child in cur_node.children]

    n_ = [1 for x in range(num_child)]
    q_ = (0.9+np.random.rand(num_child)*0.2)*(cur_node.q+1e-3)/(cur_node.n+0.01)
    for child in cur_node.children:
      n_[tree_nodes[child].arch[1]] += tree_nodes[child].n
      q_[tree_nodes[child].arch[1]] += tree_nodes[child].q
    n_all = sum(n_)
    selects = [q_[child]/n_[child]+2*c*np.sqrt(2*np.log(n_all)/n_[child]) for child in range(num_child)]

    to_choose = selects.index(max(selects))
    if to_choose not in child_idx:
      cur_node = Expansion(cur_node,tree_nodes,depth=layer,action=to_choose)
    else:
      cur_node = tree_nodes[cur_node.children[child_idx.index(to_choose)]]
    trail.append((to_choose,cur_node))
  
  leaf_node = cur_node

  return leaf_node,trail


def Sample(root_node,tree_nodes,c=0.6):
  trail = []

  cur_node = root_node
  for layer in range(NUM_LAYER): 
    num_child = NUM_CHILD_IN_LAYER[layer%NUM_LAYER]
    child_idx = [tree_nodes[child].arch[1] for child in cur_node.children]

    n_ = []
    q_ = []
    for child in cur_node.children:
      n_.append(tree_nodes[child].n)
      q_.append(tree_nodes[child].q)
    n_all = sum(n_)
    selects = [q_[child]/(n_[child]+1e-2)+2*c*np.sqrt(2*np.log(n_all)/(n_[child]+1e-2)) for child in range(len(q_))]
    to_choose = selects.index(max(selects))
    to_choose = child_idx[to_choose]
      
    cur_node = tree_nodes[cur_node.children[child_idx.index(to_choose)]]
    trail.append((to_choose,cur_node))
  
  leaf_node = cur_node
  return leaf_node,trail

def Expansion(node,tree_nodes,depth=0,action=0):
  child = Node(parent=node.num,q=0.0,n=0,arch=(depth,action))
  child.set_num(len(tree_nodes))
  node.add_child(child.num)
  tree_nodes[child.num]=child
  return child

def BackProp(leaf_node,tree_nodes,q,n):
  cur_node = leaf_node
  cur_node.update(q=q,n=n)
  while True:
    cur_node = tree_nodes[cur_node.parent]
    cur_node.update(q=q,n=n)
    if cur_node.is_root:
      break

def DeriveSelection(root_node,tree_nodes,c=0.5,max_sample=False,verbose=False):
  trail = []

  cur_node = root_node
  for layer in range(NUM_LAYER): 

    num_child = NUM_CHILD_IN_LAYER[layer%NUM_LAYER]
    child_idx = [tree_nodes[child].arch[1] for child in cur_node.children]

    n_ = []
    q_ = []
    for child in cur_node.children:
      n_.append(tree_nodes[child].n)
      q_.append(tree_nodes[child].q)
    n_all = sum(n_)
    selects = [q_[child]/(n_[child]+1e-2)+2*c*np.sqrt(2*np.log(n_all)/(n_[child]+1e-2)) for child in range(len(q_))]
    to_choose = selects.index(max(selects))
    to_choose_idx = to_choose
    to_choose = child_idx[to_choose]

    if max_sample is False:
      if len(selects)>1:
        vec = torch.Tensor(selects)
        vec = (vec - vec.mean())/(vec.std()+1e-6)
        vec = vec.softmax(dim=0)

        try:
          sample = vec.multinomial(1,replacement=True)
        except:
          print(vec,selects)
          raise
        to_choose2 = child_idx[sample]
        to_choose = to_choose2
      else:
        to_choose2 = to_choose
    if verbose:
      print(f'{layer:2d}, {len(child_idx):4d}/{num_child:4d},{to_choose:4d}',
            f'{cur_node.q:7.2f} / {cur_node.n:4.0f}',
            )
    
    if to_choose not in child_idx:
      cur_node = Expansion(cur_node,tree_nodes,depth=layer,action=to_choose)
    else:
      cur_node = tree_nodes[cur_node.children[child_idx.index(to_choose)]]
    trail.append((to_choose,cur_node))
  
  leaf_node = cur_node

  return leaf_node,trail
