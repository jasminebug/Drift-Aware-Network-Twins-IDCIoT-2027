from __future__ import annotations
import heapq
from dataclasses import dataclass,field
import numpy as np

# Canonical graph shape used by the public lightweight repository. 21 undirected, 42 directed edges.
NSFNET_EDGES=((0,1),(0,2),(0,3),(1,2),(1,7),(2,5),(2,7),
 (3,4),(3,8),(4,5),(4,6),(5,7),(5,9),(6,8),
 (6,10),(7,9),(8,10),(8,11),(9,12),(10,13),(11,12))
@dataclass
class Network:
    nodes:int
    directed_edges:list
    capacities:np.ndarray
    propagation:np.ndarray
    pairs:list
    _path_cache:dict=field(default_factory=dict,init=False,repr=False)
    @property
    def links(self):return len(self.directed_edges)
    @property
    def demands(self):return len(self.pairs)
    def adjacency(self,weights):
        adj=[[] for _ in range(self.nodes)]
        for i,(u,v) in enumerate(self.directed_edges):adj[u].append((v,float(weights[i]),i))
        return adj
    def shortest_path(self,source,target,weights):
        adj=self.adjacency(weights);dist=[float('inf')]*self.nodes;prev=[None]*self.nodes
        dist[source]=0.;heap=[(0.,source)]
        while heap:
            d,u=heapq.heappop(heap)
            if d!=dist[u]:continue
            if u==target:break
            for v,w,edge in adj[u]:
                nd=d+max(w,1e-6)
                if nd<dist[v]:
                    dist[v]=nd;prev[v]=(u,edge);heapq.heappush(heap,(nd,v))
        if prev[target] is None:raise RuntimeError(f'No route from {source} to {target}')
        path=[];node=target
        while node!=source:
            parent,e=prev[node];path.append(e);node=parent
        return path[::-1]
    def path_matrix(self,weights):
        key=tuple(np.round(np.asarray(weights,dtype=float),4))
        if key in self._path_cache:return self._path_cache[key]
        A=np.zeros((self.links,self.demands),dtype=float)
        for p,(u,v) in enumerate(self.pairs):A[self.shortest_path(u,v,weights),p]=1.
        if len(self._path_cache)>=2048:self._path_cache.pop(next(iter(self._path_cache)))
        self._path_cache[key]=A
        return A

def make_network(seed=7):
    rng=np.random.default_rng(seed)
    directed=[x for (u,v) in NSFNET_EDGES for x in ((u,v),(v,u))]
    cap=rng.uniform(.95,1.10,len(directed))
    prop=rng.uniform(.8,2.5,len(directed))
    pairs=[(i,j) for i in range(14) for j in range(14) if i!=j]
    return Network(14,directed,cap,prop,pairs)

def evaluate_network(network,weights,demand):
    A=network.path_matrix(weights)
    d=np.asarray(demand,dtype=float)
    load=A@d;util=load/network.capacities
    q=np.minimum(.45*util/np.maximum(1.-util,.03),8.)
    delay=network.propagation+q
    mean_delay=float(np.sum((A.T@delay)*d)/(np.sum(d)+1e-8))
    loss=float(np.mean(np.maximum(util-1.,0.)))
    overload=float(np.mean(np.maximum(util-.78,0.)))
    return {'delay_ms':mean_delay,'loss':loss,'overload':overload,
            'cost':float(mean_delay/3.+25.*loss+5.*overload),'load':load,'utilization':util}
