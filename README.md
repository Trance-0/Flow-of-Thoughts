# Flow of Thoughts

The Flow of Thoughts is a framework that can dynamically iterate the long passage with self-RAG based on a graph of thoughts. The idea goes as follows:

First, the LLM will split the passage into several syntactically independent paragraphs by iterating the long texts. For example, the LLM will split a paper into sections like abstract, importance of projects, related work A, related work B, proposed solutions, etc that can fit into the context windows.

Then we let LLM to construct subgraph with node of ideas determine if the passage is relevant to the problem that we are going to solve. If the constructed subnode is relevant to solving our question, we select these node to cache. (which will be updated with higher weight in thought graph.)

For example, when we ask "What related technologies did the author use when doing the project?" The LLM should ignore the proposed framework, conclusion, and experimental results section and only focus on reading the "Related work" sections.

Finally, we let LLM compose answers based on the related paper segments with supporting references.

```python
def train_knowledge_graph(binary_decision:bool=True):
    """
    G(V,E): is our constructed graph of knowledge
    depth: is our maximum "connection" used in searching relevant question in
    relevant information will have lower cost for "connection"
    topk: the update edge threshold during connection between thoughts
    
    complexity is O(V^2)
    """
    for v1 in V:
        # generate relevant question for solving questions related to content in v1
        generated_question=model.generate_question(v1.content)
        neighbor=[]
        for v2 in V:
            if v2==v1: continue
            # may use fuzzy or binary decision
            if binary_decision:
                if model.is_related(generated_question,v2.content):
                    neighbor.append(v2)
            else:
                heapq.heappush(neighbor,(model.relation_score(generate),v2))
        for ni in min(len(neighbor),topk):
            # may try directed or undirected graph version
            if binary_decision:
                graph.decrease_weight(v1,neighbor[ni])
            else:
                graph.decrease_weight(v1,heapq.heappop(neighbor))

def query(question,cost):
    # how to get the first node?
    root=?
    context=[root]
    q=[(root,cost)]
    while q:
        cur_node,cur_cost=q.pop(0)
        for adj_node,adj_cost in G[q]:
            if cur_cost-adj_cost>0 and adj_node is not in visited:
                q.append(adj_node,cur_cost-adj_cost)
                context.append(adj_node)
    return model.aggregate(question,context)
```
