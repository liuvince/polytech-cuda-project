# High Performance Computing Projet

## Merge Path

Merge two sorted array A and B in a M array

```
nvcc path_merge.cu -o path_merge
./path_merge
```

## Batch Merge

Given a large number N of sorted arrays Ai and Bi with |Ai| + |Bi| = d, 
Merge two by two for all i, Ai and Bi

```
nvcc batch_merge.cu -o batch_merge
./path_merge
```

## References

* Green, Oded & Mccoll, Rob & Bader, David. (2014). GPU merge path: a GPU merging algorithm. Proceedings of the International Conference on Supercomputing. 10.1145/2304576.2304621. 





