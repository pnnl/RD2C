# MIDDLE: Model Independent Distributed Learning

In this repository we provide code to run our model-independent distributed learning algorithm.
Within this algorithm, gradients or models are not communicated (preserving privacy) amongst devices.
Furthermore, devices *are* allowed to maintain different architectures.

## Running MIDDLE

To run an example of MIDDLE on CIFAR-10, use the following example:
```
mpirun -n 4 python cifar10.py --name MIDDLE-test
```

## Code Dependencies

We use Python 3.9.12 with the following packages:
1. TensorFlow 2.9.1
2. Mpi4py 3.1.4
3. Pandas 1.5.2
4. Numpy 1.24.0
5. Networkx 2.8.8
6. Six 1.16.0
6. Matplotlib 3.6.2