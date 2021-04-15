# Auto Tuning Bandits

## Dependencies

To run the code, you will need 
```
Python3, NumPy, Matplotlib
```


## Commands
An example:

```
python3 run.py -rep 10 -algo linucb -k 100 -t 10000 -d 5 -data simulations -lamda 1 -delta 0.1 -sigma 0.01 -lamdas 0.1 0.5 1 -js 0.01 0.1 1
```

The above commands means:
- Each experiment is independently repeated 10 times (``rep``). The value of ``rep`` must be at least 1.
- Compare tuning algorithms and theoretical explore for LinUCB algorithm (``algo``). The value of ``algo`` currently also supports ``lints`` and ``glmucb``.
- ``k`` is the number of arms. 
- ``d`` is the dimension of feature vectors.
- ``data`` supports values of ``simulations`` or ``movielens`` or ``netflix``. 
- ``lamda`` is the value of regularization parameter for ridge regression when lamda is not a tuning parameter.
- ``delta``. Usually, a bandit algorithm has theoretical guarantees like the regret upper bound holds with probability at least ``1-delta``. So ``delta`` here is the error probability, it should be a small number, but small ``delta`` leads to bigger exploration rate in the theoretical explore.
- ``sigma`` is the standard deviation of observed rewards.
- ``lamdas`` is the tuning set for ``lamda``. Type the values you want to put in the set in increasing order. Separate the values by a ``space``
- ``js`` is the tuning set for exploration rates.

The values in the above example are all default values. But you must specify what the tuning sets are. That being said, the above command is equivalent to the following command:

```
python3 run.py -lamdas 0.1 0.5 1 -js 0.01 0.1 1
```

Note that the above commands must be run in the ``autoTuning`` folder. 

## Output inside the screen after running the command
``3 :  theory 330.6814460045903, auto 226.68195466402278, op 256.47816211328393, auto_3layer 145.1782560069353, auto_combined: xxxx``

An output like the above will show on the screen when each experiment is finished. ``3`` means the ``4th`` repeated experiment. ``theory``, ``auto``, ``op``, ``auto_3layer``, ``auto_combined`` are five tuning methods compared. The number right after the method is the cumulative regret of each method, the smaller the better.



## Plots

To produce the plots, run the following command, it will create a ``plots`` folder and the figures will be saved there.

```
python3 plot.py
```

To see the plots, run the following command one by one inside the ``autoTuning`` folder:
```
cd plots
zip -r plots.zip *
```

Then use ``scp`` to send the ``plots.zip`` file to your local laptop to unzip, open and see the plots.


## Some notes
- Bigger ``d`` usually makes the results of all methods worse. But bigger ``k`` does not necessarily make the results worse. Overall, the plots of regret should look sub-linear, except for too big ``d`` or ``k``.
- Bigger ``d`` or ``k`` or ``t`` or ``rep`` or more values in the tuning sets all make the code slower to finish.
- As far as I can see, bigger exploration rate usually gives bad result for theoretical explore.
- Changing the values of ``lamdas`` only affect the results of ``auto_3layer`` and ``auto_combined`` since other methods do not tune ``lamda``.
- If ``lamda`` too large, then theoretical explore will be bad, since the theoretical exploration rate is increasing with respect to ``lamda``.
- Real datasets ``movielens`` and ``netflix`` only supports ``d=`10`` and ``d=20`` now. Setting ``d`` to other values is also doable directly with the above commands, but will need to cost lots of time for data preprocessing, especially for ``netflix`` data.
- Netflix data is very big, so runtime is slower than other datasets.
- Netflix data only supports ``k<=2042`` and Movielens data only supports ``k<=1682``.




