# Contribution guidelines
Taweret is part of the BAND Collaboration's software suite.
Thus any additions made must conform to their requirements which can be found [here](https://github.com/bandframework/bandframework/blob/main/resources/sdkpolicies/bandsdk.md).
All additions need to be added via pull request.
Any files added need the copyright disclaimer added to the top of the file, and any files edited should include your name in the author list.


## Adding new `Model` or `Mixer`
- All additional models and mixing methods need to inherit form `BaseModel` and `BaseMixer` represectively.
The interfaces of these mixers should be simple enough that they can be called with the follwoing three lines

```python
mixer = CostumMixer(...)
mixer.set_prior(...)
mixer.train(...)
```

- The mixer constructors should always consume a dictionary of models, such that they type constraint `Dict[str, Type[BaseModel]]` is statisfied.

- If your contribution requires extra pip-installable modules, you should include them in th `setup.py` file.

- If your code relies on constum binaries, such as an executable built from other code, you should include reproducible instructions fore all operating systems (Linux, MacOS, and Windows).

## Code formatting
Once your code is written, and you are ready to add your code via `autopep8` using the command
```terminal
autopep8 --in-place --aggressive --aggressive -r
```
