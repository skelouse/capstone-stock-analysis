## Folder Structure
```
\--- bin
|
\--- db
|
\--- img
|
\--- modeling
\--- \--- tests
\         \--- _python
\         \--- create
\         \--- tuner
|   
\--- old
|
\--- reports
\    \--- aapl_price_w_aapl_info
\    \--- aapl_price_w_all_price
\    \--- aapl_price_w_sector
|
\--- styles
|
\--- test_notebooks
```


## Repository Structure

```
\--- bin
\    |  __init__.py
\    |  anomoly.py
\    |  database-schema.py
\    |  NN.py
\    |  out.png
\    |  correlation data csv files
|
|
\--- db
\    |  __init__.py
\    |  database.py
\    |  firebase.py
|
|
\--- img
\    |  flow.png
|
|
\--- modeling
\--- \--- tests
\         \--- _python
\              |  test_param_setting.py
\
\         \--- create
\         \--- tuner
\              |  test_cv.py
\              \-  val_folds
\    |  __init__.py
\    |  build.py
\    |  create.py
\    |  sequential.py
\    |  tuner.py
|   
|
\--- old   
\    |  Old main.ipynb
\    |  Old main2.ipynb
\    |  Old model_creation.ipynb
\    |  Old Modeling.ipynb
\    |  Pull and update data.ipynb
\    |  scratch.ipynb
\    |  scratch.py
|
|
\--- reports
\    \--- aapl_price_w_aapl_info
\    \--- aapl_price_w_all_price
\    \--- aapl_price_w_sector
|
|
\--- styles
\   |  custom.css
\   |  
|
|
\--- test_notebooks
\    |  dashboard_test.ipynb
\    |  Firebase Test.ipynb
\    |  model_scratch_testing.ipynb
\    |  Prediction_testing.ipynb
|
|  .gitignore
|  main.ipynb
|  presentation.pdf
|  Pull and clean data.ipynb
|  Readme.ipynb
|  README.md
|  run_tests.py
|  todo.txt
|  tune.py
```