python cython_setup.py build_ext --inplace
python3 -m build
rm dist/quent-*.whl
python3 -m twine upload dist/*
