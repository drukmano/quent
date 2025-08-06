rm -f quent/*.cpython*
rm -f quent/*.c
rm -f quent/*.html
rm -f quent/*/*.cpython*
rm -f quent/*/*.c
rm -f quent/*/*.html
rm -f .coverage
rm -rf htmlcov
python cython_setup.py $1 build_ext --inplace
