rm quent/*.cpython*
rm quent/*.c
rm quent/*.html
rm quent/*/*.cpython*
rm quent/*/*.c
rm quent/*/*.html
rm .coverage
rm -rf htmlcov
python cython_setup.py $1 build_ext --inplace
